import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel, 
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor
)
from mmengine.model import BaseModule
from xtuner.registry import BUILDER # (또는 MMEngine의 MODELS)

# CLIP 정규화 파라미터 (LLaVA-1.5 기준)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


# MMEngine의 레지스트리에 이 모듈을 등록합니다.
# (MMDetection의 'MODELS' 또는 XTuner의 'BUILDER'일 수 있습니다.)
@BUILDER.register_module() 
class Mask2FormerVisualEncoder(BaseModule):
    """
    OMGSegVisualEncoder를 대체하기 위한 Mask2Former 래퍼 클래스.
    
    이 클래스는 OMG-LLaVA가 요구하는 것과 동일한 출력 형식, 즉
    (clip_feature, query_feat, attention_mask) 튜플을 반환합니다.
    """
    def __init__(self,
                 clip_model_name_or_path: str = "openai/clip-vit-large-patch14-336",
                 mask2former_model_name_or_path: str = "facebook/mask2former-swin-large-coco-panoptic",
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        print(f"Initializing Mask2FormerVisualEncoder...")
        print(f"Loading CLIP from: {clip_model_name_or_path}")
        print(f"Loading Mask2Former from: {mask2former_model_name_or_path}")

        # 1. CLIP 비전 모델 로드 (동결)
        # LLaVA의 비전 타워와 동일한 모델 사용
        self.clip_model = CLIPVisionModel.from_pretrained(
            clip_model_name_or_path,
            torch_dtype=torch.float16 # VRAM 절약을 위해 float16 사용
        )
        
        # 2. Mask2Former 모델 로드 (동결)
        self.mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            mask2former_model_name_or_path,
            torch_dtype=torch.float16 # VRAM 절약을 위해 float16 사용
        )
        
        # 3. Mask2Former용 이미지 프로세서 로드
        # 이 프로세서는 이미지를 [0, 255]의 PIL Image로 되돌리는 데 사용됩니다.
        self.mask_processor = AutoImageProcessor.from_pretrained(
            mask2former_model_name_or_path
        )
        
        # 4. 모든 모델을 동결하고 eval 모드로 설정
        self.clip_model.eval()
        self.mask2former_model.eval()
        self.requires_grad_(False)
        
        # 5. 디바이스에 맞게 정규화 텐서 준비
        self.register_buffer("clip_mean", CLIP_MEAN.view(1, 3, 1, 1))
        self.register_buffer("clip_std", CLIP_STD.view(1, 3, 1, 1))
        
        print("Mask2FormerVisualEncoder initialized and frozen.")

    def _unnormalize_for_mask2former(self, pixel_values: torch.Tensor):
        """
        CLIP에 맞게 정규화된 텐서(pixel_values)를
        Mask2Former 프로세서가 요구하는 [0, 255] 범위의
        PIL Image 리스트로 되돌립니다.
        """
        # pixel_values는 float16일 수 있으므로 float32로 계산
        pixel_values = pixel_values.to(torch.float32)
        
        # De-normalize: (tensor * std) + mean
        unnormalized_tensor = (pixel_values * self.clip_std) + self.clip_mean
        
        # [0, 1] 범위를 [0, 255] 범위로 변환
        unnormalized_tensor = unnormalized_tensor * 255.0
        
        # 텐서를 [B, C, H, W] -> [B, H, W, C]로 변경하고 CPU로 이동
        unnormalized_tensor = unnormalized_tensor.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        
        # 배치 내 각 이미지를 PIL Image 객체로 변환
        pil_images = [Image.fromarray(img) for img in unnormalized_tensor]
        return pil_images

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True):
        """
        OMG_LLaVA의 메인 forward 함수로부터 호출됩니다.
        
        Args:
            pixel_values (torch.Tensor): 
                CLIP에 맞게 이미 정규화된 이미지 텐서.
                (예: shape [B, 3, 336, 336])
            output_hidden_states (bool):
                OMG-LLaVA와의 호환성을 위해 True로 유지.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (clip_feature, query_feat, attention_mask) 튜플을 반환합니다.
        """
        
        # 1. [CLIP 특징 추출]
        # 입력 텐서를 CLIP 모델에 통과시킵니다.
        clip_outputs = self.clip_model(
            pixel_values=pixel_values.to(self.clip_model.dtype),
            output_hidden_states=True
        )
        # OMG-LLaVA는 모든 히든 스테이트를 튜플로 받기를 기대합니다.
        clip_feature = clip_outputs.hidden_states
        
        # 2. [Mask2Former 특징 추출]
        # (a) CLIP 텐서를 Mask2Former용 PIL 이미지로 되돌립니다.
        pil_images = self._unnormalize_for_mask2former(pixel_values)
        
        # (b) Mask2Former 프로세서로 이미지 전처리
        #    (Mask2Former는 자체적인 리사이징과 정규화를 수행합니다)
        mask_inputs = self.mask_processor(
            images=pil_images, 
            return_tensors="pt"
        ).to(self.mask2former_model.device)
        
        # (c) Mask2Former 모델에 통과
        mask_outputs = self.mask2former_model(
            **mask_inputs,
            output_hidden_states=True,
            output_attentions=False
        )
        
        # 3. [출력 튜플 구성]
        # (a) 쿼리 피처 (Query Features)
        #    - Mask2Former 디코더의 마지막 레이어 쿼리를 사용합니다.
        #    - Shape: (Batch, NumQueries, HiddenDim) 예: [B, 100, 256]
        query_feat = mask_outputs.decoder_hidden_states[-1]

        # (b) 어텐션 마스크 (Attention Mask)
        #    - OMG-LLaVA의 `prepare_seg_pretrain_data` 함수가 요구하는
        #      (bs, q, hw) 형태에 맞춥니다.
        #    - 원본 마스크 예측 로짓을 사용합니다.
        #    - Shape: (Batch, NumQueries, H, W)
        mask_logits = mask_outputs.masks_queries_logits
        bs, q, h, w = mask_logits.shape
        
        #    - Shape: (Batch, NumQueries, H*W)
        attention_mask = mask_logits.view(bs, q, h * w)

        # OMG-LLaVA가 기대하는 튜플 (clip_feature, query_feat, attention_mask) 반환
        return (clip_feature, query_feat, attention_mask)
