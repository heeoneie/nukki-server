import io
import os
from pathlib import Path
from typing import Union, Optional
import numpy as np
from PIL import Image
from rembg import remove, new_session
import torch


class BackgroundRemover:
    """배경 제거 모델 클래스"""

    def __init__(self, model_name: str = 'u2net', use_custom_model: bool = False):
        """
        배경 제거 모델 초기화

        Args:
            model_name: 사용할 모델 이름 (u2net, u2netp, u2net_human_seg, etc.)
            use_custom_model: 커스텀 학습 모델 사용 여부
        """
        self.model_name = model_name
        self.use_custom_model = use_custom_model
        self.session = None
        self.custom_model = None

        # rembg 세션 초기화
        if not use_custom_model:
            self._init_rembg_session()
        else:
            self._load_custom_model()

    def _init_rembg_session(self):
        """rembg 세션 초기화"""
        try:
            self.session = new_session(self.model_name)
            print(f"✓ rembg session initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Failed to initialize rembg session: {e}")
            raise

    def _load_custom_model(self):
        """커스텀 학습 모델 로드"""
        from config import CUSTOM_MODEL_PATH

        if not CUSTOM_MODEL_PATH.exists():
            raise FileNotFoundError(f"Custom model not found at {CUSTOM_MODEL_PATH}")

        try:
            self.custom_model = torch.load(CUSTOM_MODEL_PATH)
            self.custom_model.eval()
            print(f"✓ Custom model loaded from {CUSTOM_MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            raise

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10
    ) -> Image.Image:
        """
        배경 제거 수행

        Args:
            input_image: 입력 이미지 (파일 경로, PIL Image, 또는 바이트)
            alpha_matting: 알파 매팅 사용 여부 (더 정교한 가장자리 처리)
            alpha_matting_foreground_threshold: 전경 임계값
            alpha_matting_background_threshold: 배경 임계값
            alpha_matting_erode_size: 침식 크기

        Returns:
            배경이 제거된 PIL Image (RGBA)
        """
        if self.use_custom_model:
            return self._remove_background_custom(input_image)
        else:
            return self._remove_background_rembg(
                input_image,
                alpha_matting,
                alpha_matting_foreground_threshold,
                alpha_matting_background_threshold,
                alpha_matting_erode_size
            )

    def _remove_background_rembg(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        alpha_matting: bool,
        alpha_matting_foreground_threshold: int,
        alpha_matting_background_threshold: int,
        alpha_matting_erode_size: int
    ) -> Image.Image:
        """rembg를 사용한 배경 제거"""

        # 입력 이미지 처리
        if isinstance(input_image, (str, Path)):
            with open(input_image, 'rb') as f:
                input_data = f.read()
        elif isinstance(input_image, Image.Image):
            img_byte_arr = io.BytesIO()
            input_image.save(img_byte_arr, format='PNG')
            input_data = img_byte_arr.getvalue()
        elif isinstance(input_image, bytes):
            input_data = input_image
        else:
            raise ValueError("Invalid input image type")

        # 배경 제거
        output_data = remove(
            input_data,
            session=self.session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size
        )

        # PIL Image로 변환
        output_image = Image.open(io.BytesIO(output_data))
        return output_image

    def _remove_background_custom(self, input_image: Union[str, Path, Image.Image]) -> Image.Image:
        """커스텀 모델을 사용한 배경 제거"""

        # PIL Image로 변환
        if isinstance(input_image, (str, Path)):
            image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, Image.Image):
            image = input_image.convert('RGB')
        else:
            raise ValueError("Invalid input image type for custom model")

        # 이미지 전처리
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0)

        # 추론
        with torch.no_grad():
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                self.custom_model = self.custom_model.cuda()

            output = self.custom_model(input_tensor)
            mask = output[0, 0].cpu().numpy()

        # 마스크를 원본 크기로 복원
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = mask.resize(image.size, Image.LANCZOS)

        # 알파 채널 적용
        output_image = image.copy()
        output_image.putalpha(mask)

        return output_image

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs
    ) -> str:
        """
        파일 입력/출력으로 배경 제거 수행

        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 경로
            **kwargs: remove_background에 전달할 추가 인자

        Returns:
            출력 파일 경로
        """
        output_image = self.remove_background(input_path, **kwargs)
        output_image.save(output_path, format='PNG')
        return str(output_path)
