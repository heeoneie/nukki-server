import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class ImagePreprocessor:
    """이미지 전처리 클래스"""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        이미지 파일 로드

        Args:
            image_path: 이미지 파일 경로

        Returns:
            numpy array 형태의 이미지
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_image_pil(image_path: str) -> Image.Image:
        """
        PIL로 이미지 파일 로드

        Args:
            image_path: 이미지 파일 경로

        Returns:
            PIL Image 객체
        """
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int],
        keep_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        이미지 리사이즈

        Args:
            image: 원본 이미지
            target_size: 목표 크기 (width, height)
            keep_aspect_ratio: 비율 유지 여부

        Returns:
            리사이즈된 이미지
        """
        if keep_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size

            # 비율 계산
            ratio = min(target_w / w, target_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)

            # 리사이즈
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 패딩 추가
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left

            return cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        이미지 정규화 (0-255 -> 0-1)

        Args:
            image: 원본 이미지

        Returns:
            정규화된 이미지
        """
        return image.astype(np.float32) / 255.0

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> None:
        """
        이미지 저장

        Args:
            image: 저장할 이미지
            output_path: 저장 경로
        """
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)

        # RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, image)
