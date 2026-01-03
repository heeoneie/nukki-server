import io
from pathlib import Path
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    """이미지 가공 유틸리티 클래스"""

    @staticmethod
    def resize_with_aspect_ratio(
        image: Union[Image.Image, np.ndarray],
        target_size: Tuple[int, int],
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> Image.Image:
        """
        비율을 유지하면서 이미지 리사이즈 (패딩 추가)

        Args:
            image: PIL Image 또는 numpy array
            target_size: 목표 크기 (width, height)
            background_color: 패딩 배경색 (R, G, B, A)

        Returns:
            리사이즈된 PIL Image
        """
        # numpy array를 PIL Image로 변환
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        # 원본 비율 계산
        original_width, original_height = image.size
        target_width, target_height = target_size

        # 비율 계산
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # 리사이즈
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # 새 이미지 생성 (패딩 포함)
        new_image = Image.new('RGBA', target_size, background_color)

        # 중앙 배치
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(resized_image, (paste_x, paste_y))

        return new_image

    @staticmethod
    def resize_exact(
        image: Union[Image.Image, np.ndarray],
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        정확한 크기로 이미지 리사이즈 (비율 무시)

        Args:
            image: PIL Image 또는 numpy array
            target_size: 목표 크기 (width, height)

        Returns:
            리사이즈된 PIL Image
        """
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        return image.resize(target_size, Image.LANCZOS)

    @staticmethod
    def add_padding(
        image: Union[Image.Image, np.ndarray],
        padding: Union[int, Tuple[int, int, int, int]],
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> Image.Image:
        """
        이미지에 패딩 추가

        Args:
            image: PIL Image 또는 numpy array
            padding: 패딩 크기 (전체) 또는 (top, right, bottom, left)
            background_color: 패딩 배경색 (R, G, B, A)

        Returns:
            패딩이 추가된 PIL Image
        """
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        # 패딩 값 처리
        if isinstance(padding, int):
            top = right = bottom = left = padding
        else:
            top, right, bottom, left = padding

        # 새 이미지 크기
        width, height = image.size
        new_width = width + left + right
        new_height = height + top + bottom

        # 새 이미지 생성
        new_image = Image.new('RGBA', (new_width, new_height), background_color)
        new_image.paste(image, (left, top))

        return new_image

    @staticmethod
    def crop_image(
        image: Union[Image.Image, np.ndarray],
        crop_box: Tuple[int, int, int, int]
    ) -> Image.Image:
        """
        이미지 크롭

        Args:
            image: PIL Image 또는 numpy array
            crop_box: 크롭 영역 (left, top, right, bottom)

        Returns:
            크롭된 PIL Image
        """
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        return image.crop(crop_box)

    @staticmethod
    def auto_crop_transparent(
        image: Image.Image,
        margin: int = 0
    ) -> Image.Image:
        """
        투명 영역을 제외하고 자동 크롭

        Args:
            image: PIL Image (RGBA)
            margin: 추가 여백

        Returns:
            크롭된 PIL Image
        """
        # 알파 채널 확인
        if image.mode != 'RGBA':
            return image

        # 알파 채널 추출
        alpha = np.array(image.split()[-1])

        # 0이 아닌 픽셀 찾기
        rows = np.any(alpha, axis=1)
        cols = np.any(alpha, axis=0)

        if not rows.any() or not cols.any():
            # 완전히 투명한 이미지
            return image

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # 마진 추가
        row_min = max(0, row_min - margin)
        row_max = min(image.height, row_max + margin + 1)
        col_min = max(0, col_min - margin)
        col_max = min(image.width, col_max + margin + 1)

        return image.crop((col_min, row_min, col_max, row_max))

    @staticmethod
    def change_background(
        image: Image.Image,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        투명 배경을 특정 색상으로 변경

        Args:
            image: PIL Image (RGBA)
            background_color: 배경색 (R, G, B)

        Returns:
            배경이 변경된 PIL Image (RGB)
        """
        if image.mode != 'RGBA':
            return image.convert('RGB')

        # 새 배경 생성
        background = Image.new('RGB', image.size, background_color)
        background.paste(image, mask=image.split()[3])  # 알파 채널을 마스크로 사용

        return background

    @staticmethod
    def save_image(
        image: Image.Image,
        output_path: Union[str, Path],
        format: Optional[str] = None,
        quality: int = 95
    ) -> str:
        """
        이미지 저장

        Args:
            image: PIL Image
            output_path: 저장 경로
            format: 이미지 포맷 (None이면 확장자에서 추론)
            quality: JPEG 품질 (1-100)

        Returns:
            저장된 파일 경로
        """
        output_path = Path(output_path)

        if format is None:
            format = output_path.suffix[1:].upper()
            if format == 'JPG':
                format = 'JPEG'

        # JPEG는 투명도를 지원하지 않음
        if format == 'JPEG' and image.mode == 'RGBA':
            image = ImageProcessor.change_background(image)

        image.save(output_path, format=format, quality=quality)
        return str(output_path)

    @staticmethod
    def image_to_bytes(
        image: Image.Image,
        format: str = 'PNG',
        quality: int = 95
    ) -> bytes:
        """
        PIL Image를 바이트로 변환

        Args:
            image: PIL Image
            format: 이미지 포맷
            quality: JPEG 품질 (1-100)

        Returns:
            이미지 바이트
        """
        img_byte_arr = io.BytesIO()

        # JPEG는 투명도를 지원하지 않음
        if format.upper() == 'JPEG' and image.mode == 'RGBA':
            image = ImageProcessor.change_background(image)

        image.save(img_byte_arr, format=format, quality=quality)
        return img_byte_arr.getvalue()

    @staticmethod
    def bytes_to_image(image_bytes: bytes) -> Image.Image:
        """
        바이트를 PIL Image로 변환

        Args:
            image_bytes: 이미지 바이트

        Returns:
            PIL Image
        """
        return Image.open(io.BytesIO(image_bytes))

    @staticmethod
    def apply_filter(
        image: Union[Image.Image, np.ndarray],
        filter_type: str = 'blur',
        **kwargs
    ) -> Image.Image:
        """
        이미지 필터 적용

        Args:
            image: PIL Image 또는 numpy array
            filter_type: 필터 타입 ('blur', 'sharpen', 'edge')
            **kwargs: 필터별 추가 파라미터

        Returns:
            필터가 적용된 PIL Image
        """
        from PIL import ImageFilter

        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        if filter_type == 'blur':
            radius = kwargs.get('radius', 2)
            return image.filter(ImageFilter.GaussianBlur(radius))
        elif filter_type == 'sharpen':
            return image.filter(ImageFilter.SHARPEN)
        elif filter_type == 'edge':
            return image.filter(ImageFilter.FIND_EDGES)
        else:
            return image
