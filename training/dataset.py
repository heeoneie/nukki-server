import os
from pathlib import Path
from typing import Tuple, Optional, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    """이미지 세그멘테이션 데이터셋 클래스"""

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (320, 320),
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 경로 (images와 masks 서브디렉토리 포함)
            image_size: 이미지 리사이즈 크기
            transform: 이미지 변환 함수
            mask_transform: 마스크 변환 함수
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'
        self.image_size = image_size

        # 이미지 파일 목록
        self.image_files = sorted([
            f for f in self.image_dir.glob('*')
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])

        if len(self.image_files) == 0:
            print(f"Warning: No images found in {self.image_dir}")

        # 변환 함수 설정
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        데이터셋에서 아이템 가져오기

        Args:
            idx: 인덱스

        Returns:
            (image_tensor, mask_tensor) 튜플
        """
        # 이미지 로드
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # 마스크 로드 (파일명이 같은 마스크 파일)
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            # .png 확장자로 시도
            mask_path = self.mask_dir / (image_path.stem + '.png')

        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')  # Grayscale
        else:
            # 마스크가 없으면 빈 마스크 생성
            mask = Image.new('L', image.size, 0)
            print(f"Warning: Mask not found for {image_path.name}, using empty mask")

        # 변환 적용
        image_tensor = self.transform(image)
        mask_tensor = self.mask_transform(mask)

        return image_tensor, mask_tensor

    def get_image_path(self, idx: int) -> str:
        """인덱스에 해당하는 이미지 파일 경로 반환"""
        return str(self.image_files[idx])


class AugmentedSegmentationDataset(SegmentationDataset):
    """데이터 증강이 포함된 세그멘테이션 데이터셋"""

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (320, 320),
        augment: bool = True
    ):
        # 증강 변환 정의
        if augment:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            mask_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor()
            ])
        else:
            transform = None
            mask_transform = None

        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            transform=transform,
            mask_transform=mask_transform
        )
