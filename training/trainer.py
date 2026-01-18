import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from .dataset import SegmentationDataset, AugmentedSegmentationDataset


class ModelTrainer:
    """모델 학습 클래스"""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model: 학습할 모델 (None이면 U2NET 사용)
            device: 학습 디바이스 ('cuda' 또는 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if model is None:
            self.model = self._load_default_model()
        else:
            self.model = model

        self.model = self.model.to(self.device)

        # 손실 함수 및 옵티마이저
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )

        # 학습 상태
        self.current_epoch = 0
        self.best_loss = float('inf')

    def _load_default_model(self) -> nn.Module:
        """기본 U2NET 모델 로드"""
        try:
            # U2NET 모델 (간단한 버전 - 실제로는 rembg의 u2net 구조 사용)
            from torchvision.models.segmentation import deeplabv3_resnet50
            model = deeplabv3_resnet50(pretrained=True)

            # 출력 레이어를 1채널로 수정 (배경 제거)
            model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
            return model

        except Exception as e:
            print(f"Failed to load default model: {e}")
            raise

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        한 에포크 학습

        Args:
            train_loader: 학습 데이터 로더
            epoch: 현재 에포크

        Returns:
            평균 손실값
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # DeepLabV3의 경우 'out' 키 사용
            if isinstance(outputs, dict):
                outputs = outputs['out']

            # 손실 계산
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 진행 상황 업데이트
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

                # 학습 상태 업데이트
                config.TRAINING_STATUS['current_epoch'] = epoch
                config.TRAINING_STATUS['loss'] = loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        검증 수행

        Args:
            val_loader: 검증 데이터 로더

        Returns:
            평균 검증 손실값
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']

                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def save_checkpoint(self, epoch: int, loss: float, filepath: str) -> None:
        """
        체크포인트 저장

        Args:
            epoch: 현재 에포크
            loss: 현재 손실값
            filepath: 저장 경로
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        체크포인트 로드

        Args:
            filepath: 체크포인트 파일 경로
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filepath}")

    def train(
        self,
        train_data_dir: str,
        val_data_dir: Optional[str] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        전체 학습 프로세스 실행

        Args:
            train_data_dir: 학습 데이터 디렉토리
            val_data_dir: 검증 데이터 디렉토리
            num_epochs: 에포크 수
            batch_size: 배치 크기

        Returns:
            학습 결과 딕셔너리
        """
        # 파라미터 설정
        num_epochs = num_epochs or config.NUM_EPOCHS
        batch_size = batch_size or config.BATCH_SIZE

        # 데이터셋 및 데이터로더 생성
        print(f"Loading training data from {train_data_dir}")
        train_dataset = AugmentedSegmentationDataset(train_data_dir, augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        val_loader = None
        if val_data_dir and Path(val_data_dir).exists():
            print(f"Loading validation data from {val_data_dir}")
            val_dataset = SegmentationDataset(val_data_dir)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )

        # 학습 상태 초기화
        config.TRAINING_STATUS['is_training'] = True
        config.TRAINING_STATUS['total_epochs'] = num_epochs
        config.TRAINING_STATUS['message'] = 'Training started'

        # 학습 루프
        train_losses = []
        val_losses = []

        try:
            for epoch in range(1, num_epochs + 1):
                print(f"\n=== Epoch {epoch}/{num_epochs} ===")
                start_time = time.time()

                # 학습
                train_loss = self.train_epoch(train_loader, epoch)
                train_losses.append(train_loss)
                print(f"Train Loss: {train_loss:.4f}")

                # 검증
                if val_loader:
                    val_loss = self.validate(val_loader)
                    val_losses.append(val_loss)
                    print(f"Val Loss: {val_loss:.4f}")
                else:
                    val_loss = train_loss

                # 체크포인트 저장
                checkpoint_path = config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(epoch, train_loss, str(checkpoint_path))

                # 최고 모델 저장
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    best_model_path = config.CUSTOM_MODEL_PATH
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Best model saved to {best_model_path}")

                epoch_time = time.time() - start_time
                print(f"Epoch time: {epoch_time:.2f}s")

            # 학습 완료
            config.TRAINING_STATUS['is_training'] = False
            config.TRAINING_STATUS['message'] = 'Training completed'

            return {
                'status': 'success',
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_loss': self.best_loss,
                'epochs_completed': num_epochs
            }

        except Exception as e:
            config.TRAINING_STATUS['is_training'] = False
            config.TRAINING_STATUS['message'] = f'Training failed: {str(e)}'
            raise
