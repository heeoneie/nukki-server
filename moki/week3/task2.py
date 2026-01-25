# 과제 2: Skip Connection이 있는 U-Net
import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """Skip Connection 추가"""
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        # 입력 채널 128+64 = 업샘플링 특징맵(128) + 인코더 e1 특징맵(64) 결합
        # Skip Connection 효과: 인코더의 저수준 특징(에지, 텍스처)을 디코더가 활용 -> 정교한 복원
        self.dec1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = torch.relu(self.enc1(x))       # (B, 64, H, W)
        e2 = self.pool(e1)                   # (B, 64, H/2, W/2)
        e3 = torch.relu(self.enc2(e2))      # (B, 128, H/2, W/2)

        # Decoder with skip
        d1 = self.up(e3)                     # (B, 128, H, W)
        # Skip Connection: 인코더 e1과 디코더 d1을 채널 차원에서 concat
        # -> 디코더가 초기 특징 정보도 함께 활용하여 더 풍부한 복원 가능
        d2 = torch.cat([d1, e1], dim=1)     # (B, 192, H, W)
        d3 = torch.relu(self.dec1(d2))      # (B, 64, H, W)
        out = torch.sigmoid(self.dec2(d3))  # (B, 1, H, W)

        return out


if __name__ == '__main__':
    # 테스트
    model = SimpleUNet()
    x = torch.randn(2, 3, 256, 256)  # 배치=2
    mask = model(x)
    print(f"Input: {x.shape}, Output: {mask.shape}")
    print(f"Output range: [{mask.min():.3f}, {mask.max():.3f}]")

    # SimpleSegNet과 비교
    from task1 import SimpleSegNet
    seg_model = SimpleSegNet()

    # 파라미터 수: 많으면 복잡한 패턴 학습 가능하지만 과적합 위험 증가
    # U-Net이 일반적으로 성능 우수 (실제 성능은 데이터셋/문제에 따라 실험 필요)
    print(f"\n=== 파라미터 수 비교 ===")
    seg_params = sum(p.numel() for p in seg_model.parameters())
    unet_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleSegNet: {seg_params:,}")
    print(f"SimpleUNet: {unet_params:,}")
