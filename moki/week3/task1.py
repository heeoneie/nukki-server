# 과제 1: 가장 간단한 세그멘테이션 모델 (Encoder-Decoder)
import torch
import torch.nn as nn


class SimpleSegNet(nn.Module):
    """Encoder-Decoder 기본 구조"""
    def __init__(self):
        super().__init__()

        # Encoder (특징 추출)
        """
        nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        - enc1: 입력 채널 3 (RGB 이미지), 출력 채널 64, 커널 크기 3x3, 패딩 1 (특징 64개 추출)
        - enc2: 입력 채널 64, 출력 채널 128, 커널 크기 3x3, 패딩 1 (특징 128개 추출)
        2x2 맥스풀링: 특성 맵 크기를 절반으로 줄임
        """
        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (복원)
        """
        scale_factor: 배율 > 32x32 -> 64x64
        mode='bilinear': 보간법 종류
        - 이미지를 키울 때 빈 공간을 어떤 값으로 채울지 결정하는 알고리즘
        - bilinear: 주변 픽셀 값을 거리 비율에 따라 평균 내어 채웁니다. 결과물이 부드럽고 자연스러움
        - nearest: 가장 가까운 픽셀 값을 복사하여 확대 (연산은 빠르지만 계단 현상 발생 가능)
        nn.Conv2d(128, 64, 3, padding=1): 특징 맵을 64채널로 줄임
        nn.Conv2d(64, 1, 1): 최종 출력 채널을 1로 줄여 이진 마스크 생성
        1x1 컨볼루션: 각 위치에서 채널 간의 선형 조합을 수행하여 채널 수를 줄임
        """
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 1)  # 최종 마스크 (1채널)

    def forward(self, x):
        # Encoder
        """
        x1 : 입력 이미지 -> enc1 -> ReLU 활성화 함수 적용
        x2 : x1 -> 맥스풀링 (크기 절반)
        x3 : x2 -> enc2 -> ReLU 활성화 함수 적용
        ----------------------------------------
        x4 : x3 -> 업샘플링 (크기 2배)
        x5 : x4 -> dec1 -> ReLU 활성화 함수 적용
        out: x5 -> dec2 -> 시그모이드 활성화 함수 적용 (출력값을 [0, 1] 범위로 변환)
        """
        x1 = torch.relu(self.enc1(x))      # (B, 64, H, W)
        x2 = self.pool(x1)                  # (B, 64, H/2, W/2)
        x3 = torch.relu(self.enc2(x2))     # (B, 128, H/2, W/2)

        # Decoder
        x4 = self.up(x3)                    # (B, 128, H, W)
        x5 = torch.relu(self.dec1(x4))     # (B, 64, H, W)
        out = torch.sigmoid(self.dec2(x5)) # (B, 1, H, W) [0~1]

        return out


if __name__ == '__main__':
    # 테스트
    model = SimpleSegNet()
    """
    배치 크기 2, 채널 3 (RGB), 높이 256, 너비 256인 더미 입력 생성
    컴퓨터 입장에서 이미지는 결국 [0~255] 혹은 [0~1] 사이의 숫자가 가득 찬 표일 뿐입니다.
    """
    x = torch.randn(2, 3, 256, 256)  # 배치=2
    mask = model(x)
    print(f"Input: {x.shape}, Output: {mask.shape}")
    # output range가 [0, 1]이면 성공 (시그모이드 통과했기 때문)
    print(f"Output range: [{mask.min():.3f}, {mask.max():.3f}]")
