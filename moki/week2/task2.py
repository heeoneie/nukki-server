# 간단한 신경망 만들기
import torch
import torch.nn as nn  # 신경망 모델을 구축하는 데 필요한 기본 구성요소를 제공하는 핵심 모듈

class SimpleCNN(nn.Module):
    """3-layer CNN"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)  # 1x1 conv

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # (B, 16, H, W)
        x = torch.relu(self.conv2(x))  # (B, 32, H, W)
        x = self.conv3(x)              # (B, 1, H, W)
        return x

# 모델 테스트
model = SimpleCNN()
dummy_input = torch.randn(1, 3, 320, 320)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # (1, 1, 320, 320)

# 모델 저장
torch.save(model.state_dict(), 'simple_model.pth')
print("✓ 모델 저장 완료")
