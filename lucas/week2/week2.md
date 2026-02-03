
## 2단계: PyTorch 기초 (2-3주)

### 목표
PyTorch로 딥러닝 모델을 다루는 기본을 익힙니다.

### 학습 내용

**PyTorch 핵심 개념**
- Tensor 기본 연산
- torchvision transforms (Resize, Normalize, ToTensor)
- GPU/CPU 디바이스 관리
- 모델 저장/로드 (state_dict)

### 실습 과제

```python
# 과제 1: 이미지 전처리 파이프라인
import torch
from torchvision import transforms
from PIL import Image

# inference/preprocessor.py 참고
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open('test.jpg')
tensor = transform(img)
print(f"Tensor shape: {tensor.shape}")  # (3, 320, 320)
print(f"Min: {tensor.min():.2f}, Max: {tensor.max():.2f}")

# 배치 차원 추가
batch = tensor.unsqueeze(0)  # (1, 3, 320, 320)
print(f"Batch shape: {batch.shape}")
```

```python
# 과제 2: 간단한 신경망 만들기
import torch.nn as nn

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
```

```python
# 과제 3: GPU 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = SimpleCNN().to(device)
input_tensor = torch.randn(1, 3, 320, 320).to(device)

with torch.no_grad():
    output = model(input_tensor)

print(f"Output device: {output.device}")
```

### 추천 자료
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch로 시작하는 딥러닝 기초](https://wikidocs.net/book/2788)
- [모두를 위한 딥러닝 시즌2](https://deeplearningzerotoall.github.io/season2/lec_pytorch.html)

### 체크포인트
- [ ] Tensor 생성 및 연산을 할 수 있다
- [ ] transforms로 이미지 전처리를 할 수 있다
- [ ] 간단한 CNN 모델을 만들 수 있다
- [ ] 모델을 저장하고 로드할 수 있다
- [ ] GPU/CPU 디바이스를 이해하고 사용할 수 있다
