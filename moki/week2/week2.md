---

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

---
## Task1.py

### transforms 내부 함수

1. transforms.Compose([])
    - 여러개의 변환 도구들을 하나의 리스트로 묶어 순차 실행하도록 함
    - 컨테이너 역할
2. transforms.Resize((320, 320))
   - 이미지의 가로, 세로 크기를 320픽셀로 강제 조정
   - 딥러닝 모델은 입력 데이터 크기가 일정해야 한번에 여러장의 이미지를 처리할 수 있음
3. transforms.ToTensor()
   - 데이터 타입 변환 : 일반 이미지 데이터 (PIL Image, Numpy 배열)를 PyTorch가 계산할 수 있는 Tensor 형태로 변환
   - 값의 범위 : 0~255 사이의 정수 값을 0.0~1.0 사이의 실수값으로 변환
   - 차원 순서 변경 : (높이, 너비, 채널) -> (채널, 높이, 너비)
4. transform.normalize()
   - 각 채널(R,G,B)의 픽셀 값에서 평균을 빼고 표준편차로 나눔
   - 이유 : 데이터의 분포를 중심에 맞추고 일정한 범위로 제한해 모델의 학습 속도 높임
   - 설정값 : 수백만 장의 이미지 데이터셋인 ImageNet의 통계값 / 보통 사전 학습된 모델을 사용할 때에는 이 값을 그대로 사용

### 배치처리
- task1.py 에서 배치 처리를 한다는 것의 의미가 없어보임
  - 이미지 전처리는 개별처리 (어쩔 수 없다!)
  - task1에서는 추론작업이 없어서 묶음 배치를 한번에 처리하는 작업이 없어보이는게 맞다.
  - 나중에 배치로 쓸 수 있게 준비해 둔 것

### 오류 해결
1. RuntimeError: output with shape [1, 320, 320] doesn't match the broadcast shape [3, 320, 320]
   - 일부 이미지의 타입이 '그레이스케일'(회색조) 로 1채널인 이미지
   - Normalize 함수의 mean/std가 3채널용이라 채널수가 맞지 않으면 에러 발생
   - 해결 : 이미지 객체 로드 시 .convert('RGB') 추가해 채널 수 맞춰주기

---

