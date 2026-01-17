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
- [X] Tensor 생성 및 연산을 할 수 있다
- [X] transforms로 이미지 전처리를 할 수 있다
- [X] 간단한 CNN 모델을 만들 수 있다
- [X] 모델을 저장하고 로드할 수 있다
- [X] GPU/CPU 디바이스를 이해하고 사용할 수 있다

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

## Task2

### CNN 모델이란? (합성곱 신경망)

- 이미지의 '특징'을 찾아내기 위해 특화된 인공지능

- 3단계 핵심 원리
  1. 특징 찾기 (Convolution - 합성곱)
     - 원리 : 필터를 들고 이미지 구석구석 훑음
     - 역할 : 선, 곡선, 색의 경계 같은 아주 기초적인 특징을 찾아냄
     
  2. 중요한 것만 남기기 (Pooling - 풀링)
     - 원리 : 이미지의 크기를 줄이면서 가장 강렬한 특징만 남김
     - 역할 : 정보의 양은 줄이고(압축), 사물이 약간 옆으로 이동하거나 회전해도 '이건 여전히 강아지야'라고 인식할 수 있게 함

  3. 결론 내기 (Fully Connected Layer)
     - 원리 : 위에서 뽑아낸 수많은 특징(털, 귀 모양, 꼬리 등)을 다 모아서 종합
     - 역할 : 이 특징들을 다 합쳐보니 98% 확률로 강아지다 라고 최종 판정

- 누끼 따기(배경 제거)와 CNN의 관계
  - 일반적인 CNN에서 한단계 더 나아간 '이미지 세그멘테이션' 기술을 사용
  - 어디가 물체이고 어디가 배경인지 정밀하게 지도를 그리는 작업

---
### 파이썬 Class

```python
class SimpleCNN(nn.Module):
```
- nn.Module 인공지능 기본 틀을 빌려와서 SimpleCNN 이라는 이름의 새로운 설계도를 생성할 것
- __init__ : 클래스 생성 시 자동으로 실행되는 함수
- self : 나 자신을 의미함
- nn.Conv2d : 이미지에서 특징 추출하는 합성곱 기계
- 3 : 입력 채널 수. 컬러 이미지(RGB)라서 3
- 16 : 출력 채널 수. 이 기계를 통과하면 16개의 서로 다른 특징 지도가 만들어진다는 의미
- 3 : 필터의 크기가 3x3 이라는 뜻 (보통 채널의 수와 동일함)
  - 2면 2x2 / 최근 딥러닝에서는 3x3을 가장 선호함 (섬세함)
- padding=1 : 이미지 테두리에 가짜 픽셀을 한줄 둘러서, 기계를 통과한 후에도 이미지 크기가 줄어들지 않게 방어

```python
self.conv3 = nn.Conv2d(32, 1, 1) # 1x1 conv
```
- 1 : 마지막 출력 채널은 1. 누끼 따기에서 최종 결과물은 물체인지(1), 배경인지(0)를 나타내는 지도 1장을 필요로 함

---

### 모델 내부에서 데이터가 흐를 때 왜 ReLU라는 필터링 과정을 거칠까

1. 배경제거 : 필터가 훑은 값 중 배경이라고 판단되는 부분(마이너스 값)을 0으로 확실히 밀어버림
2. 형태 인식 : 모델이 단순히 직선적인 색상 차이가 아니라, 물체의 복잡한 테두리 곡선을 배울 수 있게 '유연성'을 부여

---
### 궁금해서 만들어본 실제 학습 및 결과 (train_and_test.py)

  ┌─────────────────────────────────────────────────────────┐
  │  1. 데이터 생성                                            │
  │     create_synthetic_data(200) → 빨간 원 이미지 + 마스크      │
  └─────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────┐
  │  2. 모델/손실함수/옵티마이저 준비                               │
  │     SimpleCNN() ← task2.py에서 내가 만든 모델!               │
  │     BCEWithLogitsLoss() ← 이진 분류용 손실함수                │
  │     Adam(lr=0.001) ← 가중치 업데이트 방법                     │
  └─────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────┐
  │  3. 학습 루프 (20 에포크)                                   │
  │     for each batch:                                     │
  │       ① zero_grad()  ← gradient 초기화                    │
  │       ② output = model(x)  ← 순전파                       │
  │       ③ loss = criterion(output, y)  ← 손실 계산          │
  │       ④ loss.backward()  ← 역전파                         │
  │       ⑤ optimizer.step()  ← 가중치 업데이트                 │
  └─────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────┐
  │  4. 저장 & 테스트                                          │
  │     torch.save() → trained_model.pth                    │
  │     model.eval() + torch.no_grad() → 추론                │
  │     시각화 → training_result.png                          │
  └─────────────────────────────────────────────────────────┘
---
### Mac에는 GPU가 없다용
