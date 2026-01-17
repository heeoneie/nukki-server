# 학습 및 결과 확인
# ===========================================
# 이 스크립트는 task2.py에서 만든 SimpleCNN 모델을
# 합성 데이터로 학습시키고 결과를 시각화합니다.
# ===========================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# task2.py에서 정의한 SimpleCNN 모델을 가져옴
# 이게 바로 내가 만든 3-layer CNN 모델!
from task2 import SimpleCNN


def create_synthetic_data(n_samples=100):
    """
    합성 학습 데이터 생성 함수

    실제 데이터셋이 없어도 학습을 테스트할 수 있도록
    랜덤한 위치에 빨간 원이 있는 이미지와 정답 마스크를 생성

    Args:
        n_samples: 생성할 샘플 개수

    Returns:
        images: (N, 3, 320, 320) 형태의 이미지 텐서
        masks: (N, 1, 320, 320) 형태의 마스크 텐서
    """
    images = []
    masks = []

    for _ in range(n_samples):
        # 1) 랜덤 노이즈 배경 생성 (어두운 배경, 0~0.3 범위)
        img = np.random.rand(320, 320, 3).astype(np.float32) * 0.3

        # 2) 정답 마스크 초기화 (모두 0 = 배경)
        mask = np.zeros((320, 320), dtype=np.float32)

        # 3) 랜덤 위치(cx, cy)와 랜덤 반지름(r)으로 원 정의
        cx, cy = np.random.randint(50, 270, 2)  # 중심 좌표
        r = np.random.randint(30, 80)            # 반지름

        # 4) 원 영역 계산 (수학적으로 원의 방정식 사용)
        #    (x - cx)^2 + (y - cy)^2 <= r^2 이면 원 내부
        y, x = np.ogrid[:320, :320]
        circle = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

        # 5) 원 영역에 빨간색 칠하기 (R=1, G=0, B=0)
        img[circle] = [1.0, 0.0, 0.0]

        # 6) 원 영역의 마스크를 1로 설정 (전경)
        mask[circle] = 1.0

        # 7) NumPy -> PyTorch 텐서 변환
        #    이미지: (H, W, C) -> (C, H, W) 로 차원 변경 (PyTorch 형식)
        images.append(torch.tensor(img).permute(2, 0, 1))

        #    마스크: (H, W) -> (1, H, W) 로 채널 차원 추가
        masks.append(torch.tensor(mask).unsqueeze(0))

    # 8) 리스트를 하나의 배치 텐서로 합침
    #    torch.stack: 새로운 차원(배치)을 추가하며 합침
    return torch.stack(images), torch.stack(masks)


# ===========================================
# 데이터 생성 단계
# ===========================================
print("데이터 생성 중...")
train_images, train_masks = create_synthetic_data(200)
print(f"Images: {train_images.shape}, Masks: {train_masks.shape}")
# 출력: Images: torch.Size([200, 3, 320, 320])  <- 200장, RGB, 320x320
#       Masks: torch.Size([200, 1, 320, 320])   <- 200장, 1채널, 320x320


# ===========================================
# 모델, 손실함수, 옵티마이저 설정
# ===========================================

# 내가 만든 SimpleCNN 모델 인스턴스 생성
model = SimpleCNN()

# 손실 함수: BCEWithLogitsLoss (Binary Cross Entropy + Sigmoid)
# - 이진 분류(배경 vs 전경)에 적합
# - 모델 출력에 sigmoid를 자동으로 적용해줌
criterion = nn.BCEWithLogitsLoss()

# 옵티마이저: Adam
# - model.parameters(): 모델의 모든 학습 가능한 가중치
# - lr=0.001: 학습률 (한 번에 얼마나 가중치를 조정할지)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ===========================================
# 학습 루프 (Training Loop)
# ===========================================
epochs = 20      # 전체 데이터를 20번 반복 학습
batch_size = 16  # 한 번에 16장씩 처리

print("\n학습 시작...")
for epoch in range(epochs):
    # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등이 학습용으로 동작)
    model.train()
    total_loss = 0

    # 미니배치 단위로 학습 (0, 16, 32, 48, ... 인덱스로 슬라이싱)
    for i in range(0, len(train_images), batch_size):
        # 현재 배치 추출
        batch_x = train_images[i : i + batch_size]  # 입력 이미지
        batch_y = train_masks[i : i + batch_size]   # 정답 마스크

        # 1) 이전 배치의 gradient 초기화 (안 하면 누적됨)
        optimizer.zero_grad()

        # 2) 순전파 (Forward): 입력 -> 모델 -> 예측값
        output = model(batch_x)

        # 3) 손실 계산: 예측값과 정답 비교
        loss = criterion(output, batch_y)

        # 4) 역전파 (Backward): 손실에서 각 가중치의 gradient 계산
        loss.backward()

        # 5) 가중치 업데이트: gradient 방향으로 가중치 조정
        optimizer.step()

        total_loss += loss.item()

    # 에포크마다 손실 출력 (점점 줄어들면 학습이 잘 되는 것!)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ===========================================
# 모델 저장
# ===========================================
# state_dict(): 모델의 모든 가중치를 딕셔너리로 반환
# 이렇게 저장하면 나중에 load_state_dict()로 불러올 수 있음
torch.save(model.state_dict(), "trained_model.pth")
print("\n모델 저장 완료: trained_model.pth")


# ===========================================
# 테스트 (추론) 단계
# ===========================================
print("\n테스트 중...")

# 평가 모드로 전환 (Dropout 비활성화, BatchNorm 고정 등)
model.eval()

# 테스트용 새 데이터 3개 생성 (학습에 사용 안 한 데이터)
test_images, test_masks = create_synthetic_data(3)

# torch.no_grad(): gradient 계산 비활성화
# - 추론할 때는 역전파가 필요 없으므로 메모리 절약
with torch.no_grad():
    # 모델 출력에 sigmoid 적용하여 0~1 확률값으로 변환
    # (학습 때는 BCEWithLogitsLoss가 내부적으로 sigmoid 적용)
    predictions = torch.sigmoid(model(test_images))


# ===========================================
# 결과 시각화
# ===========================================
# 3행 3열 그래프 생성
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i in range(3):
    # 1열: 입력 이미지
    # permute(1,2,0): (C,H,W) -> (H,W,C) matplotlib 형식으로 변환
    axes[i, 0].imshow(test_images[i].permute(1, 2, 0))
    axes[i, 0].set_title("Input")
    axes[i, 0].axis("off")

    # 2열: 정답 마스크 (Ground Truth)
    # squeeze(): (1,H,W) -> (H,W) 불필요한 차원 제거
    axes[i, 1].imshow(test_masks[i].squeeze(), cmap="gray")
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis("off")

    # 3열: 모델 예측 결과
    # 학습이 잘 됐다면 Ground Truth와 비슷해야 함!
    axes[i, 2].imshow(predictions[i].squeeze(), cmap="gray")
    axes[i, 2].set_title("Prediction")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("training_result.png")
plt.show()
print("결과 저장 완료: training_result.png")
