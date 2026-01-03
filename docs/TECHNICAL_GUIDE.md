# 누끼 제거 기술 가이드

## 1. 누끼 제거 원리

### 1.1 핵심 개념: 이미지 세그멘테이션 (Image Segmentation)

누끼 제거는 **이미지 세그멘테이션** 기술을 사용합니다. 이미지의 각 픽셀을 "전경(객체)"과 "배경"으로 분류하는 과정입니다.

```
원본 이미지 → AI 모델 → 세그멘테이션 마스크 → 배경 제거
[RGB 이미지]   [U2NET]   [0 or 255 픽셀]   [RGBA 이미지]
```

### 1.2 사용 중인 모델: U2NET

**U2NET (U Square Network)**는 이미지 세그멘테이션에 특화된 딥러닝 모델입니다.

#### 주요 특징:
1. **U 구조** - Encoder-Decoder 아키텍처
2. **RSU (Residual U-blocks)** - 다양한 스케일의 특징 추출
3. **경량화** - 4.7MB 모델 크기 (u2netp 버전)
4. **고성능** - 실시간 처리 가능

#### 모델 구조:
```
입력 이미지 (H x W x 3)
    ↓
┌─────────────────┐
│   Encoder       │  → 특징 추출 (점점 작아짐)
│   (6 stages)    │     512x512 → 256 → 128 → 64 → 32 → 16
└─────────────────┘
         ↓
    Bridge (중간층)
         ↓
┌─────────────────┐
│   Decoder       │  → 이미지 복원 (점점 커짐)
│   (6 stages)    │     16 → 32 → 64 → 128 → 256 → 512
└─────────────────┘
         ↓
출력 마스크 (H x W x 1)
각 픽셀: 0(배경) ~ 255(전경)
```

### 1.3 배경 제거 과정

#### Step 1: 이미지 전처리
```python
# inference/preprocessor.py 동작 방식
원본 이미지 (예: 1920x1080 JPG)
    ↓
리사이즈 (320x320) - 모델 입력 크기에 맞춤
    ↓
정규화 (0-255 → 0-1)
    ↓
텐서 변환 (PyTorch Tensor)
```

#### Step 2: 모델 추론
```python
# inference/model.py의 핵심 로직
입력 텐서 (1, 3, 320, 320)  # [배치, RGB, 높이, 너비]
    ↓
U2NET 모델 통과
    ↓
출력 마스크 (1, 1, 320, 320)  # [배치, 채널, 높이, 너비]
각 픽셀 값: 0.0 ~ 1.0 (전경일 확률)
```

#### Step 3: 후처리
```python
마스크 (0.0 ~ 1.0)
    ↓
이진화 (임계값 0.5 기준)
    ↓
원본 크기로 복원 (320x320 → 1920x1080)
    ↓
알파 채널로 적용
    ↓
최종 이미지 (RGBA)
- RGB: 원본 색상
- A(Alpha): 마스크 값 (0=투명, 255=불투명)
```

---

## 2. 학습 프로세스

### 2.1 데이터 준비

#### 필요한 데이터:
```
data/train/
  ├── images/          # 원본 이미지
  │   ├── person1.jpg  # 배경이 있는 원본
  │   ├── person2.jpg
  │   └── ...
  └── masks/           # Ground Truth 마스크
      ├── person1.png  # 흑백 이미지 (전경=255, 배경=0)
      ├── person2.png
      └── ...
```

#### 마스크 생성 방법:
1. **수동**: Photoshop, GIMP 등으로 직접 제작
2. **자동**: 기존 모델로 초벌 마스크 생성 후 보정
3. **라벨링 도구**: LabelMe, CVAT 등 사용

### 2.2 학습 알고리즘

#### 손실 함수 (Loss Function)
```python
# training/trainer.py:27
criterion = nn.BCEWithLogitsLoss()

# Binary Cross Entropy Loss
Loss = -[y * log(ŷ) + (1-y) * log(1-ŷ)]

여기서:
- y: Ground Truth 마스크 (0 or 1)
- ŷ: 모델 예측 마스크 (0 ~ 1)
```

**의미:**
- 모델이 예측한 마스크와 실제 마스크의 차이를 계산
- 차이가 클수록 손실값이 커짐
- 학습 목표: 손실값을 최소화

#### 학습 루프
```python
for epoch in range(NUM_EPOCHS):
    for batch in train_loader:
        # 1. Forward pass: 예측
        images, masks = batch
        predictions = model(images)

        # 2. 손실 계산
        loss = criterion(predictions, masks)

        # 3. Backward pass: 그래디언트 계산
        loss.backward()

        # 4. 가중치 업데이트
        optimizer.step()
```

### 2.3 실제 학습 예시

```python
# training/trainer.py 동작 흐름

1. 데이터 로드
   - SegmentationDataset이 이미지와 마스크 쌍을 로드
   - DataLoader가 배치로 묶음 (예: 8장씩)

2. 에포크마다 반복
   Epoch 1:
     Batch 1: loss = 0.543
     Batch 2: loss = 0.512
     ...
     Average loss = 0.489

   Epoch 2:
     Batch 1: loss = 0.421
     ...
     Average loss = 0.398  ← 점점 감소

3. 체크포인트 저장
   - 각 에포크마다 models/checkpoints/에 저장
   - 최고 성능 모델은 models/custom_model.pth에 저장

4. 학습 완료
   - 50 에포크 후 최종 모델 저장
   - 이 모델로 새로운 이미지의 배경 제거 가능
```

---

## 3. 추론 프로세스 (실제 사용)

### 3.1 rembg 사용 시 (기본)

```python
# inference/model.py:74-99
def _remove_background_rembg(self, input_image):
    # 1. 이미지를 바이트로 변환
    image_bytes = convert_to_bytes(input_image)

    # 2. rembg 라이브러리 호출
    output_bytes = remove(
        image_bytes,
        session=self.session,  # U2NET 모델 세션
        alpha_matting=False    # 가장자리 정제 옵션
    )

    # 3. 결과를 PIL Image로 변환
    output_image = Image.open(BytesIO(output_bytes))

    return output_image  # RGBA 이미지 반환
```

**내부 동작:**
```
입력: photo.jpg (RGB)
  ↓
rembg 내부:
  1. U2NET 모델 로드
  2. 이미지 전처리 (320x320 리사이즈)
  3. 모델 추론 (마스크 생성)
  4. 원본 크기로 복원
  5. 알파 채널 적용
  ↓
출력: photo_nobg.png (RGBA)
```

### 3.2 커스텀 모델 사용 시

```python
# inference/model.py:101-143
def _remove_background_custom(self, input_image):
    # 1. 이미지 전처리
    image = Image.open(input_image).convert('RGB')

    transform = Compose([
        Resize((320, 320)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # 2. 모델 추론
    with torch.no_grad():
        output = self.custom_model(input_tensor)
        mask = output[0, 0].cpu().numpy()  # (320, 320)

    # 3. 후처리
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask = mask.resize(image.size, Image.LANCZOS)

    # 4. 알파 채널 적용
    output_image = image.copy()
    output_image.putalpha(mask)

    return output_image
```

---

## 4. 알파 매팅 (Alpha Matting)

### 4.1 기본 vs 알파 매팅

#### 기본 방식:
```
각 픽셀: 0 (배경) 또는 255 (전경)
→ 가장자리가 거칠게 보임
```

#### 알파 매팅 방식:
```
각 픽셀: 0 ~ 255 사이 값 가능
→ 부드러운 가장자리, 머리카락 디테일 살림
```

### 4.2 알파 매팅 파라미터

```python
# app.py에서 사용 가능한 옵션
output_image = bg_remover.remove_background(
    input_image,
    alpha_matting=True,                          # 알파 매팅 활성화
    alpha_matting_foreground_threshold=240,      # 전경 임계값 (높을수록 엄격)
    alpha_matting_background_threshold=10,       # 배경 임계값 (낮을수록 엄격)
    alpha_matting_erode_size=10                  # 침식 크기
)
```

---

## 5. 전체 워크플로우 예시

### 시나리오: 사람 사진의 배경 제거

```
1. 사용자가 이미지 업로드
   POST /upload
   - file: person.jpg (1920x1080)

2. Flask 서버 처리
   ├─ 파일 저장 (uploads/xxx_person.jpg)
   ├─ BackgroundRemover 호출
   │   ├─ rembg 세션 사용 (U2NET 모델)
   │   ├─ 이미지 → 바이트 변환
   │   ├─ remove() 함수 호출
   │   │   ├─ 전처리 (320x320)
   │   │   ├─ U2NET 추론
   │   │   │   ├─ Encoder: 특징 추출
   │   │   │   ├─ Decoder: 마스크 생성
   │   │   │   └─ 출력: (320, 320, 1)
   │   │   ├─ 후처리
   │   │   │   ├─ 원본 크기 복원 (1920x1080)
   │   │   │   └─ 알파 채널 적용
   │   │   └─ 결과: RGBA 이미지
   │   └─ 바이트 → PIL Image 변환
   │
   ├─ 추가 처리 (선택사항)
   │   ├─ 리사이즈 (800x600)
   │   ├─ 자동 크롭 (투명 영역 제거)
   │   └─ 포맷 변환 (PNG)
   │
   └─ 파일 저장 및 반환
       └─ uploads/yyy_output.png

3. 사용자에게 반환
   - Content-Type: image/png
   - 배경이 제거된 이미지
```

---

## 6. 성능 최적화

### 6.1 GPU 가속

```python
# 자동 GPU 감지 (training/trainer.py:17)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU 사용 시:
- 추론 속도: ~10배 빠름
- 학습 속도: ~100배 빠름
```

### 6.2 배치 처리

```python
# 여러 이미지를 한 번에 처리
images = torch.stack([img1, img2, img3, img4])  # (4, 3, 320, 320)
outputs = model(images)  # (4, 1, 320, 320)

# 개별 처리보다 ~2배 빠름
```

### 6.3 모델 최적화

```python
# 1. 모델 양자화 (INT8)
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# 크기: 1/4 감소, 속도: 1.5배 향상

# 2. ONNX 변환
torch.onnx.export(model, dummy_input, "model.onnx")
# 다양한 플랫폼에서 실행 가능
```

---

## 7. 실전 예제

### 예제 1: 기본 배경 제거

```python
from inference import BackgroundRemover

# 모델 초기화
remover = BackgroundRemover(model_name='u2net')

# 배경 제거
output = remover.remove_background('input.jpg')
output.save('output.png')
```

### 예제 2: 고급 옵션 사용

```python
from inference import BackgroundRemover
from utils import ImageProcessor

# 배경 제거
remover = BackgroundRemover()
output = remover.remove_background(
    'portrait.jpg',
    alpha_matting=True,  # 머리카락 디테일 살림
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10
)

# 추가 가공
output = ImageProcessor.resize_with_aspect_ratio(output, (1024, 768))
output = ImageProcessor.auto_crop_transparent(output, margin=20)

# 흰색 배경 추가
final = ImageProcessor.change_background(output, (255, 255, 255))
final.save('final.jpg')
```

### 예제 3: 커스텀 모델 학습 및 사용

```python
from training import ModelTrainer
from inference import BackgroundRemover

# 1. 모델 학습
trainer = ModelTrainer()
result = trainer.train(
    train_data_dir='../data/train',
    val_data_dir='../data/val',
    num_epochs=50,
    batch_size=8
)
# 결과: models/custom_model.pth 생성

# 2. 커스텀 모델로 배경 제거
remover = BackgroundRemover(use_custom_model=True)
output = remover.remove_background('test.jpg')
output.save('result.png')
```

---

## 8. 모델 비교

| 모델 | 크기 | 속도 (CPU) | 정확도 | 용도 |
|------|------|-----------|--------|------|
| u2net | 176MB | ~2초 | 높음 | 일반 객체 |
| u2netp | 4.7MB | ~0.5초 | 중간 | 실시간 처리 |
| u2net_human_seg | 176MB | ~2초 | 매우높음 | 사람 전용 |

---

## 9. 디버깅 팁

### 마스크 시각화

```python
import matplotlib.pyplot as plt

# 원본, 마스크, 결과 비교
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original)
axes[0].set_title('Original')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mask')

axes[2].imshow(result)
axes[2].set_title('Result')

plt.show()
```

### 학습 과정 모니터링

```python
# 학습 중 손실값 시각화
import matplotlib.pyplot as plt

losses = [0.543, 0.489, 0.421, 0.398, ...]
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```
