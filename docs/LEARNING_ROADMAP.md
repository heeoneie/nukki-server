# 🎓 배경 제거 모델 직접 구현하기 - 학습 로드맵

U2NET 대신 직접 세그멘테이션 모델을 구현하여 이 서비스를 완성하기 위한 단계별 학습 가이드입니다.

## 📚 전체 학습 기간: 약 10-14주

```
1단계: 기초 다지기 (1-2주)
2단계: PyTorch 기초 (2-3주)
3단계: 세그멘테이션 기초 (2-3주)
4단계: 학습 코드 이해 (1-2주)
5단계: U2NET 구현 (3-4주)
6단계: 모델 통합 (1주)
7단계: 실험 및 개선 (지속적)
```

---

## 1단계: 기초 다지기 (1-2주)

### 목표
이미지를 코드로 다루는 기본기를 익힙니다.

### 학습 내용

**이미지 처리 기초**
- PIL/Pillow: 이미지 로드, 저장, 변환 (RGB ↔ RGBA)
- NumPy: 배열 연산, 이미지를 배열로 다루기
- OpenCV 기초: resize, crop, 색상 공간 변환

### 실습 과제

```python
# 과제 1: utils/image_processing.py 이해하기
from PIL import Image
import numpy as np

# 1. 이미지를 NumPy 배열로 변환
img = Image.open('test.jpg')
img_array = np.array(img)
print(f"Shape: {img_array.shape}")  # (H, W, 3)

# 2. 알파 채널 직접 만들기
height, width = img_array.shape[:2]
mask = np.zeros((height, width), dtype=np.uint8)

# 중앙 사각형만 255 (전경)
mask[100:200, 100:200] = 255

# 3. RGBA로 변환
img_rgba = Image.fromarray(img_array).convert('RGBA')
img_rgba.putalpha(Image.fromarray(mask))
img_rgba.save('output.png')

print("✓ 알파 채널이 적용된 이미지 저장 완료!")
```

```python
# 과제 2: 이미지 리사이즈와 패딩
from utils import ImageProcessor

# 본인의 이미지로 테스트
test_img = Image.open('my_photo.jpg')

# 비율 유지하며 리사이즈
resized = ImageProcessor.resize_with_aspect_ratio(
    test_img,
    target_size=(512, 512),
    background_color=(255, 255, 255, 0)  # 투명 배경
)

# 패딩 추가
padded = ImageProcessor.add_padding(resized, padding=50)
padded.save('result.png')
```

### 체크포인트
- [ ] PIL로 이미지를 열고 저장할 수 있다
- [ ] NumPy 배열과 PIL Image를 변환할 수 있다
- [ ] 알파 채널의 개념을 이해했다
- [ ] utils/image_processing.py의 모든 함수를 실행해봤다

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

## 3단계: 세그멘테이션 기초 이해 (2-3주)

### 목표
이미지 세그멘테이션이 무엇인지, U-Net 구조를 이해합니다.

### 학습 내용

**필독 문서**
- `HOW_IT_WORKS.md` - 누끼 제거 동작 원리
- `TECHNICAL_GUIDE.md` - 기술 가이드

**핵심 개념**
- 세그멘테이션: 픽셀 단위 분류
- Encoder-Decoder 구조
- Skip Connection의 역할
- BCE (Binary Cross Entropy) 손실 함수

### 실습 과제

```python
# 과제 1: 가장 간단한 세그멘테이션 모델
import torch
import torch.nn as nn

class SimpleSegNet(nn.Module):
    """Encoder-Decoder 기본 구조"""
    def __init__(self):
        super().__init__()

        # Encoder (특징 추출)
        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (복원)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 1)  # 최종 마스크 (1채널)

    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.enc1(x))      # (B, 64, H, W)
        x2 = self.pool(x1)                  # (B, 64, H/2, W/2)
        x3 = torch.relu(self.enc2(x2))     # (B, 128, H/2, W/2)

        # Decoder
        x4 = self.up(x3)                    # (B, 128, H, W)
        x5 = torch.relu(self.dec1(x4))     # (B, 64, H, W)
        out = torch.sigmoid(self.dec2(x5)) # (B, 1, H, W) [0~1]

        return out

# 테스트
model = SimpleSegNet()
x = torch.randn(2, 3, 256, 256)  # 배치=2
mask = model(x)
print(f"Input: {x.shape}, Output: {mask.shape}")
print(f"Output range: [{mask.min():.3f}, {mask.max():.3f}]")
```

```python
# 과제 2: Skip Connection이 있는 U-Net
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
        self.dec1 = nn.Conv2d(128 + 64, 64, 3, padding=1)  # skip connection
        self.dec2 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = torch.relu(self.enc1(x))       # (B, 64, H, W)
        e2 = self.pool(e1)                   # (B, 64, H/2, W/2)
        e3 = torch.relu(self.enc2(e2))      # (B, 128, H/2, W/2)

        # Decoder with skip
        d1 = self.up(e3)                     # (B, 128, H, W)
        d2 = torch.cat([d1, e1], dim=1)     # (B, 192, H, W) - skip!
        d3 = torch.relu(self.dec1(d2))      # (B, 64, H, W)
        out = torch.sigmoid(self.dec2(d3))  # (B, 1, H, W)

        return out
```

```python
# 과제 3: BCE 손실 함수 이해
import torch.nn.functional as F

# 가상의 예측과 정답
prediction = torch.tensor([[0.9, 0.1], [0.3, 0.7]])  # 모델 예측
target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])      # 정답 마스크

# BCE Loss 계산
loss = F.binary_cross_entropy(prediction, target)
print(f"BCE Loss: {loss.item():.4f}")

# 직접 계산해보기
# Loss = -[y*log(p) + (1-y)*log(1-p)]
manual_loss = -(target * torch.log(prediction) +
                (1 - target) * torch.log(1 - prediction))
print(f"Manual calculation: {manual_loss.mean().item():.4f}")
```

### 추천 자료
- **논문**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **블로그**: [Image Segmentation 이해하기](https://medium.com/@msmapark2)
- **영상**: YouTube "U-Net 설명" 검색

### 체크포인트
- [ ] Encoder-Decoder 구조를 이해했다
- [ ] Skip Connection의 목적을 안다
- [ ] SimpleSegNet을 구현했다
- [ ] BCE 손실 함수를 이해했다
- [ ] U-Net 논문을 읽었다

---

## 4단계: 학습 코드 이해 (1-2주)

### 목표
현재 코드베이스의 학습 파이프라인을 완전히 이해합니다.

### 학습 내용

**분석할 파일**
- `training/dataset.py` - 데이터 로딩
- `training/trainer.py` - 학습 루프
- `config.py` - 설정 관리

### 실습 과제

```python
# 과제 1: 데이터셋 만들기
# 1. 이미지 5-10장 준비
#    - data/train/images/img1.jpg, img2.jpg, ...
#
# 2. 마스크 만들기 (Photoshop, GIMP, Paint.NET 등)
#    - data/train/masks/img1.png, img2.png, ...
#    - 흑백 이미지 (전경=255, 배경=0)
#
# 3. 데이터셋 로드 테스트

from training import SegmentationDataset
from torch.utils.data import DataLoader

dataset = SegmentationDataset('../data/train')
print(f"Dataset size: {len(dataset)}")

# 첫 번째 샘플 확인
image, mask = dataset[0]
print(f"Image shape: {image.shape}")  # (3, H, W)
print(f"Mask shape: {mask.shape}")  # (1, H, W)

# DataLoader로 배치 만들기
loader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch_imgs, batch_masks in loader:
    print(f"Batch images: {batch_imgs.shape}")
    print(f"Batch masks: {batch_masks.shape}")
    break
```

```python
# 과제 2: 학습 루프 직접 작성
import torch
import torch.nn as nn
import torch.optim as optim
from training import SegmentationDataset
from torch.utils.data import DataLoader

# 1. 모델, 손실 함수, 옵티마이저
model = SimpleUNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. 데이터 로더
dataset = SegmentationDataset('data/train')
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. 학습 루프
model.train()
for epoch in range(5):
    total_loss = 0

    for images, masks in loader:
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}")

# 4. 모델 저장
torch.save(model.state_dict(), 'my_first_model.pth')
print("✓ 모델 저장 완료!")
```

```python
# 과제 3: ModelTrainer 사용해보기
from training import ModelTrainer

trainer = ModelTrainer()
result = trainer.train(
    train_data_dir='data/train',
    num_epochs=10,
    batch_size=2
)

print(f"학습 완료: {result}")
```

### 체크포인트
- [ ] 데이터셋을 직접 만들었다 (최소 5장)
- [ ] DataLoader를 사용할 수 있다
- [ ] 학습 루프를 이해하고 직접 작성했다
- [ ] ModelTrainer로 학습을 실행했다
- [ ] 체크포인트 저장/로드를 이해했다

---

## 5단계: U2NET 구조 이해 및 구현 (3-4주)

### 목표
U2NET의 핵심 아이디어를 이해하고 직접 구현합니다.

### 학습 내용

**U2NET의 핵심**
1. **RSU (Residual U-block)**: U-Net 안에 작은 U-Net
2. **Multi-scale**: 다양한 스케일에서 특징 추출
3. **Deep Supervision**: 여러 출력에서 손실 계산

### 구현 단계

#### 5-1. RSU 블록 구현

```python
# models/rsu.py
import torch
import torch.nn as nn

class RSU(nn.Module):
    """Residual U-block"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, depth=5):
        """
        Args:
            in_ch: 입력 채널
            mid_ch: 중간 채널
            out_ch: 출력 채널
            depth: U-block 깊이
        """
        super().__init__()
        self.depth = depth

        # Input conv
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        for i in range(depth):
            if i == 0:
                self.encoder_convs.append(self._make_conv_block(out_ch, mid_ch))
            else:
                self.encoder_convs.append(self._make_conv_block(mid_ch, mid_ch))

            if i < depth - 1:
                self.encoder_pools.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        # Bottleneck
        self.bottleneck = self._make_conv_block(mid_ch, mid_ch, dilation=2)

        # Decoder
        self.decoder_convs = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for i in range(depth - 1):
            self.decoder_upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            self.decoder_convs.append(self._make_conv_block(mid_ch * 2, mid_ch))

        # Output
        self.output_conv = self._make_conv_block(mid_ch * 2, out_ch)

    def _make_conv_block(self, in_ch, out_ch, dilation=1):
        """Conv-BN-ReLU 블록"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input
        hx = self.input_conv(x)

        # Encoder
        enc_features = []
        hxin = hx

        for i in range(self.depth):
            hxin = self.encoder_convs[i](hxin)
            enc_features.append(hxin)
            if i < self.depth - 1:
                hxin = self.encoder_pools[i](hxin)

        # Bottleneck
        hxin = self.bottleneck(hxin)

        # Decoder
        for i in range(self.depth - 1, 0, -1):
            hxin = self.decoder_upsamples[i - 1](hxin)
            hxin = torch.cat([hxin, enc_features[i - 1]], dim=1)
            hxin = self.decoder_convs[i - 1](hxin)

        # Output with residual
        hxin = torch.cat([hxin, hx], dim=1)
        out = self.output_conv(hxin)

        return out + x  # Residual connection

# 테스트
if __name__ == '__main__':
    rsu = RSU(in_ch=3, mid_ch=12, out_ch=64, depth=5)
    x = torch.randn(1, 3, 256, 256)
    out = rsu(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
```

#### 5-2. U2NET 전체 구조

```python
# models/u2net.py
import torch
import torch.nn as nn
from .rsu import RSU

class U2NET(nn.Module):
    """U2NET 전체 구조"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        # Encoder
        self.stage1 = RSU(in_ch, 32, 64, depth=7)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU(64, 32, 128, depth=6)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU(128, 64, 256, depth=5)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU(256, 128, 512, depth=4)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU(512, 256, 512, depth=4)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Bridge
        self.stage6 = RSU(512, 256, 512, depth=4)

        # Decoder
        self.stage5d = RSU(1024, 256, 512, depth=4)
        self.stage4d = RSU(1024, 128, 256, depth=4)
        self.stage3d = RSU(512, 64, 128, depth=5)
        self.stage2d = RSU(256, 32, 64, depth=6)
        self.stage1d = RSU(128, 16, 64, depth=7)

        # Side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # Output fusion
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool3(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool4(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool5(hx5)

        # Bridge
        hx6 = self.stage6(hx)

        # Decoder
        hx5d = self.stage5d(torch.cat([hx6, hx5], dim=1))
        hx4d = self.stage4d(torch.cat([hx5d, hx4], dim=1))
        hx3d = self.stage3d(torch.cat([hx4d, hx3], dim=1))
        hx2d = self.stage2d(torch.cat([hx3d, hx2], dim=1))
        hx1d = self.stage1d(torch.cat([hx2d, hx1], dim=1))

        # Side outputs
        d1 = self.side1(hx1d)
        d2 = self._upsample_like(self.side2(hx2d), d1)
        d3 = self._upsample_like(self.side3(hx3d), d1)
        d4 = self._upsample_like(self.side4(hx4d), d1)
        d5 = self._upsample_like(self.side5(hx5d), d1)
        d6 = self._upsample_like(self.side6(hx6), d1)

        # Fusion
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), \
               torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), \
               torch.sigmoid(d6)

    def _upsample_like(self, src, tar):
        """src를 tar와 같은 크기로 upsample"""
        return nn.functional.interpolate(src, size=tar.shape[2:],
                                        mode='bilinear', align_corners=False)

# 테스트
if __name__ == '__main__':
    model = U2NET(in_ch=3, out_ch=1)
    x = torch.randn(1, 3, 320, 320)
    outputs = model(x)
    print(f"Main output: {outputs[0].shape}")
    print(f"Side outputs: {[o.shape for o in outputs[1:]]}")
```

#### 5-3. Multi-output 손실 함수

```python
# models/loss.py
import torch
import torch.nn as nn

class MultiScaleBCELoss(nn.Module):
    """여러 출력에 대한 BCE Loss"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, outputs, target):
        """
        Args:
            outputs: tuple of (d0, d1, d2, d3, d4, d5, d6)
            target: ground truth mask
        """
        loss = 0.0

        for output in outputs:
            loss += self.bce(output, target)

        return loss
```

### 추천 자료
- **논문**: [U²-Net: Going Deeper with Nested U-Structure](https://arxiv.org/abs/2005.09007)
- **공식 구현**: https://github.com/xuebinqin/U-2-Net
- **설명 영상**: YouTube "U2Net explained"

### 체크포인트
- [ ] RSU 블록을 이해하고 구현했다
- [ ] U2NET 전체 구조를 구현했다
- [ ] Multi-scale 출력을 이해했다
- [ ] 모델 테스트를 통과했다

---

## 6단계: 모델 통합 (1주)

### 목표
구현한 모델을 실제 서비스에 통합합니다.

### 실습 과제

```python
# 과제 1: 모델 학습
from models.u2net import U2NET
from models.loss import MultiScaleBCELoss
from training.dataset import SegmentationDataset
from torch.utils.data import DataLoader
import torch.optim as optim

# 모델, 손실 함수, 옵티마이저
model = U2NET(in_ch=3, out_ch=1)
criterion = MultiScaleBCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터
dataset = SegmentationDataset('data/train')
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 학습
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

for epoch in range(20):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(loader):.4f}")

# 저장
torch.save(model.state_dict(), 'models/my_u2net.pth')
print("✓ U2NET 학습 완료!")
```

```python
# 과제 2: inference/model.py 수정
# BackgroundRemover 클래스에 U2NET 추가

class BackgroundRemover:
    def __init__(self, model_type='rembg', custom_model_path=None):
        """
        Args:
            model_type: 'rembg' 또는 'u2net_custom'
            custom_model_path: 커스텀 모델 경로
        """
        self.model_type = model_type

        if model_type == 'rembg':
            self.session = new_session('u2net')
        elif model_type == 'u2net_custom':
            from models.u2net import U2NET
            self.model = U2NET(in_ch=3, out_ch=1)
            if custom_model_path:
                self.model.load_state_dict(torch.load(custom_model_path))
            self.model.eval()

    def remove_background(self, input_image):
        if self.model_type == 'rembg':
            return self._remove_background_rembg(input_image)
        else:
            return self._remove_background_u2net(input_image)

    def _remove_background_u2net(self, input_image):
        """U2NET으로 배경 제거"""
        # 전처리
        from torchvision import transforms

        image = Image.open(input_image).convert('RGB')
        original_size = image.size

        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0)

        # 추론
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            d0, *_ = self.model(input_tensor)  # 첫 번째 출력만 사용
            mask = d0[0, 0].cpu().numpy()  # (320, 320)

        # 후처리
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = mask.resize(original_size, Image.LANCZOS)

        # 알파 채널 적용
        output_image = image.copy()
        output_image.putalpha(mask)

        return output_image
```

```python
# 과제 3: app.py에서 테스트
# app.py 수정

# 모델 초기화 부분
def init_model():
    global bg_remover
    try:
        # 내가 만든 U2NET 사용!
        bg_remover = BackgroundRemover(
            model_type='u2net_custom',
            custom_model_path='models/my_u2net.pth'
        )
        print("✓ Custom U2NET model initialized")
    except Exception as e:
        print(f"Failed to initialize model: {e}")

# 서버 실행 후 테스트
# curl -X POST http://localhost:5000/upload -F "file=@test.jpg" -o result.png
```

### 체크포인트
- [ ] U2NET으로 모델을 학습했다
- [ ] BackgroundRemover에 통합했다
- [ ] API 서버에서 테스트 성공
- [ ] 결과 품질을 확인했다

---

## 7단계: 실험 및 개선 (지속적)

### 개선 아이디어

#### 7-1. 데이터 증강
```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

#### 7-2. 다른 손실 함수
```python
class DiceLoss(nn.Module):
    """Dice Loss - 세그멘테이션에 효과적"""
    def forward(self, pred, target):
        smooth = 1.0
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """BCE + Dice"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)
```

#### 7-3. 학습 기법
```python
# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Early Stopping
best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(100):
    # ... 학습 ...

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break

    scheduler.step(val_loss)
```

#### 7-4. 모델 경량화
```python
# 1. 채널 수 줄이기
model_lite = U2NET_Lite(base_ch=16)  # 기본 32 → 16

# 2. Depth 줄이기
rsu_shallow = RSU(in_ch=3, mid_ch=12, out_ch=64, depth=3)  # 7 → 3

# 3. Quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
```

### 성능 측정
```python
import time

def measure_performance(model, input_size=(1, 3, 320, 320)):
    """모델 성능 측정"""
    model.eval()
    dummy_input = torch.randn(input_size)

    # Warm-up
    with torch.no_grad():
        _ = model(dummy_input)

    # 측정
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end = time.time()

    avg_time = (end - start) / 100
    fps = 1 / avg_time

    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"FPS: {fps:.2f}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")
```

---

## 📖 추천 학습 자료

### 온라인 강의
- [모두를 위한 딥러닝 시즌2 - PyTorch](https://deeplearningzerotoall.github.io/season2/)
- [Stanford CS231n - CNN](http://cs231n.stanford.edu/)
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)

### 책
- "밑바닥부터 시작하는 딥러닝" (사이토 고키)
- "PyTorch로 시작하는 딥러닝" (윤대희)
- "컴퓨터 비전과 딥러닝" (라자링암)

### 논문
- [U-Net](https://arxiv.org/abs/1505.04597)
- [U²-Net](https://arxiv.org/abs/2005.09007)
- [DeepLabv3](https://arxiv.org/abs/1706.05587)

### 커뮤니티
- PyTorch 공식 포럼
- 모두의 연구소
- Kaggle Competitions

---

## ✅ 최종 체크리스트

### 기초 (필수)
- [ ] PIL/NumPy로 이미지 처리를 할 수 있다
- [ ] PyTorch Tensor 연산을 이해한다
- [ ] 간단한 CNN을 만들 수 있다
- [ ] U-Net 구조를 이해한다

### 중급 (중요)
- [ ] 데이터셋을 만들고 DataLoader를 사용한다
- [ ] 학습 루프를 작성할 수 있다
- [ ] 손실 함수를 이해하고 선택할 수 있다
- [ ] 모델을 저장하고 로드한다

### 고급 (목표)
- [ ] RSU 블록을 구현했다
- [ ] U2NET을 완성했다
- [ ] 실제 데이터로 학습했다
- [ ] API 서버에 통합했다

### 마스터 (도전)
- [ ] 성능을 측정하고 개선했다
- [ ] 다양한 손실 함수를 실험했다
- [ ] 모델을 경량화했다
- [ ] 논문을 읽고 새로운 아이디어를 적용했다

---

## 🎯 학습 팁

1. **작게 시작하기**: U2NET 바로 구현하지 말고 SimpleSegNet → SimpleUNet → U2NET 순서로

2. **시각화하기**: 중간 결과를 항상 이미지로 확인
```python
import matplotlib.pyplot as plt

plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.subplot(133)
plt.imshow(prediction, cmap='gray')
plt.show()
```

3. **적은 데이터로 과적합시키기**: 먼저 2-3장으로 loss가 0에 가까워지는지 확인

4. **에러를 두려워하지 않기**: Shape 에러는 print로 디버깅

5. **커뮤니티 활용**: 막힐 때는 질문하기 (Stack Overflow, PyTorch Forum)

---

## 🚀 다음 단계

이 로드맵을 완료하면:
- 이미지 세그멘테이션 전문가 수준
- 다른 컴퓨터 비전 모델도 쉽게 이해
- 논문 구현 능력 획득
- 실무 프로젝트 가능

**화이팅! 궁금한 점이 있으면 언제든 질문하세요!** 🎓
