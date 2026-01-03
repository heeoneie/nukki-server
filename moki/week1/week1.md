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
- [x] PIL로 이미지를 열고 저장할 수 있다
- [X] NumPy 배열과 PIL Image를 변환할 수 있다
- [X] 알파 채널의 개념을 이해했다
- [X] utils/image_processing.py의 모든 함수를 실행해봤다

---

### PIL(Python Imaging Libaray)

- 기존 PIL 라이브러리 개발은 2011년 중단
- Pillow 라는 이름의 프로젝트가 현재까지 개발되고 있음

- img_array.shape
  - 세로높이 (Height) : 이미지 행(Row) 개수
  - 가로너비 (Width) : 이미지 열(Column) 개수
  - 채널 수 (Channels) : RGB 이미지의 경우 보통 3 (Red, Green, Blue)

  - 흑백 이미지는 색상 채널이 하나라서 보통 2개의 숫자만 출력됨
  - 넘파이 배열에서는 (세로, 가로) 순서로 출력된다는 점 주의하기

### 알파채널 (Alpha Channel)

- 일반적인 디지털 이미지는 RGB(Red, Green, Blue) 색상 모델을 기반으로 함
- 여기에 Alpah 채널이 추가되면 RGBA가 되어서 색상 뿐만 아니라 투명도까지 제어 가능
- 0부터 255까지의 8비트 값으로 표현 > 256단계의 투명도를 표현할 수 있다는 의미
- 0은 투명 255는 완전 불투명