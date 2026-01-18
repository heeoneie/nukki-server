from PIL import Image
import numpy as np

# 1. 이미지를 NumPy 배열로 변환
choonsik = Image.open('choonsik.jpg')
image_arr = np.array(choonsik)
print(f"choonsik image Shape: {image_arr.shape}")

# 2. 알파 채널 직접 만들기
# 여기서 알파 채널이란 투명도 마스크를 의미
height, width = image_arr.shape[:2] # slice를 사용하여 높이와 너비 값 가져오기

# np.zeros(...): 주어진 크기에 맞는 0으로 채워진 배열 생성
# dtype=np.uint8: 0~255 범위의 정수형 (이미지 픽셀 타입)
# 초기값 0 = 완전 투명 (또는 배경) 즉 이 작업은 이미지의 투명 배경을 생성하는 단계
mask = np.zeros((height, width), dtype=np.uint8)

# 중앙 사각형만 255로 변경
mask[100:200, 100:200] = 255

# 3. RGBA로 변환
img_rgba = Image.fromarray(image_arr).convert('RGBA')
img_rgba.putalpha(Image.fromarray(mask))
img_rgba.save('choonsik_output.png')

print("알파 채널이 적용!")