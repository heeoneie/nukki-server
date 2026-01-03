from PIL import Image
import numpy as np

# 이미지 열고 배열로 변환하기
img = Image.open("test_image.jpeg")
img_array = np.array(img);

print(f"Shape: {img_array.shape}") # (H, W, 3)

# 알파 채널 직접 만들기
height, width = img_array.shape[:2] # 슬라이싱 : 리스트나 튜플에서 앞부분 일부만 잘라냄

# np.zeros(...) 모든 원소의 값이 0인 배열 생성 / 첫번째 인자는 형태를 나타내는 튜플이어야 함
# (height, width) : 새로 만들 배열의 형태
# dtype=np.uint8 : 배열에 저장될 데이터 타입 (부호 없는 8비트 정수)
mask = np.zeros((height, width), dtype=np.uint8)

# 중앙 사각형만 255
# 가로 100px 세로 100px 크기의 사각형 영역에 들어있는 모든 값을 255로 변경
mask[100:200, 100:200] = 255

# 3. RGBA로 변환
img_rgba = Image.fromarray(img_array).convert('RGBA')
img_rgba.putalpha(Image.fromarray(mask))
img_rgba.save('output.png')

print("알파채널 적용한 이미지 저장")
