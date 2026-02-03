from torchvision import transforms
from PIL import Image

# inference/preprocessor.py 참고
# Compose는 배열을 받는데 배열에 있는 식을 순차적으로 실행시킴
transform = transforms.Compose([
    # 이미지를 입력 받은 사이즈로 리사이징
    transforms.Resize((320, 320)),
    # Tensor로 변환 각 픽셀의 범위가 0~255에서 0.0~1.0으로 변경 (이렇게 함으로 각각의 픽셀을 계산하기 쉽게 하는 듯)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # RGB 각 채널의 평균
        std=[0.229, 0.224, 0.225] # RGB 각 채널의 표준편차
    )
])

img = Image.open('choonsik.jpg')
tensor = transform(img)
print(f"Tensor shape: {tensor.shape}")  # (3, 320, 320)
print(f"Min: {tensor.min():.2f}, Max: {tensor.max():.2f}")

# 배치 차원 추가
batch = tensor.unsqueeze(0)  # (1, 3, 320, 320)
print(f"Batch shape: {batch.shape}")