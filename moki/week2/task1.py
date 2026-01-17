# 과제1: 이미지 전처리 파이프라인
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open('test_image.jpeg')
tensor = transform(img)
print(f"Tensor shape: {tensor.shape}") # (3, 320, 320)
print(f"Min: {tensor.min():.2f}, Max: {tensor.max(): 2f}")

batch = tensor.unsqueeze(0)
print(f"Batch shape: {batch.shape}")

# 배치처리 해보기
img_paths = ['test1.png', 'test2.png', 'test3.jpeg']

tensors = []
for path in img_paths:
    img = Image.open(path).convert('RGB')
    tensor = transform(img)
    tensors.append(tensor)

batch = torch.stack(tensors)
print(f"Batch Shape: {batch.shape}")
