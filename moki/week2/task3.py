# 과제3 : GPU 사용
# Mac에는 NVIDIA GPU 없음
import torch

from task2 import SimpleCNN

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  # Apple Silicon GPU
else:
    device = 'cpu'

print(f"Using device: {device}")

model = SimpleCNN().to(device)
input_tensor = torch.randn(1, 3, 320, 320).to(device)

with torch.no_grad():
    output = model(input_tensor)

print(f"Output device: {output.device}")