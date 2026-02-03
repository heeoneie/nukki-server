import torch
from task2 import SimpleCNN

# 과제 3: GPU 사용
# mac은 cuda를 사용할 수 있는 NVIDIA GPU를 사용하지 않기에 mps를 할당
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

model = SimpleCNN().to(device)
input_tensor = torch.randn(1, 3, 320, 320).to(device)

with torch.no_grad():
    output = model(input_tensor)

print(f"Output device: {output.device}")