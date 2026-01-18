import sys
from pathlib import Path
from PIL import Image

# 지금 실행하려는 파일이 루트가 아니라서 ImageProcessor 파일을 찾지 못하여 설정해 줘야 함
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils import ImageProcessor

choonsik = Image.open('choonsik.jpg')

# 비율 유지하며 리사이즈
resized = ImageProcessor.resize_with_aspect_ratio(
    choonsik,
    target_size=(512, 512),
    background_color=(255, 255, 255, 0)
)

# 패딩 추가
padded = ImageProcessor.add_padding(resized, padding=50)
padded.save('resized_choonsik.png')

