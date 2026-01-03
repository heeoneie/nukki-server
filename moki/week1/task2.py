import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from utils import ImageProcessor
from PIL import Image

# 현재 스크립트 디렉토리 기준으로 파일 경로 설정
script_dir = Path(__file__).parent
test_img = Image.open(script_dir / "my_image.jpeg")
resized = ImageProcessor.resize_with_aspect_ratio(
    test_img,
    target_size=(512, 512),
    background_color=(255, 255, 255, 0) # 투명 배경
)

padded = ImageProcessor.add_padding(resized, padding=50)
padded.save(script_dir / "result.png")