# AI 누끼 제거 서비스 (Nukki Server)

- Flask 기반 AI 이미지 배경 제거 백엔드 서비스입니다.
- claude init : U2NET 모델을 직접 구현한 모델로 교체

## 프로젝트 구조

```
nukki-server/
├── app.py                      # Flask 메인 애플리케이션
├── config.py                   # 설정 파일
├── requirements.txt            # 의존성 패키지
│
├── inference/                  # 추론(Inference) 모듈
│   ├── __init__.py
│   ├── model.py               # 배경 제거 모델
│   └── preprocessor.py        # 이미지 전처리
│
├── training/                   # 학습(Training) 모듈
│   ├── __init__.py
│   ├── trainer.py             # 모델 학습 로직
│   └── dataset.py             # 데이터셋 관리
│
├── utils/                      # 유틸리티
│   ├── __init__.py
│   └── image_processing.py    # 이미지 가공 유틸리티
│
├── uploads/                    # 업로드된 이미지 임시 저장
├── models/                     # 학습된 모델 저장
│   └── checkpoints/           # 학습 체크포인트
│
└── data/                       # 학습 데이터
    ├── train/                 # 학습 데이터
    │   ├── images/           # 원본 이미지
    │   └── masks/            # 마스크 이미지
    └── val/                   # 검증 데이터
        ├── images/
        └── masks/
```

## 주요 기능

### 1. 배경 제거 API (`/upload`)
- 이미지 업로드 후 AI가 자동으로 배경 제거
- rembg 라이브러리 기반 U2NET 모델 사용
- 출력 크기 조정, 자동 크롭, 포맷 변환 지원

### 2. 고급 이미지 처리 API (`/process`)
- 배경 제거 + 추가 가공 (리사이즈, 패딩, 배경색 변경)
- 비율 유지 옵션
- 다양한 포맷 지원 (PNG, JPG, WEBP)

### 3. 모델 학습 API (`/train`)
- 관리자용 학습 엔드포인트
- 백그라운드 스레드로 실행 (API 서비스 중단 없음)
- 커스텀 데이터셋으로 모델 미세 조정

### 4. 학습 상태 조회 (`/status`)
- 현재 학습 진행 상황 확인
- 에포크, 손실값 등 실시간 모니터링

## 설치 및 실행

### 1. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 서버 실행
```bash
python app.py
```

서버는 기본적으로 `http://0.0.0.0:5000`에서 실행됩니다.

## API 사용 예시

### 1. 배경 제거 (기본)
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@image.jpg" \
  -o output.png
```

### 2. 배경 제거 + 크기 조정
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@image.jpg" \
  -F "output_size=512,512" \
  -F "auto_crop=true" \
  -F "format=png" \
  -o output.png
```

### 3. 고급 처리 (리사이즈 + 패딩)
```bash
curl -X POST http://localhost:5000/process \
  -F "file=@image.jpg" \
  -F "remove_background=true" \
  -F "resize_width=800" \
  -F "resize_height=600" \
  -F "keep_aspect_ratio=true" \
  -F "padding=20" \
  -F "background_color=255,255,255" \
  -F "format=png" \
  -o processed.png
```

### 4. 모델 학습 시작
```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "num_epochs": 50,
    "batch_size": 8
  }'
```

### 5. 학습 상태 확인
```bash
curl http://localhost:5000/status
```

## 학습 데이터 준비

모델을 학습하려면 다음과 같은 구조로 데이터를 준비하세요:

```
data/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── masks/
│       ├── img001.png
│       ├── img002.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

- `images/`: 원본 이미지 (RGB)
- `masks/`: 세그멘테이션 마스크 (흑백, 전경=255, 배경=0)

## 설정 (config.py)

주요 설정값을 `config.py`에서 변경할 수 있습니다:

```python
# 모델 설정
MODEL_NAME = 'u2net'  # u2net, u2netp, u2net_human_seg 등

# 학습 설정
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# 업로드 설정
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

## 백그라운드 작업 처리

학습 작업은 `threading.Thread`를 사용하여 백그라운드에서 실행됩니다:
- API 서비스는 학습 중에도 정상적으로 동작
- `/status` 엔드포인트로 실시간 진행 상황 확인 가능
- 프로덕션 환경에서는 Celery + Redis 사용 권장

## 향후 확장 계획

- [ ] Celery를 사용한 비동기 작업 큐
- [ ] Redis를 통한 학습 상태 관리
- [ ] 모델 버전 관리
- [ ] S3 연동 (이미지 저장)
- [ ] Docker 컨테이너화
- [ ] API 인증 및 권한 관리

## 기술 스택

- **Framework**: Flask 3.0
- **AI/ML**: PyTorch, rembg, torchvision
- **Image Processing**: OpenCV, Pillow
- **Background Tasks**: Python Threading (향후 Celery)

## 라이선스

MIT License
