# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

rembg 라이브러리를 통해 U2NET 딥러닝 모델을 사용하는 Flask 기반 AI 배경 제거 서비스입니다. 이미지 배경 제거, 커스텀 이미지 처리, 모델 학습 기능을 제공하는 REST API 엔드포인트를 제공합니다.

## 개발 명령어

### 설정 및 설치
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 서버 실행
```bash
# Flask 개발 서버 시작 (기본값: http://0.0.0.0:5000)
python app.py
```

### 테스트
```bash
# 단위 테스트 실행 (기본 테스트 및 모델 기반 테스트 포함)
python test_simple.py
```

## 아키텍처

### 핵심 컴포넌트

**Flask 애플리케이션 (`app.py`)**
- 6개 엔드포인트를 가진 메인 API 서버: `/`, `/upload`, `/process`, `/train`, `/status`, `/health`
- 서버 시작 시 한 번만 초기화되는 전역 모델 인스턴스 (`bg_remover`)
- 모델 학습을 위한 백그라운드 스레드 (`threading.Thread` 사용)
- 파일 처리: UUID 기반 파일명으로 `uploads/`에 업로드 저장

**배경 제거 (`inference/`)**
- `model.py`: rembg와 커스텀 모델을 모두 지원하는 `BackgroundRemover` 클래스
  - `_remove_background_rembg()`: U2NET 세션과 함께 rembg 라이브러리 사용
  - `_remove_background_custom()`: 수동 전처리를 통한 학습된 PyTorch 모델 사용
- `preprocessor.py`: 이미지 전처리 유틸리티 (리사이즈, 정규화, 텐서 변환)

**모델 학습 (`training/`)**
- `trainer.py`: 전체 학습 루프를 가진 `ModelTrainer` 클래스
  - 기본 모델로 DeepLabV3-ResNet50 사용 (단일 채널 출력으로 수정됨)
  - 이진 세그멘테이션을 위한 BCE 손실 함수
  - GPU/CPU 자동 감지 및 디바이스 선택
  - `config.TRAINING_STATUS`에서 학습 상태 추적 (공유 전역 상태)
- `dataset.py`: `SegmentationDataset`과 `AugmentedSegmentationDataset`
  - 예상 데이터 구조: `{data_dir}/images/*.jpg` 및 `{data_dir}/masks/*.png`
  - 마스크는 그레이스케일이어야 함 (0=배경, 255=전경)

**이미지 처리 (`utils/image_processing.py`)**
- 유틸리티 함수: 비율 유지 리사이즈, 투명 영역 자동 크롭, 패딩 추가, 배경색 변경
- 모든 함수는 PIL Image 객체와 함께 작동

**설정 (`config.py`)**
- 경로, 하이퍼파라미터, Flask 서버 설정의 중앙 집중식 설정
- import 시 필요한 디렉토리 생성
- `TRAINING_STATUS` dict: 백그라운드 학습 진행 상황을 위한 공유 가변 상태

### 데이터 흐름

**배경 제거 요청:**
```
POST /upload → 파일 저장 → BackgroundRemover.remove_background() →
rembg.remove() [U2NET 추론] → 선택적 resize/crop → 출력 저장 →
send_file() → 임시 파일 정리
```

**모델 학습 요청:**
```
POST /train → 파라미터 검증 → 백그라운드 스레드 생성 →
ModelTrainer.train() [에포크 루프: train_epoch → validate → 체크포인트 저장] →
TRAINING_STATUS 업데이트 → custom_model.pth에 최고 모델 저장
```

### 주요 기술 세부사항

**모델 아키텍처:**
- 기본 추론: rembg를 통한 U2NET (u2net, u2netp, u2net_human_seg 변형)
- 커스텀 학습: 출력 레이어가 수정된 DeepLabV3-ResNet50 (256→1 채널)
- 모델 가중치: `models/custom_model.pth` 및 `models/checkpoints/`에 저장

**이미지 처리 파이프라인:**
1. 이미지 로드 → RGB 변환
2. 320×320으로 리사이즈 (모델 입력 크기)
3. 정규화: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
4. 모델 추론 → sigmoid 출력 (0-1 확률)
5. 마스크를 원본 크기로 리사이즈
6. 알파 채널로 적용 → RGBA 출력

**학습 데이터 요구사항:**
- 구조: `data/train/images/` + `data/train/masks/`, `data/val/images/` + `data/val/masks/`
- 이미지: RGB (JPG/PNG), 마스크: 그레이스케일 PNG (이진 0/255)
- 파일명이 일치해야 함: `image.jpg` → `image.png`

**백그라운드 학습:**
- Python threading 사용 (Celery 아님) - 단일 스레드 백그라운드 실행
- 학습 중 `/status` 엔드포인트를 통해 학습 상태 접근 가능
- 학습 중에도 API가 응답 유지 (논블로킹)

## 중요한 패턴

**에러 처리:**
- 파일 작업은 정리(os.remove)와 함께 try/except로 감싸짐
- 모델 초기화 실패는 로그에 기록되지만 서버를 중단시키지 않음
- 학습 오류는 `TRAINING_STATUS`를 실패 상태로 업데이트

**파일 관리:**
- 모든 업로드된 파일은 충돌 방지를 위해 UUID 접두사를 가짐
- 원본 파일은 처리 후 삭제됨
- 출력 파일은 `uploads/`에 유지됨 (수동 정리 필요)

**모델 세션 관리:**
- rembg 세션은 `BackgroundRemover.__init__`에서 한 번 초기화됨
- GPU 자동 감지: `torch.cuda.is_available()`
- 모델은 추론 전에 디바이스로 이동됨

**설정 접근:**
- config 모듈 import: `import config`
- `config.MODEL_NAME`, `config.BATCH_SIZE` 등을 통해 접근
- 경로는 `pathlib.Path` 객체이므로, 파일 작업 시 문자열로 변환

## 테스트 접근 방식

테스트 스위트(`test_simple.py`)는 두 계층으로 구성됨:
1. 기본 테스트 (모델 다운로드 불필요): utils, config, dataset, preprocessor
2. 고급 테스트 (모델 필요): BackgroundRemover 초기화, 전체 파이프라인

테스트는 실제 이미지 파일 의존성을 피하기 위해 합성 이미지(배경 위의 원)를 생성합니다.
