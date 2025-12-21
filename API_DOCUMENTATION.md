# API 문서

## 엔드포인트 목록

### 1. GET `/`
서비스 정보 및 사용 가능한 엔드포인트 목록

**응답 예시:**
```json
{
  "service": "AI Background Removal API",
  "version": "1.0.0",
  "endpoints": {
    "POST /upload": "Upload image and remove background",
    "POST /process": "Process with custom options",
    "POST /train": "Start model training (admin)",
    "GET /status": "Get training status"
  }
}
```

---

### 2. POST `/upload`
이미지 업로드 및 배경 제거

**요청 (Form Data):**
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| file | File | O | 이미지 파일 (PNG, JPG, JPEG, WEBP) |
| output_size | String | X | 출력 크기 "width,height" (예: "512,512") |
| auto_crop | Boolean | X | 투명 영역 자동 크롭 (default: false) |
| crop_margin | Integer | X | 크롭 시 여백 픽셀 (default: 10) |
| format | String | X | 출력 포맷 (png, jpg, jpeg, webp) (default: png) |

**응답:**
- Content-Type: `image/png` (또는 요청한 포맷)
- 배경이 제거된 이미지 파일

**cURL 예시:**
```bash
# 기본 사용
curl -X POST http://localhost:5000/upload \
  -F "file=@photo.jpg" \
  -o result.png

# 크기 조정 + 자동 크롭
curl -X POST http://localhost:5000/upload \
  -F "file=@photo.jpg" \
  -F "output_size=800,600" \
  -F "auto_crop=true" \
  -F "crop_margin=20" \
  -F "format=png" \
  -o result.png
```

---

### 3. POST `/process`
배경 제거 및 고급 이미지 처리

**요청 (Form Data):**
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| file | File | O | 이미지 파일 |
| remove_background | Boolean | X | 배경 제거 여부 (default: true) |
| resize_width | Integer | X | 리사이즈 너비 |
| resize_height | Integer | X | 리사이즈 높이 |
| keep_aspect_ratio | Boolean | X | 비율 유지 (default: true) |
| padding | Integer | X | 패딩 크기 (픽셀) |
| background_color | String | X | 배경색 "R,G,B" (예: "255,255,255") |
| format | String | X | 출력 포맷 (default: png) |

**응답:**
- Content-Type: `image/png` (또는 요청한 포맷)
- 처리된 이미지 파일

**cURL 예시:**
```bash
# 배경 제거 + 리사이즈 + 패딩 + 흰색 배경
curl -X POST http://localhost:5000/process \
  -F "file=@photo.jpg" \
  -F "remove_background=true" \
  -F "resize_width=1024" \
  -F "resize_height=768" \
  -F "keep_aspect_ratio=true" \
  -F "padding=50" \
  -F "background_color=255,255,255" \
  -F "format=jpg" \
  -o processed.jpg
```

---

### 4. POST `/train`
모델 학습 시작 (관리자용)

**요청 (JSON):**
```json
{
  "train_data_dir": "/path/to/train/data",  // 선택사항
  "val_data_dir": "/path/to/val/data",      // 선택사항
  "num_epochs": 50,                         // 선택사항
  "batch_size": 8                           // 선택사항
}
```

**응답 (202 Accepted):**
```json
{
  "message": "Training started in background",
  "config": {
    "train_data_dir": "/Users/eugene/Documents/GitHub/nukki-server/data/train",
    "val_data_dir": "/Users/eugene/Documents/GitHub/nukki-server/data/val",
    "num_epochs": 50,
    "batch_size": 8
  }
}
```

**에러 응답 (400 Bad Request):**
```json
{
  "error": "Training already in progress",
  "status": {
    "is_training": true,
    "current_epoch": 15,
    "total_epochs": 50,
    "loss": 0.234,
    "message": "Training in progress"
  }
}
```

**cURL 예시:**
```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "num_epochs": 30,
    "batch_size": 4
  }'
```

---

### 5. GET `/status`
학습 상태 조회

**응답:**
```json
{
  "is_training": true,
  "current_epoch": 15,
  "total_epochs": 50,
  "loss": 0.234,
  "message": "Training in progress"
}
```

**필드 설명:**
- `is_training`: 현재 학습 진행 여부
- `current_epoch`: 현재 에포크
- `total_epochs`: 전체 에포크 수
- `loss`: 현재 손실값
- `message`: 상태 메시지

**cURL 예시:**
```bash
curl http://localhost:5000/status
```

---

### 6. GET `/health`
헬스 체크

**응답:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "training": false
}
```

**cURL 예시:**
```bash
curl http://localhost:5000/health
```

---

## 에러 코드

| 코드 | 설명 |
|------|------|
| 400 | Bad Request - 잘못된 요청 파라미터 |
| 413 | Request Entity Too Large - 파일 크기 초과 (최대 16MB) |
| 500 | Internal Server Error - 서버 내부 오류 |

**에러 응답 형식:**
```json
{
  "error": "Error message here"
}
```

---

## Python 클라이언트 예시

```python
import requests

# 1. 배경 제거
def remove_background(image_path, output_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/upload', files=files)

        if response.status_code == 200:
            with open(output_path, 'wb') as out:
                out.write(response.content)
            print(f"Saved to {output_path}")
        else:
            print(f"Error: {response.json()}")

# 2. 고급 처리
def process_image(image_path, output_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'remove_background': 'true',
            'resize_width': '1024',
            'resize_height': '768',
            'keep_aspect_ratio': 'true',
            'background_color': '255,255,255',
            'format': 'jpg'
        }
        response = requests.post('http://localhost:5000/process', files=files, data=data)

        if response.status_code == 200:
            with open(output_path, 'wb') as out:
                out.write(response.content)
            print(f"Processed and saved to {output_path}")

# 3. 모델 학습
def start_training():
    data = {
        'num_epochs': 30,
        'batch_size': 8
    }
    response = requests.post('http://localhost:5000/train', json=data)
    print(response.json())

# 4. 상태 확인
def check_status():
    response = requests.get('http://localhost:5000/status')
    status = response.json()

    if status['is_training']:
        print(f"Training: Epoch {status['current_epoch']}/{status['total_epochs']}, Loss: {status['loss']:.4f}")
    else:
        print(f"Status: {status['message']}")

# 사용 예시
remove_background('input.jpg', 'output.png')
process_image('photo.jpg', 'processed.jpg')
start_training()
check_status()
```

---

## JavaScript/TypeScript 클라이언트 예시

```javascript
// 1. 배경 제거
async function removeBackground(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('format', 'png');

  const response = await fetch('http://localhost:5000/upload', {
    method: 'POST',
    body: formData
  });

  if (response.ok) {
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  } else {
    const error = await response.json();
    throw new Error(error.error);
  }
}

// 2. 고급 처리
async function processImage(file, options = {}) {
  const formData = new FormData();
  formData.append('file', file);

  if (options.width && options.height) {
    formData.append('resize_width', options.width);
    formData.append('resize_height', options.height);
  }

  if (options.backgroundColor) {
    formData.append('background_color', options.backgroundColor);
  }

  formData.append('format', options.format || 'png');

  const response = await fetch('http://localhost:5000/process', {
    method: 'POST',
    body: formData
  });

  if (response.ok) {
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  }
}

// 3. 학습 시작
async function startTraining(config) {
  const response = await fetch('http://localhost:5000/train', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(config)
  });

  return await response.json();
}

// 4. 상태 확인
async function checkStatus() {
  const response = await fetch('http://localhost:5000/status');
  return await response.json();
}

// 사용 예시
const fileInput = document.querySelector('input[type="file"]');
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const imageUrl = await removeBackground(file);
  document.querySelector('img').src = imageUrl;
});
```
