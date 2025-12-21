import os
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import config
from inference import BackgroundRemover
from training import ModelTrainer
from utils import ImageProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# 글로벌 모델 인스턴스 (서버 시작 시 한 번만 로드)
bg_remover = None


def init_model():
    """모델 초기화"""
    global bg_remover
    try:
        bg_remover = BackgroundRemover(model_name=config.MODEL_NAME)
        print("✓ Background removal model initialized")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        bg_remover = None


def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """API 루트"""
    return jsonify({
        'service': 'AI Background Removal API',
        'version': '1.0.0',
        'endpoints': {
            'POST /upload': 'Upload image and remove background',
            'POST /process': 'Process with custom options',
            'POST /train': 'Start model training (admin)',
            'GET /status': 'Get training status'
        }
    })


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    이미지 업로드 및 배경 제거

    Form data:
        - file: 이미지 파일 (required)
        - output_size: 출력 크기 "width,height" (optional)
        - format: 출력 포맷 (png, jpg, jpeg, webp) (optional, default: png)

    Returns:
        배경이 제거된 이미지 파일
    """
    # 모델 확인
    if bg_remover is None:
        return jsonify({'error': 'Model not initialized'}), 500

    # 파일 확인
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {config.ALLOWED_EXTENSIONS}'}), 400

    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        input_path = config.UPLOAD_FOLDER / unique_filename
        file.save(input_path)

        # 배경 제거
        output_image = bg_remover.remove_background(str(input_path))

        # 출력 크기 조정 (선택사항)
        if 'output_size' in request.form:
            try:
                width, height = map(int, request.form['output_size'].split(','))
                output_image = ImageProcessor.resize_with_aspect_ratio(
                    output_image,
                    (width, height)
                )
            except ValueError:
                pass  # 잘못된 형식이면 무시

        # 자동 크롭 (투명 영역 제거)
        if request.form.get('auto_crop', 'false').lower() == 'true':
            margin = int(request.form.get('crop_margin', 10))
            output_image = ImageProcessor.auto_crop_transparent(output_image, margin=margin)

        # 출력 포맷
        output_format = request.form.get('format', 'png').upper()
        if output_format == 'JPG':
            output_format = 'JPEG'

        # 출력 파일 저장
        output_filename = f"{uuid.uuid4()}_output.{output_format.lower()}"
        output_path = config.UPLOAD_FOLDER / output_filename
        ImageProcessor.save_image(output_image, output_path, format=output_format)

        # 원본 파일 삭제
        os.remove(input_path)

        # 파일 전송
        return send_file(
            output_path,
            mimetype=f'image/{output_format.lower()}',
            as_attachment=True,
            download_name=f'removed_bg_{filename}'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_image():
    """
    이미지 처리 (배경 제거 + 추가 가공)

    JSON body:
        - file: base64 encoded image or file upload
        - remove_background: 배경 제거 여부 (default: true)
        - resize: {width, height} 리사이즈 옵션
        - padding: 패딩 추가
        - background_color: 배경색 변경 (R,G,B)
        - format: 출력 포맷

    Returns:
        JSON with base64 encoded image or file
    """
    # 모델 확인
    if bg_remover is None:
        return jsonify({'error': 'Model not initialized'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed'}), 400

    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        input_path = config.UPLOAD_FOLDER / unique_filename
        file.save(input_path)

        # 배경 제거
        remove_bg = request.form.get('remove_background', 'true').lower() == 'true'
        if remove_bg:
            output_image = bg_remover.remove_background(str(input_path))
        else:
            output_image = ImageProcessor.bytes_to_image(file.read())

        # 추가 처리
        # 리사이즈
        if 'resize_width' in request.form and 'resize_height' in request.form:
            width = int(request.form['resize_width'])
            height = int(request.form['resize_height'])
            keep_ratio = request.form.get('keep_aspect_ratio', 'true').lower() == 'true'

            if keep_ratio:
                output_image = ImageProcessor.resize_with_aspect_ratio(
                    output_image, (width, height)
                )
            else:
                output_image = ImageProcessor.resize_exact(
                    output_image, (width, height)
                )

        # 패딩 추가
        if 'padding' in request.form:
            padding = int(request.form['padding'])
            output_image = ImageProcessor.add_padding(output_image, padding)

        # 배경색 변경
        if 'background_color' in request.form:
            try:
                r, g, b = map(int, request.form['background_color'].split(','))
                output_image = ImageProcessor.change_background(output_image, (r, g, b))
            except ValueError:
                pass

        # 저장 및 반환
        output_format = request.form.get('format', 'png').upper()
        output_filename = f"{uuid.uuid4()}_processed.{output_format.lower()}"
        output_path = config.UPLOAD_FOLDER / output_filename
        ImageProcessor.save_image(output_image, output_path, format=output_format)

        # 원본 파일 삭제
        os.remove(input_path)

        return send_file(
            output_path,
            mimetype=f'image/{output_format.lower()}',
            as_attachment=True,
            download_name=f'processed_{filename}'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """
    모델 학습 시작 (관리자용)

    JSON body:
        - train_data_dir: 학습 데이터 디렉토리 (optional, default: config)
        - val_data_dir: 검증 데이터 디렉토리 (optional)
        - num_epochs: 에포크 수 (optional)
        - batch_size: 배치 크기 (optional)

    Returns:
        학습 시작 상태
    """
    # 이미 학습 중인지 확인
    if config.TRAINING_STATUS['is_training']:
        return jsonify({
            'error': 'Training already in progress',
            'status': config.TRAINING_STATUS
        }), 400

    # 파라미터 가져오기
    data = request.get_json() or {}
    train_data_dir = data.get('train_data_dir', str(config.TRAIN_DATA_DIR))
    val_data_dir = data.get('val_data_dir', str(config.VAL_DATA_DIR))
    num_epochs = data.get('num_epochs', config.NUM_EPOCHS)
    batch_size = data.get('batch_size', config.BATCH_SIZE)

    # 학습 데이터 디렉토리 확인
    if not Path(train_data_dir).exists():
        return jsonify({'error': f'Training data directory not found: {train_data_dir}'}), 400

    # 백그라운드에서 학습 시작
    def train_in_background():
        try:
            trainer = ModelTrainer()
            result = trainer.train(
                train_data_dir=train_data_dir,
                val_data_dir=val_data_dir if Path(val_data_dir).exists() else None,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            print(f"Training completed: {result}")
        except Exception as e:
            print(f"Training failed: {e}")
            config.TRAINING_STATUS['is_training'] = False
            config.TRAINING_STATUS['message'] = f'Training failed: {str(e)}'

    training_thread = threading.Thread(target=train_in_background, daemon=True)
    training_thread.start()

    return jsonify({
        'message': 'Training started in background',
        'config': {
            'train_data_dir': train_data_dir,
            'val_data_dir': val_data_dir,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }
    }), 202


@app.route('/status', methods=['GET'])
def get_status():
    """
    현재 학습 상태 조회

    Returns:
        학습 상태 정보
    """
    return jsonify(config.TRAINING_STATUS)


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': bg_remover is not None,
        'training': config.TRAINING_STATUS['is_training']
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """파일 크기 초과 에러"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_error(error):
    """내부 서버 에러"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # 모델 초기화
    print("Initializing AI Background Removal Service...")
    init_model()

    # 서버 실행
    print(f"Starting server on {config.HOST}:{config.PORT}")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
