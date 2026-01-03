"""
누끼 제거 서비스 사용 예제
"""

import os
from pathlib import Path
from inference import BackgroundRemover
from training import ModelTrainer
from utils import ImageProcessor


def example_1_basic_removal():
    """예제 1: 기본 배경 제거"""
    print("=== 예제 1: 기본 배경 제거 ===")

    # 모델 초기화
    remover = BackgroundRemover(model_name='u2net')

    # 배경 제거
    # output = remover.remove_background('input.jpg')
    # output.save('output.png')

    print("✓ 배경이 제거된 이미지가 output.png로 저장되었습니다.")


def example_2_with_processing():
    """예제 2: 배경 제거 + 이미지 가공"""
    print("\n=== 예제 2: 배경 제거 + 이미지 가공 ===")

    # 1. 배경 제거 (알파 매팅 사용)
    remover = BackgroundRemover()

    # output = remover.remove_background(
    #     'portrait.jpg',
    #     alpha_matting=True,  # 머리카락 디테일 살림
    #     alpha_matting_foreground_threshold=240,
    #     alpha_matting_background_threshold=10
    # )

    # 2. 크기 조정 (비율 유지)
    # output = ImageProcessor.resize_with_aspect_ratio(
    #     output,
    #     target_size=(1024, 768),
    #     background_color=(0, 0, 0, 0)  # 투명 배경
    # )

    # 3. 투명 영역 자동 크롭
    # output = ImageProcessor.auto_crop_transparent(output, margin=20)

    # 4. 흰색 배경 추가
    # final = ImageProcessor.change_background(output, (255, 255, 255))
    # final.save('processed.jpg', quality=95)

    print("✓ 처리된 이미지가 processed.jpg로 저장되었습니다.")


def example_3_batch_processing():
    """예제 3: 여러 이미지 일괄 처리"""
    print("\n=== 예제 3: 배치 처리 ===")

    # 모델 초기화 (한 번만)
    remover = BackgroundRemover(model_name='u2netp')  # 빠른 모델

    # 입력/출력 디렉토리
    input_dir = Path('input_images')
    output_dir = Path('output_images')
    output_dir.mkdir(exist_ok=True)

    # 모든 이미지 처리
    # if input_dir.exists():
    #     for image_path in input_dir.glob('*.jpg'):
    #         print(f"Processing: {image_path.name}")
    #
    #         # 배경 제거
    #         output = remover.remove_background(str(image_path))
    #
    #         # 저장
    #         output_path = output_dir / f"{image_path.stem}_nobg.png"
    #         output.save(output_path)
    #
    #         print(f"  → Saved: {output_path}")

    print("✓ 모든 이미지 처리 완료")


def example_4_custom_training():
    """예제 4: 커스텀 모델 학습"""
    print("\n=== 예제 4: 커스텀 모델 학습 ===")

    # 데이터 디렉토리 확인
    train_dir = Path('data/train')
    if not (train_dir / 'images').exists():
        print("⚠ 학습 데이터가 없습니다. data/train/images와 data/train/masks를 준비하세요.")
        return

    # 학습 시작
    trainer = ModelTrainer()

    print("학습 시작...")
    # result = trainer.train(
    #     train_data_dir='data/train',
    #     val_data_dir='data/val',
    #     num_epochs=10,  # 테스트용으로 적은 에포크
    #     batch_size=4
    # )

    # print(f"✓ 학습 완료!")
    # print(f"  - 최종 손실값: {result['best_loss']:.4f}")
    # print(f"  - 완료 에포크: {result['epochs_completed']}")
    # print(f"  - 모델 저장: models/custom_model.pth")


def example_5_use_custom_model():
    """예제 5: 학습한 커스텀 모델 사용"""
    print("\n=== 예제 5: 커스텀 모델 사용 ===")

    # 커스텀 모델 확인
    if not Path('models/custom_model.pth').exists():
        print("⚠ 커스텀 모델이 없습니다. 먼저 example_4를 실행하세요.")
        return

    # 커스텀 모델로 배경 제거
    # remover = BackgroundRemover(use_custom_model=True)
    # output = remover.remove_background('test.jpg')
    # output.save('custom_result.png')

    print("✓ 커스텀 모델로 배경 제거 완료")


def example_6_api_client():
    """예제 6: API 클라이언트 사용"""
    print("\n=== 예제 6: API 클라이언트 ===")

    import requests

    # API 서버가 실행 중이어야 함
    api_url = 'http://localhost:5000'

    # 1. 서버 상태 확인
    try:
        response = requests.get(f'{api_url}/health')
        if response.status_code == 200:
            print(f"✓ 서버 연결 성공: {response.json()}")
        else:
            print("⚠ 서버가 실행 중이 아닙니다. 먼저 'python app.py'를 실행하세요.")
            return
    except Exception as e:
        print(f"⚠ 서버 연결 실패: {e}")
        return

    # 2. 배경 제거 요청
    # with open('test.jpg', 'rb') as f:
    #     files = {'file': f}
    #     data = {
    #         'output_size': '512,512',
    #         'auto_crop': 'true',
    #         'format': 'png'
    #     }
    #
    #     response = requests.post(f'{api_url}/upload', files=files, data=data)
    #
    #     if response.status_code == 200:
    #         with open('api_result.png', 'wb') as out:
    #             out.write(response.content)
    #         print("✓ API로 배경 제거 완료: api_result.png")
    #     else:
    #         print(f"⚠ 에러: {response.json()}")


def example_7_advanced_options():
    """예제 7: 고급 옵션 활용"""
    print("\n=== 예제 7: 고급 옵션 ===")

    remover = BackgroundRemover()

    # 시나리오: 증명사진 만들기
    # 1. 배경 제거 (알파 매팅으로 깔끔하게)
    # output = remover.remove_background(
    #     'portrait.jpg',
    #     alpha_matting=True,
    #     alpha_matting_foreground_threshold=240,
    #     alpha_matting_background_threshold=10,
    #     alpha_matting_erode_size=10
    # )

    # 2. 투명 영역 크롭 (여백 제거)
    # output = ImageProcessor.auto_crop_transparent(output, margin=10)

    # 3. 3x4cm 비율로 리사이즈 (300dpi 기준: 354x472 픽셀)
    # output = ImageProcessor.resize_with_aspect_ratio(
    #     output,
    #     target_size=(354, 472),
    #     background_color=(0, 0, 0, 0)
    # )

    # 4. 파란색 배경 추가 (증명사진용)
    # final = ImageProcessor.change_background(output, (44, 101, 176))  # 파란색

    # 5. 저장
    # final.save('id_photo.jpg', quality=100)

    print("✓ 증명사진 생성 완료: id_photo.jpg")


def example_8_compare_models():
    """예제 8: 모델 성능 비교"""
    print("\n=== 예제 8: 모델 성능 비교 ===")

    import time

    models = ['u2net', 'u2netp']

    for model_name in models:
        print(f"\n{model_name} 테스트:")

        # 모델 초기화
        start_time = time.time()
        remover = BackgroundRemover(model_name=model_name)
        init_time = time.time() - start_time
        print(f"  - 초기화 시간: {init_time:.2f}초")

        # 추론
        # start_time = time.time()
        # output = remover.remove_background('test.jpg')
        # inference_time = time.time() - start_time
        # print(f"  - 추론 시간: {inference_time:.2f}초")

        # 저장
        # output.save(f'output_{model_name}.png')


def main():
    """모든 예제 실행"""
    print("🎨 누끼 제거 서비스 예제\n")

    # 기본 예제들
    example_1_basic_removal()
    example_2_with_processing()
    example_3_batch_processing()

    # 학습 관련 예제 (선택)
    # example_4_custom_training()
    # example_5_use_custom_model()

    # API 클라이언트 예제
    example_6_api_client()

    # 고급 예제
    example_7_advanced_options()
    example_8_compare_models()

    print("\n\n✨ 모든 예제 완료!")
    print("\n📖 더 자세한 정보:")
    print("  - README.md: 프로젝트 개요")
    print("  - API_DOCUMENTATION.md: API 상세 문서")
    print("  - TECHNICAL_GUIDE.md: 기술 원리 설명")


if __name__ == '__main__':
    main()
