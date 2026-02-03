"""
간단한 테스트 스크립트
실제 이미지 없이 모듈 동작 확인용
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path


def create_test_image():
    """테스트용 더미 이미지 생성"""
    # 512x512 이미지 생성 (중앙에 원)
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # 배경 (하늘색)
    img[:, :] = [135, 206, 235]

    # 중앙에 원 그리기 (사람 대신)
    center_x, center_y = 256, 256
    radius = 150

    for y in range(512):
        for x in range(512):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < radius:
                # 피부색으로 채우기
                img[y, x] = [255, 220, 177]

    return Image.fromarray(img)


def test_1_image_processor():
    """테스트 1: 이미지 처리 유틸리티"""
    print("=== 테스트 1: 이미지 처리 유틸리티 ===")

    try:
        from utils import ImageProcessor

        # 테스트 이미지 생성
        test_img = create_test_image()
        print("✓ 테스트 이미지 생성 완료 (512x512)")

        # 리사이즈 테스트
        resized = ImageProcessor.resize_with_aspect_ratio(
            test_img,
            target_size=(256, 256),
            background_color=(0, 0, 0, 0)
        )
        print(f"✓ 리사이즈 완료: {resized.size}")

        # RGBA 변환
        if resized.mode != 'RGBA':
            resized = resized.convert('RGBA')

        # 패딩 테스트
        padded = ImageProcessor.add_padding(resized, padding=20)
        print(f"✓ 패딩 추가 완료: {padded.size}")

        # 저장 테스트
        output_dir = Path('test_output')
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / 'test_processed.png'
        ImageProcessor.save_image(resized, output_path)
        print(f"✓ 이미지 저장 완료: {output_path}")

        print("✓ 이미지 처리 유틸리티 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_2_preprocessor():
    """테스트 2: 이미지 전처리"""
    print("=== 테스트 2: 이미지 전처리 ===")

    try:
        from inference import ImagePreprocessor

        # 테스트 이미지 저장
        test_img = create_test_image()
        test_path = 'test_output/test_input.jpg'
        test_img.save(test_path)

        # PIL로 로드
        loaded = ImagePreprocessor.load_image_pil(test_path)
        print(f"✓ 이미지 로드 완료: {loaded.size}, {loaded.mode}")

        # 리사이즈
        img_np = np.array(loaded)
        resized = ImagePreprocessor.resize_image(
            img_np,
            target_size=(320, 320),
            keep_aspect_ratio=True
        )
        print(f"✓ 리사이즈 완료: {resized.shape}")

        # 정규화
        normalized = ImagePreprocessor.normalize_image(resized)
        print(f"✓ 정규화 완료: min={normalized.min():.2f}, max={normalized.max():.2f}")

        print("✓ 이미지 전처리 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_3_config():
    """테스트 3: 설정 파일"""
    print("=== 테스트 3: 설정 파일 ===")

    try:
        import config

        print(f"✓ MODEL_NAME: {config.MODEL_NAME}")
        print(f"✓ BATCH_SIZE: {config.BATCH_SIZE}")
        print(f"✓ LEARNING_RATE: {config.LEARNING_RATE}")
        print(f"✓ NUM_EPOCHS: {config.NUM_EPOCHS}")
        print(f"✓ UPLOAD_FOLDER: {config.UPLOAD_FOLDER}")
        print(f"✓ MODEL_DIR: {config.MODEL_DIR}")

        # 디렉토리 존재 확인
        assert config.UPLOAD_FOLDER.exists(), "UPLOAD_FOLDER가 없습니다"
        assert config.MODEL_DIR.exists(), "MODEL_DIR이 없습니다"

        print("✓ 설정 파일 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_4_dataset():
    """테스트 4: 데이터셋 클래스 (데이터 없이)"""
    print("=== 테스트 4: 데이터셋 클래스 ===")

    try:
        from training import SegmentationDataset
        from pathlib import Path

        # 임시 데이터 디렉토리 생성
        test_data_dir = Path('test_output/test_data')
        (test_data_dir / 'images').mkdir(parents=True, exist_ok=True)
        (test_data_dir / 'masks').mkdir(parents=True, exist_ok=True)

        # 테스트 이미지 생성
        test_img = create_test_image()
        test_img.save(test_data_dir / 'images' / 'test1.jpg')

        # 테스트 마스크 생성 (중앙 원 부분만 255)
        mask = Image.new('L', (512, 512), 0)
        mask_np = np.array(mask)

        center_x, center_y = 256, 256
        radius = 150
        for y in range(512):
            for x in range(512):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < radius:
                    mask_np[y, x] = 255

        mask = Image.fromarray(mask_np)
        mask.save(test_data_dir / 'masks' / 'test1.png')

        # 데이터셋 로드
        dataset = SegmentationDataset(str(test_data_dir))
        print(f"✓ 데이터셋 로드 완료: {len(dataset)}개 샘플")

        if len(dataset) > 0:
            # 첫 번째 샘플 가져오기
            image, mask = dataset[0]
            print(f"✓ 샘플 로드 완료:")
            print(f"  - 이미지 shape: {image.shape}")
            print(f"  - 마스크 shape: {mask.shape}")

        print("✓ 데이터셋 클래스 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_5_background_remover_init():
    """테스트 5: BackgroundRemover 초기화 (모델 다운로드 없이)"""
    print("=== 테스트 5: BackgroundRemover 초기화 ===")

    try:
        print("⚠ 이 테스트는 rembg 모델 다운로드가 필요합니다 (~176MB)")
        print("처음 실행 시 시간이 걸릴 수 있습니다.")

        response = input("계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("테스트 건너뛰기\n")
            return None

        from inference import BackgroundRemover

        # 경량 모델로 초기화
        print("모델 초기화 중... (u2netp)")
        remover = BackgroundRemover(model_name='u2netp')
        print(f"✓ BackgroundRemover 초기화 완료")
        print(f"  - 모델: u2netp")
        print(f"  - 세션: {remover.session is not None}")

        print("✓ BackgroundRemover 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_6_full_pipeline():
    """테스트 6: 전체 파이프라인 (배경 제거 포함)"""
    print("=== 테스트 6: 전체 파이프라인 ===")

    try:
        print("⚠ 이 테스트는 실제 배경 제거를 수행합니다")

        response = input("계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("테스트 건너뛰기\n")
            return None

        from inference import BackgroundRemover
        from utils import ImageProcessor

        # 테스트 이미지 생성 및 저장
        test_img = create_test_image()
        test_path = 'test_output/test_full.jpg'
        test_img.save(test_path)
        print(f"✓ 테스트 이미지 저장: {test_path}")

        # 배경 제거
        print("배경 제거 중...")
        remover = BackgroundRemover(model_name='u2netp')
        output = remover.remove_background(test_path)
        print(f"✓ 배경 제거 완료: {output.size}, {output.mode}")

        # 추가 처리
        output = ImageProcessor.resize_with_aspect_ratio(output, (512, 512))
        output = ImageProcessor.auto_crop_transparent(output, margin=10)

        # 저장
        output_path = 'test_output/test_result.png'
        ImageProcessor.save_image(output, output_path)
        print(f"✓ 결과 저장: {output_path}")

        print("✓ 전체 파이프라인 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 테스트 실행"""
    print("🧪 누끼 제거 서비스 - 단위 테스트\n")
    print("=" * 50)

    results = []

    # 기본 테스트 (모델 없이)
    results.append(("이미지 처리 유틸리티", test_1_image_processor()))
    results.append(("이미지 전처리", test_2_preprocessor()))
    results.append(("설정 파일", test_3_config()))
    results.append(("데이터셋 클래스", test_4_dataset()))

    # 고급 테스트 (모델 필요)
    print("\n" + "=" * 50)
    print("고급 테스트 (선택사항)\n")

    results.append(("BackgroundRemover 초기화", test_5_background_remover_init()))
    results.append(("전체 파이프라인", test_6_full_pipeline()))

    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약\n")

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)

    for name, result in results:
        if result is True:
            print(f"✓ {name}: 통과")
        elif result is False:
            print(f"✗ {name}: 실패")
        else:
            print(f"○ {name}: 건너뜀")

    print(f"\n총 {len(results)}개 테스트:")
    print(f"  - 통과: {passed}")
    print(f"  - 실패: {failed}")
    print(f"  - 건너뜀: {skipped}")

    if failed == 0:
        print("\n✨ 모든 필수 테스트 통과!")
    else:
        print(f"\n⚠ {failed}개 테스트 실패")
        sys.exit(1)


if __name__ == '__main__':
    main()
