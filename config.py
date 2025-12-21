import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Upload settings
UPLOAD_FOLDER = BASE_DIR / 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Model settings
MODEL_DIR = BASE_DIR / 'models'
MODEL_NAME = 'u2net'  # rembg 기본 모델 (u2net, u2netp, u2net_human_seg, etc.)
CUSTOM_MODEL_PATH = MODEL_DIR / 'custom_model.pth'

# Training settings
TRAIN_DATA_DIR = BASE_DIR / 'data' / 'train'
VAL_DATA_DIR = BASE_DIR / 'data' / 'val'
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
CHECKPOINT_DIR = MODEL_DIR / 'checkpoints'

# Image processing settings
DEFAULT_OUTPUT_SIZE = (512, 512)
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'webp']

# Training status
TRAINING_STATUS = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'loss': 0.0,
    'message': 'Ready'
}

# Flask settings
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Create necessary directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
VAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
