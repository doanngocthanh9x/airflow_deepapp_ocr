"""
Document Classifier Training - Config
PPLCNetV2 fine-tuning for Vietnamese ID documents
"""

from pathlib import Path
import os

# ============================================================================
# PATHS
# ============================================================================

# Auto-detect environment
if os.path.exists('/opt/airflow'):
    # Docker
    BASE_PATH = Path('/opt/airflow')
else:
    # Windows
    BASE_PATH = Path('C:/Automation/Airflow')

# Data paths
DATA_DIR = BASE_PATH / 'dags' / 'data'
OUTPUT_DIR = BASE_PATH / 'models' / 'doc_classifier'
LOGS_DIR = BASE_PATH / 'logs' / 'training'
PRETRAINED_DIR = BASE_PATH / 'models' / '.dags_train'  # Local pretrained weights

# ============================================================================
# MODEL CONFIG
# ============================================================================

MODEL_CONFIG = {
    # Model architecture - MobileNetV3 Small (lightweight and fast)
    'model_name': 'mobilenet_v3_small',
    
    # Pretrained weights - local path
    'pretrained': True,
    'pretrained_path': str(PRETRAINED_DIR / 'mobilenet_v3_small_x1.0.pdparams'),
    
    # Number of classes (auto-detected from data folders)
    'num_classes': None,  # Will be set dynamically
    
    # Input image size
    'image_size': 224,  # 224x224 is standard for classification
}

# ============================================================================
# TRAINING CONFIG
# ============================================================================

TRAIN_CONFIG = {
    # Data split
    'train_ratio': 0.8,
    'val_ratio': 0.15,
    'test_ratio': 0.05,
    
    # Training params
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 3e-4,  # Lower LR for fine-tuning
    
    # Optimizer
    'optimizer': 'Adam',
    'weight_decay': 1e-4,
    
    # Learning rate scheduler
    'lr_scheduler': 'CosineAnnealingDecay',
    'warmup_epochs': 2,
    
    # Early stopping
    'early_stop_patience': 5,
    
    # Save best model
    'save_best': True,
    'metric_for_best': 'accuracy',
}

# ============================================================================
# AUGMENTATION CONFIG
# ============================================================================

AUGMENT_CONFIG = {
    # Basic augmentations (safe for documents)
    'random_crop': True,
    'random_flip': False,  # Don't flip - documents have orientation
    'random_rotation': 5,  # Small rotation (±5 degrees)
    
    # Color augmentations
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.1,
    
    # Advanced
    'random_erasing': False,  # Might hide important text
    'mixup': False,
    'cutmix': False,
}

# ============================================================================
# CLASSES (will be auto-detected)
# ============================================================================

def get_classes():
    """Get class names from data folder"""
    if DATA_DIR.exists():
        classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
        return classes
    return []

CLASSES = get_classes()
MODEL_CONFIG['num_classes'] = len(CLASSES)

# ============================================================================
# LABEL MAPPING
# ============================================================================

LABEL_MAP = {
    'cccd_12_number_back': 'CCCD 12 số - Mặt sau',
    'cccd_12_number_front': 'CCCD 12 số - Mặt trước',
    'cccd_new_back': 'CCCD mới - Mặt sau',
    'cccd_new_front': 'CCCD mới - Mặt trước',
    'cccd_qr_back': 'CCCD QR - Mặt sau',
    'cccd_qr_front': 'CCCD QR - Mặt trước',
    'cmnd_12_number_front': 'CMND 12 số - Mặt trước',
    'cmnd_back': 'CMND - Mặt sau',
    'cmnd_front': 'CMND - Mặt trước',
    'giay_khai_sinh': 'Giấy khai sinh',
}


if __name__ == '__main__':
    print(f"Base path: {BASE_PATH}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"\nClasses ({len(CLASSES)}):")
    for i, cls in enumerate(CLASSES):
        label = LABEL_MAP.get(cls, cls)
        print(f"  {i}: {cls} -> {label}")
    print(f"\nModel: {MODEL_CONFIG['model_name']}")
    print(f"Image size: {MODEL_CONFIG['image_size']}x{MODEL_CONFIG['image_size']}")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
