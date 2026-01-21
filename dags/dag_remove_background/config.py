"""
Configuration for Remove Background DAG
"""
from pathlib import Path

# Paths
BASE_DIR = Path('/opt/airflow')
DATA_DIR = BASE_DIR / 'dags' / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Input/Output directories
INPUT_DIR = DATA_DIR / 'remove_bg_input'
OUTPUT_DIR = DATA_DIR / 'remove_bg_output'

# Rembg model options: u2net, u2netp, u2net_human_seg, silueta, isnet-general-use, isnet-anime
REMBG_CONFIG = {
    'model': 'u2net',           # Default model (best quality)
    'alpha_matting': False,     # Enable alpha matting for better edges
    'alpha_matting_fg_threshold': 240,
    'alpha_matting_bg_threshold': 10,
    'alpha_matting_erode_size': 10,
    'output_format': 'png',     # PNG for transparency
    'bgcolor': None,            # None = transparent, or (R, G, B, A)
}

# Batch processing
BATCH_CONFIG = {
    'batch_size': 10,           # Process 10 images at a time
    'max_workers': 4,           # Parallel workers
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
}

# Model download cache
REMBG_HOME = MODELS_DIR / '.rembg'
