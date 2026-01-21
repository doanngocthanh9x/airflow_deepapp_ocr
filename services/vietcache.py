"""
VietOCR Config Loader - Load from local cache instead of downloading
"""
import os
import yaml
from pathlib import Path
import sys


def _get_cache_dir():
    """Get VietOCR cache directory - auto-detect Docker vs Windows"""
    # Check if running in Docker
    if os.path.exists('/opt/airflow'):
        # Docker: use mounted volume
        return Path('/opt/airflow/models/.vietocr_cache')
    else:
        # Windows: use local path
        return Path(__file__).parent.parent / 'models' / '.vietocr_cache'


# Local cache paths
VIETOCR_CACHE_DIR = _get_cache_dir()
CONFIG_DIR = VIETOCR_CACHE_DIR / 'config'
WEIGHTS_DIR = VIETOCR_CACHE_DIR / 'weights'

# Available configs
CONFIGS = {
    'vgg_transformer': 'vgg_transformer.yml',
    'vgg_seq2seq': 'vgg_seq2seq.yml',
}

WEIGHTS = {
    'vgg_transformer': 'vgg_transformer.pth',
    'vgg_seq2seq': 'vgg_seq2seq.pth',
}


def load_config_from_name(name):
    """
    Load config from local cache (no download)
    
    Args:
        name: 'vgg_transformer' or 'vgg_seq2seq'
        
    Returns:
        Config dict
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    
    config_file = CONFIG_DIR / CONFIGS[name]
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config not found: {config_file}\n"
            f"Download it first:\n"
            f"  Invoke-WebRequest -Uri 'https://vocr.vn/data/vietocr/config/{CONFIGS[name]}' "
            f"-OutFile '{config_file}'"
        )
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_weights_path(name):
    """
    Get local weights path (no download)
    
    Args:
        name: 'vgg_transformer' or 'vgg_seq2seq'
        
    Returns:
        Path to weights file
    """
    if name not in WEIGHTS:
        raise ValueError(f"Unknown weights: {name}. Available: {list(WEIGHTS.keys())}")
    
    weights_file = WEIGHTS_DIR / WEIGHTS[name]
    
    if not weights_file.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_file}\n"
            f"Download it first:\n"
            f"  Invoke-WebRequest -Uri 'https://vocr.vn/data/vietocr/{WEIGHTS[name]}' "
            f"-OutFile '{weights_file}'"
        )
    
    return str(weights_file)


def get_predictor(name='vgg_transformer', device='cpu'):
    """
    Get VietOCR Predictor with local config and weights
    
    Args:
        name: 'vgg_transformer' or 'vgg_seq2seq'
        device: 'cpu' or 'cuda'
        
    Returns:
        Predictor instance with batch prediction support
    """
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    
    # Load local config
    config = load_config_from_name(name)
    
    # Override weights path to local
    config['weights'] = get_weights_path(name)
    config['device'] = device
    
    # Create Cfg object
    cfg = Cfg(config)
    
    predictor = Predictor(cfg)
    
    # Add batch prediction method
    def predict_batch(images, batch_size=16):
        """
        Batch predict multiple images
        
        Args:
            images: List of PIL Images
            batch_size: Number of images per batch
            
        Returns:
            List of predicted texts
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # VietOCR's internal batch processing
            for img in batch:
                try:
                    text = predictor.predict(img)
                    results.append(text)
                except Exception as e:
                    results.append("")
        return results
    
    # Attach method to predictor instance
    predictor.predict_batch = predict_batch
    
    return predictor


def ensure_cache_exists():
    """Create cache directories if not exist"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache directories:")
    print(f"  Config: {CONFIG_DIR}")
    print(f"  Weights: {WEIGHTS_DIR}")


def download_all_to_cache():
    """Download all configs and weights to local cache"""
    import requests
    from tqdm import tqdm
    
    ensure_cache_exists()
    
    # Download configs
    print("\n[1] Downloading configs...")
    for name, filename in CONFIGS.items():
        url = f"https://vocr.vn/data/vietocr/config/{filename}"
        dest = CONFIG_DIR / filename
        if dest.exists():
            print(f"  ✓ {filename} (exists)")
        else:
            print(f"  ↓ {filename}...")
            r = requests.get(url, verify=False)
            with open(dest, 'w', encoding='utf-8') as f:
                f.write(r.text)
            print(f"  ✓ {filename}")
    
    # Download weights
    print("\n[2] Downloading weights...")
    for name, filename in WEIGHTS.items():
        url = f"https://vocr.vn/data/vietocr/{filename}"
        dest = WEIGHTS_DIR / filename
        if dest.exists():
            print(f"  ✓ {filename} (exists)")
        else:
            print(f"  ↓ {filename} (this may take a while)...")
            with requests.get(url, stream=True, verify=False) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(dest, 'wb') as f:
                    with tqdm(total=total, unit='B', unit_scale=True) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"  ✓ {filename}")
    
    print("\n✓ All files cached!")
    print(f"  Config dir: {CONFIG_DIR}")
    print(f"  Weights dir: {WEIGHTS_DIR}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VietOCR Cache Manager')
    parser.add_argument('--download', action='store_true', help='Download all files to cache')
    parser.add_argument('--check', action='store_true', help='Check cache status')
    
    args = parser.parse_args()
    
    if args.download:
        download_all_to_cache()
    elif args.check:
        ensure_cache_exists()
        print("\nCache status:")
        for name, filename in CONFIGS.items():
            path = CONFIG_DIR / filename
            status = "✓" if path.exists() else "✗"
            print(f"  {status} Config: {filename}")
        for name, filename in WEIGHTS.items():
            path = WEIGHTS_DIR / filename
            status = "✓" if path.exists() else "✗"
            size = f"({path.stat().st_size / 1024 / 1024:.1f}MB)" if path.exists() else ""
            print(f"  {status} Weights: {filename} {size}")
    else:
        parser.print_help()