"""
VietOCR Transformer with Caching and Batch Processing
- Singleton pattern: load model once, reuse everywhere
- True batch processing for GPU efficiency
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import math
from PIL import Image
import numpy as np
import cv2
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))
from pathUtlis import get_path

# Global cache for singleton pattern
_model_cache = {}
_cache_lock = threading.Lock()


def get_vietocr_transformer(device='cpu', force_reload=False):
    """
    Get cached VietOCR Transformer instance (Singleton pattern)
    
    Args:
        device: 'cpu' or 'cuda'
        force_reload: Force reload model even if cached
        
    Returns:
        VietOCRTransformerFast instance
    """
    global _model_cache
    
    cache_key = f"transformer_{device}"
    
    with _cache_lock:
        if cache_key not in _model_cache or force_reload:
            _model_cache[cache_key] = VietOCRTransformerFast(device=device)
        return _model_cache[cache_key]


def clear_cache():
    """Clear model cache to free memory"""
    global _model_cache
    with _cache_lock:
        _model_cache.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


class VietOCRTransformerFast:
    """
    VietOCR Transformer with caching and batch processing
    
    Usage:
        # Singleton (recommended):
        ocr = get_vietocr_transformer()
        text = ocr(image)
        texts = ocr.batch(images)  # True batch processing
        
        # Direct instantiation:
        ocr = VietOCRTransformerFast()
    """
    
    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize VietOCR Transformer
        
        Args:
            model_path: Path to saved .pt file with state dict
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        if model_path is None:
            model_path = get_path('models') / 'vietocr_transformer_onnx' / 'vietocr_transformer_full.pt'
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run this first to create it:\n"
                "python -c \"from services.vietocr_transformer import VietOCRTransformerInference; VietOCRTransformerInference()\""
            )
        
        print(f"Loading VietOCR Transformer...")
        
        # Load saved checkpoint
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Build model architecture
        from vietocr.model.transformerocr import VietOCR
        from vietocr.model.vocab import Vocab
        
        # Get vocab
        vocab_chars = config.get('vocab', '')
        self.vocab = Vocab(vocab_chars)
        
        # Build model
        cnn_config = config.get('cnn', {})
        transformer_config = config.get('transformer', {})
        
        self.model = VietOCR(
            len(self.vocab),
            config.get('backbone', 'vgg19_bn'),
            cnn_config,
            transformer_config,
            config.get('seq_modeling', 'transformer')
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Config
        dataset_cfg = config.get('dataset', {})
        self.img_height = dataset_cfg.get('image_height', 32)
        self.img_min_width = dataset_cfg.get('image_min_width', 32)
        self.img_max_width = dataset_cfg.get('image_max_width', 512)
        
        print("[OK] VietOCR Transformer loaded")
    
    def preprocess_image(self, image):
        """Preprocess image to PIL"""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img = Image.fromarray(image).convert('RGB')
            else:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert('RGB')
        return img
    
    def __call__(self, image):
        """Run OCR on single image"""
        img = self.preprocess_image(image)
        
        from vietocr.tool.translate import translate, process_input
        
        img_tensor = process_input(
            img, 
            self.img_height, 
            self.img_min_width, 
            self.img_max_width
        ).to(self.device)
        
        with torch.no_grad():
            indices = translate(img_tensor, self.model)[0]
        
        # Convert to list for vocab.decode
        if hasattr(indices, 'squeeze'):
            indices = indices.squeeze()
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        if not isinstance(indices, list):
            indices = list(indices)
        
        return self.vocab.decode(indices)
    
    def batch(self, images, batch_size=8):
        """
        True batch processing for better GPU utilization
        
        Args:
            images: List of images (paths, numpy arrays, or PIL Images)
            batch_size: Number of images per batch (default 8)
            
        Returns:
            List of recognized texts
        """
        from vietocr.tool.translate import translate, process_input
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess all images in batch
            tensors = []
            valid_indices = []
            
            for j, img in enumerate(batch_images):
                try:
                    pil_img = self.preprocess_image(img)
                    tensor = process_input(
                        pil_img,
                        self.img_height,
                        self.img_min_width,
                        self.img_max_width
                    )
                    tensors.append(tensor)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"Error preprocessing image {i+j}: {e}")
            
            if not tensors:
                results.extend([""] * len(batch_images))
                continue
            
            # Pad tensors to same width for batching
            max_width = max(t.shape[3] for t in tensors)
            padded_tensors = []
            
            for tensor in tensors:
                if tensor.shape[3] < max_width:
                    padding = torch.zeros(1, 3, self.img_height, max_width - tensor.shape[3])
                    tensor = torch.cat([tensor, padding], dim=3)
                padded_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(padded_tensors, dim=0).to(self.device)
            
            # Run inference - translate returns (sentences, probs)
            with torch.no_grad():
                translated_sentences, _ = translate(batch_tensor, self.model)
            
            # translated_sentences shape: (batch, seq_len) - numpy array
            # Decode results
            batch_results = [""] * len(batch_images)
            for batch_idx, orig_idx in enumerate(valid_indices):
                # Get indices for this image
                indices = translated_sentences[batch_idx].tolist()
                batch_results[orig_idx] = self.vocab.decode(indices)
            
            results.extend(batch_results)
        
        return results
    
    def recognize_batch(self, images, batch_size=8):
        """Alias for batch() method"""
        return self.batch(images, batch_size)


def test_cache_and_batch():
    """Test caching and batch processing"""
    import time
    
    print("=" * 60)
    print("VietOCR Transformer - Cache & Batch Test")
    print("=" * 60)
    
    img_path = str(get_path('root') / 'image.png')
    
    # Test 1: First load (cold)
    print("\n[1] First Load (cold start):")
    clear_cache()
    start = time.time()
    ocr = get_vietocr_transformer()
    cold_load = (time.time() - start) * 1000
    print(f"    Load time: {cold_load:.0f}ms")
    
    # Test 2: Second load (cached)
    print("\n[2] Second Load (cached):")
    start = time.time()
    ocr2 = get_vietocr_transformer()
    cached_load = (time.time() - start) * 1000
    print(f"    Load time: {cached_load:.2f}ms")
    print(f"    Same instance: {ocr is ocr2}")
    
    # Test 3: Single inference
    print("\n[3] Single Image Inference:")
    _ = ocr(img_path)  # Warmup
    start = time.time()
    for _ in range(5):
        result = ocr(img_path)
    single_time = (time.time() - start) / 5 * 1000
    print(f"    Time per image: {single_time:.0f}ms")
    print(f"    Result: {result}")
    
    # Test 4: Batch inference
    print("\n[4] Batch Inference (5 images):")
    images = [img_path] * 5
    _ = ocr.batch(images)  # Warmup
    start = time.time()
    results = ocr.batch(images)
    for res in results:
        print(f"    Result: {res}")
    batch_time = (time.time() - start) * 1000
    print(f"    Total time: {batch_time:.0f}ms")
    print(f"    Time per image: {batch_time/5:.0f}ms")
    print(f"    Speedup vs single: {(single_time * 5) / batch_time:.1f}x")
    
    # Test 5: Larger batch
    print("\n[5] Larger Batch (20 images):")
    images = [img_path] * 20
    start = time.time()
    results = ocr.batch(images, batch_size=8)
    batch_time = (time.time() - start) * 1000
    print(f"    Total time: {batch_time:.0f}ms")
    print(f"    Time per image: {batch_time/20:.0f}ms")
    print(f"    Speedup vs single: {(single_time * 20) / batch_time:.1f}x")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cold load: {cold_load:.0f}ms")
    print(f"Cached load: {cached_load:.2f}ms ({cold_load/max(cached_load,0.01):.0f}x faster)")
    print(f"Single inference: {single_time:.0f}ms/image")
    print(f"Batch inference: {batch_time/20:.0f}ms/image")
    print("=" * 60)


if __name__ == '__main__':
    test_cache_and_batch()
