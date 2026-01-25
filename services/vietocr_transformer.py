"""
VietOCR Transformer PyTorch Inference
Uses transformer.pth model for Vietnamese OCR
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathUtlis import get_path


class VietOCRTransformerInference:
    """Vietnamese OCR using PyTorch Transformer model"""
    
    def __init__(self, model_path=None, config_path=None, device=None):
        """
        Initialize Transformer model
        
        Args:
            model_path: Path to transformer.pth file
            config_path: Path to config yaml file
            device: "cpu" or "cuda" (auto-detect if None)
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device        
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
            
            # Load config
            config = Cfg.load_config_from_name('vgg_transformer')
            config['device'] = 'cpu'
            # Create predictor
            self.predictor = Predictor(config)
            
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    @staticmethod
    def _load_vocab(vocab_string):
        """Load vocabulary from string"""
        chars = ['<pad>', '<sos>', '<eos>']
        chars.extend(list(vocab_string))
        return chars
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            # Convert numpy array to PIL
            img = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        
        return img
    
    def __call__(self, image_path):
        """
        Run OCR on single image
        
        Args:
            image_path: Path to image or PIL Image or numpy array
            
        Returns:
            Recognized text string
        """
        try:
            # Preprocess
            img = self.preprocess_image(image_path)
            
            # Predict
            text = self.predictor.predict(img)
            
            return text
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return ""
    
    def recognize_batch(self, images):
        """
        Run OCR on batch of PIL images
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of recognized text strings
        """
        results = []
        for img in images:
            try:
                # Predict
                text = self.predictor.predict(img)
                results.append(text)
            except Exception as e:
                print(f"Error during batch prediction: {e}")
                results.append("")
        
        return results


# Convenience function
def ocr_vietocr_transformer(image_path):
    """Quick OCR function using Transformer model"""
    ocr = VietOCRTransformerInference()
    return ocr(image_path)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = ocr_vietocr_transformer(image_path)
        print(f"\nOCR Result: {result}")
    else:
        print("Usage: python vietocr_transformer.py <image_path>")
