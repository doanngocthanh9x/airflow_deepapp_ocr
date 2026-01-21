"""
Document Classifier - Inference
Load trained model and predict document type
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


class DocumentClassifier:
    """
    Classify Vietnamese ID documents
    
    Usage:
        classifier = DocumentClassifier('/path/to/inference')
        result = classifier.predict(image)
        print(result)
        # {'class': 'cccd_qr_front', 'label': 'CCCD QR - Mặt trước', 'confidence': 0.98}
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize classifier
        
        Args:
            model_dir: Path to inference model directory
        """
        import paddle
        from paddle.inference import create_predictor, Config
        
        self.model_dir = Path(model_dir)
        
        # Load config
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load label map
        label_path = self.model_dir / 'label_map.json'
        with open(label_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        # Label names for display
        self.label_names = {
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
        
        # Create predictor
        model_file = str(self.model_dir / 'model.pdmodel')
        params_file = str(self.model_dir / 'model.pdiparams')
        
        config = Config(model_file, params_file)
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(4)
        config.enable_memory_optim()
        
        self.predictor = create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(
            self.predictor.get_input_names()[0]
        )
        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0]
        )
        
        # Image params
        self.img_size = self.config.get('image_size', 224)
        self.mean = np.array(self.config['normalize']['mean']).reshape(3, 1, 1)
        self.std = np.array(self.config['normalize']['std']).reshape(3, 1, 1)
    
    def preprocess(self, image) -> np.ndarray:
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image, numpy array, or path
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Load image
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize
        image = image.resize((self.img_size, self.img_size))
        
        # To numpy [H, W, C] -> [C, H, W]
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array.transpose((2, 0, 1))
        
        # Normalize
        img_array = img_array / 255.0
        img_array = (img_array - self.mean) / self.std
        
        # Add batch dimension [1, C, H, W]
        img_array = img_array[np.newaxis, :].astype(np.float32)
        
        return img_array
    
    def predict(self, image, top_k: int = 3) -> dict:
        """
        Predict document type
        
        Args:
            image: Image (path, PIL, numpy)
            top_k: Number of top predictions to return
            
        Returns:
            dict with class, label, confidence, and top_k predictions
        """
        # Preprocess
        input_data = self.preprocess(image)
        
        # Inference
        self.input_handle.copy_from_cpu(input_data)
        self.predictor.run()
        output = self.output_handle.copy_to_cpu()
        
        # Softmax
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / exp_output.sum(axis=1, keepdims=True)
        probs = probs[0]  # Remove batch dimension
        
        # Top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = []
        for idx in top_indices:
            class_name = self.label_map[str(idx)]
            label = self.label_names.get(class_name, class_name)
            top_k_results.append({
                'class': class_name,
                'label': label,
                'confidence': float(probs[idx])
            })
        
        # Best prediction
        best = top_k_results[0]
        
        return {
            'class': best['class'],
            'label': best['label'],
            'confidence': best['confidence'],
            'top_k': top_k_results
        }
    
    def predict_batch(self, images: list) -> list:
        """
        Batch prediction
        
        Args:
            images: List of images
            
        Returns:
            List of prediction results
        """
        return [self.predict(img) for img in images]


def create_classifier(model_dir: str = None) -> DocumentClassifier:
    """
    Factory function to create classifier
    
    Args:
        model_dir: Path to model. If None, use default path.
    """
    import os
    
    if model_dir is None:
        # Auto-detect
        if os.path.exists('/opt/airflow'):
            model_dir = '/opt/airflow/models/doc_classifier/inference'
        else:
            model_dir = 'C:/Automation/Airflow/models/doc_classifier/inference'
    
    return DocumentClassifier(model_dir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Document Classifier')
    parser.add_argument('image', help='Path to image')
    parser.add_argument('--model-dir', default=None, help='Model directory')
    
    args = parser.parse_args()
    
    classifier = create_classifier(args.model_dir)
    result = classifier.predict(args.image)
    
    print(f"\nPrediction:")
    print(f"  Class: {result['class']}")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"\nTop-3:")
    for i, r in enumerate(result['top_k']):
        print(f"  {i+1}. {r['label']} ({r['confidence']:.4f})")
