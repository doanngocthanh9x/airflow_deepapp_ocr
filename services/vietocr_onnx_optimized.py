"""
VietOCR ONNX Inference - OPTIMIZED VERSION
✨ Improvements:
- Direct PIL Image processing (no temp file I/O)
- Batch inference support for speed
- Better memory management
- Optimized preprocessing pipeline
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import onnxruntime
import os
from typing import Union, List

# Suppress ONNX Runtime warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
onnxruntime.set_default_logger_severity(3)


class VietOCRVocab:
    """Vietnamese OCR Vocabulary"""
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3
        
        self.chars = chars
        self.c2i = {c: i + 4 for i, c in enumerate(chars)}
        self.i2c = {i + 4: c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'
    
    def decode(self, ids):
        """Decode token ids to text"""
        first = 1 if self.go in ids else 0
        try:
            last = ids.index(self.eos)
        except ValueError:
            last = None
        
        # Safely decode - skip invalid tokens
        chars = []
        for i in ids[first:last]:
            if i in self.i2c:
                chars.append(self.i2c[i])
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.c2i) + 4


class VietOCRVocabProvider:
    """Load Vietnamese character set from config"""
    @staticmethod
    def get_vocab(config_dir=None):
        """Load vocab from config files"""
        if config_dir is None:
            # Try to find config directory
            possible_paths = [
                Path('models/buiquangmanhhp1999/config'),
                Path('ConvertVietOcr2Onnx/config'),
                Path('config'),
            ]
            
            for path in possible_paths:
                if (path / 'base.yml').exists():
                    config_dir = path
                    break
        
        if config_dir is None:
            # Fallback: use default Vietnamese character set
            return VietOCRVocab(VietOCRVocabProvider._get_default_chars())
        
        import yaml
        
        # Load base.yml
        base_yml = Path(config_dir) / 'base.yml'
        with open(base_yml, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        chars = config.get('vocab', VietOCRVocabProvider._get_default_chars())
        return VietOCRVocab(chars)
    
    @staticmethod
    def _get_default_chars():
        """Default Vietnamese character set"""
        return 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưỨừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '


class VietOCROnnxInferenceOptimized:
    """
    Vietnamese OCR using ONNX Runtime - OPTIMIZED
    
    Features:
    - Direct PIL/numpy image processing (no temp files)
    - Batch inference support
    - Cached preprocessing parameters
    - Memory efficient
    """
    
    def __init__(self, cnn_model=None, encoder_model=None, decoder_model=None, 
                 vocab=None, config_dir=None, input_size=(32, 320)):
        """
        Initialize ONNX models
        
        Args:
            cnn_model: Path to CNN ONNX model
            encoder_model: Path to Encoder ONNX model
            decoder_model: Path to Decoder ONNX model
            vocab: VietOCRVocab instance
            config_dir: Path to config directory (for vocab loading)
            input_size: (height, width) for input images, default (32, 320)
        """
        # Use default paths if not provided
        if cnn_model is None:
            weight_dir = Path('models/buiquangmanhhp1999/weights')
            if not weight_dir.exists():
                weight_dir = Path('weights')
            
            cnn_model = str(weight_dir / 'cnn.onnx')
            encoder_model = str(weight_dir / 'encoder.onnx')
            decoder_model = str(weight_dir / 'decoder.onnx')
        
        # Load ONNX sessions with optimizations
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.cnn_session = onnxruntime.InferenceSession(
            cnn_model,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Cache input names for speed
        self.cnn_input_name = self.cnn_session.get_inputs()[0].name
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        
        # Load vocab
        if vocab is None:
            vocab = VietOCRVocabProvider.get_vocab(config_dir)
        self.vocab = vocab
        
        self.sos_token = 1
        self.eos_token = 2
        self.max_seq_length = 128
        self.input_size = input_size  # (H, W)
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for CNN input - OPTIMIZED
        
        Args:
            image: Path to image, PIL Image, or numpy array
            
        Returns:
            Preprocessed image array (1, 3, H, W)
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(image)
        else:
            img = image
        
        # Resize (use LANCZOS for better quality)
        h, w = self.input_size
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize in one step
        img_array = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, 0)
    
    def preprocess_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Preprocess multiple images for batch inference
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            
        Returns:
            Batch of preprocessed images (N, 3, H, W)
        """
        batch = [self.preprocess_image(img)[0] for img in images]
        return np.stack(batch, axis=0)
    
    def recognize_single(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """
        Run OCR on a single image
        
        Args:
            image: Image to process
            
        Returns:
            Recognized text
        """
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # CNN forward pass
        cnn_outputs = self.cnn_session.run(None, {self.cnn_input_name: img_array})
        src = cnn_outputs[0]
        
        # Encoder forward pass
        encoder_outputs = self.encoder_session.run(None, {self.encoder_input_name: src})
        encoder_out = encoder_outputs[0]
        hidden = encoder_outputs[1]
        
        # Decoding loop
        translated_sentence = [[self.sos_token]]
        max_length = 0
        
        while max_length <= self.max_seq_length:
            tgt_inp = np.array(translated_sentence[-1], dtype=np.int64)
            
            decoder_inputs = {
                self.decoder_input_names[0]: tgt_inp,
                self.decoder_input_names[1]: hidden,
                self.decoder_input_names[2]: encoder_out
            }
            
            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
            output = decoder_outputs[0]
            hidden = decoder_outputs[1]
            
            if output.ndim > 1:
                output = output[0]
            
            predicted_token = int(np.argmax(output, axis=-1))
            translated_sentence.append([predicted_token])
            max_length += 1
            
            if predicted_token == self.eos_token:
                break
        
        # Decode to text
        sentence_ids = np.concatenate(translated_sentence).flatten().tolist()
        return self.vocab.decode(sentence_ids)
    
    def recognize_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> List[str]:
        """
        Run OCR on multiple images (batch processing for speed)
        
        Args:
            images: List of images to process
            
        Returns:
            List of recognized texts
        """
        results = []
        
        # Process in batches of size 1 (ONNX models are sequence models)
        # Note: True batching requires model redesign
        for img in images:
            try:
                text = self.recognize_single(img)
                results.append(text)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append("")
        
        return results
    
    def __call__(self, image: Union[str, Image.Image, np.ndarray, List]) -> Union[str, List[str]]:
        """
        Run OCR on image(s)
        
        Args:
            image: Single image or list of images
            
        Returns:
            Recognized text or list of texts
        """
        if isinstance(image, list):
            return self.recognize_batch(image)
        else:
            return self.recognize_single(image)


# Convenience functions
def ocr_vietocr_onnx_optimized(image: Union[str, Image.Image, np.ndarray],
                               cnn_model=None, encoder_model=None, decoder_model=None) -> str:
    """Quick OCR function using optimized ONNX"""
    ocr = VietOCROnnxInferenceOptimized(cnn_model, encoder_model, decoder_model)
    return ocr(image)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Test optimized version
        print(f"Processing: {image_path}")
        
        ocr = VietOCROnnxInferenceOptimized()
        result = ocr(image_path)
        
        print(f"OCR Result: {result}")
    else:
        print("Usage: python vietocr_onnx_optimized.py <image_path>")