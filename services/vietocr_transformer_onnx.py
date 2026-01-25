"""
VietOCR Transformer ONNX Inference
Uses ONNX Runtime for fast Vietnamese OCR with Transformer architecture

This uses the exported ONNX models from convert_vietocr_transformer_to_onnx.py:
- vietocr_encoder.onnx: CNN + Transformer Encoder
- vietocr_decoder.onnx: Transformer Decoder + FC
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import os

# Suppress ONNX Runtime warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ort.set_default_logger_severity(3)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathUtlis import get_path


class VietOCRTransformerVocab:
    """Vocabulary for VietOCR Transformer"""
    
    def __init__(self, chars):
        self.pad = 0
        self.sos = 1
        self.eos = 2
        self.mask = 3
        
        self.chars = chars
        self.c2i = {c: i + 4 for i, c in enumerate(chars)}
        self.i2c = {i + 4: c for i, c in enumerate(chars)}
        
        # Special tokens
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'
    
    def decode(self, ids):
        """Decode token ids to text"""
        # Find SOS and EOS positions
        start = 0
        end = len(ids)
        
        for i, idx in enumerate(ids):
            if idx == self.sos:
                start = i + 1
            if idx == self.eos:
                end = i
                break
        
        # Decode characters
        chars = []
        for i in ids[start:end]:
            if i in self.i2c and i > 3:  # Skip special tokens
                chars.append(self.i2c[i])
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.chars) + 4


class VietOCRTransformerONNX:
    """VietOCR Transformer using ONNX Runtime"""
    
    def __init__(self, model_dir=None, use_gpu=False):
        """
        Initialize ONNX inference
        
        Args:
            model_dir: Directory containing ONNX models and vocab
            use_gpu: Use CUDA if available
        """
        if model_dir is None:
            model_dir = get_path('models') / 'vietocr_transformer_onnx'
        
        model_dir = Path(model_dir)
        
        # Check files exist
        encoder_path = model_dir / 'vietocr_encoder.onnx'
        decoder_path = model_dir / 'vietocr_decoder.onnx'
        vocab_path = model_dir / 'vocab.txt'
        
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder model not found: {decoder_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        
        # Set providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # Load ONNX sessions
        print(f"Loading VietOCR Transformer ONNX models from {model_dir}...")
        
        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            providers=providers
        )
        
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            providers=providers
        )
        
        # Load vocab
        vocab_chars = vocab_path.read_text(encoding='utf-8')
        self.vocab = VietOCRTransformerVocab(vocab_chars)
        
        # Config
        self.img_height = 32
        self.img_width = 512  # Will resize dynamically
        self.max_seq_len = 128
        self.sos_token = 1
        self.eos_token = 2
        
        print(f"[OK] Loaded encoder: {encoder_path.name}")
        print(f"[OK] Loaded decoder: {decoder_path.name}")
        print(f"[OK] Vocab size: {len(self.vocab)}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            numpy array (1, 3, H, W) where W=512
        """
        # Load image if path
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert('RGB')
        
        # Resize keeping aspect ratio
        w, h = img.size
        new_h = self.img_height  # 32
        new_w = int(w * new_h / h)
        
        # Limit width
        if new_w > self.img_width:
            new_w = self.img_width
        elif new_w < 32:
            new_w = 32
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Pad to fixed width (512) for ONNX model
        # Create white background
        padded_img = Image.new('RGB', (self.img_width, self.img_height), color='white')
        padded_img.paste(img, (0, 0))
        
        # Convert to numpy
        img_array = np.array(padded_img).astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)
        
        return img_array
    
    def encode(self, image):
        """
        Encode image to memory representation
        
        Args:
            image: Preprocessed image (1, 3, H, W)
            
        Returns:
            memory: Encoder output (seq_len, batch, d_model)
        """
        input_name = self.encoder_session.get_inputs()[0].name
        outputs = self.encoder_session.run(None, {input_name: image})
        return outputs[0]
    
    def decode_greedy(self, memory):
        """
        Greedy decoding from memory
        
        Note: Due to ONNX limitations with dynamic attention,
        we need to pass the full sequence each time.
        
        Args:
            memory: Encoder output (seq_len, batch, d_model)
            
        Returns:
            List of token ids
        """
        batch_size = memory.shape[1]
        
        # Build sequence incrementally
        # Start with SOS token padded to match export length
        max_decode_len = 128  # Must match export tgt_len
        
        decoded_ids = [self.sos_token]
        
        for step in range(max_decode_len - 1):
            # Create full sequence with padding
            tgt = np.zeros((max_decode_len, batch_size), dtype=np.int64)
            for i, tid in enumerate(decoded_ids):
                tgt[i, 0] = tid
            
            # Run decoder
            outputs = self.decoder_session.run(
                None,
                {
                    'tgt': tgt,
                    'memory': memory
                }
            )
            
            logits = outputs[0]  # (batch, tgt_len, vocab_size)
            
            # Get prediction at current position
            current_pos = len(decoded_ids) - 1
            current_logits = logits[0, current_pos, :]  # (vocab_size,)
            
            # Greedy: take argmax
            next_token = int(np.argmax(current_logits))
            
            decoded_ids.append(next_token)
            
            # Check EOS
            if next_token == self.eos_token:
                break
        
        return decoded_ids
    
    def __call__(self, image):
        """
        Run OCR on single image
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            Recognized text string
        """
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Encode
        memory = self.encode(img_array)
        
        # Decode
        token_ids = self.decode_greedy(memory)
        
        # Convert to text
        text = self.vocab.decode(token_ids)
        
        return text
    
    def recognize_batch(self, images):
        """
        Run OCR on batch of images
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of recognized text strings
        """
        results = []
        for img in images:
            try:
                text = self(img)
                results.append(text)
            except Exception as e:
                print(f"Error recognizing image: {e}")
                results.append("")
        
        return results


def test_vietocr_transformer_onnx():
    """Test the ONNX inference"""
    print("=" * 60)
    print("Testing VietOCR Transformer ONNX Inference")
    print("=" * 60)
    
    try:
        ocr = VietOCRTransformerONNX()
        
        # Test with sample image
        test_image = get_path('root') / 'image.png'
        if test_image.exists():
            print(f"\nTesting with: {test_image}")
            result = ocr(str(test_image))
            print(f"Result: {result}")
        else:
            print(f"\nNo test image found at: {test_image}")
            print("Creating dummy test...")
            
            # Create dummy image
            dummy_img = Image.new('RGB', (200, 32), color='white')
            result = ocr(dummy_img)
            print(f"Dummy result: {result}")
        
        print("\n[OK] VietOCR Transformer ONNX working!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', help='Image path to OCR')
    parser.add_argument('--test', action='store_true', help='Run test')
    
    args = parser.parse_args()
    
    if args.test or not args.image:
        test_vietocr_transformer_onnx()
    else:
        ocr = VietOCRTransformerONNX()
        result = ocr(args.image)
        print(f"OCR Result: {result}")
