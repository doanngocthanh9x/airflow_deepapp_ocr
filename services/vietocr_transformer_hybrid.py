"""
VietOCR Transformer Hybrid Inference
Uses ONNX for Encoder (61% faster) + PyTorch for Decoder (5% faster)
Total: ~163ms vs 192ms pure PyTorch (15% speedup)
"""
import sys
from pathlib import Path
import numpy as np
import math
import torch
from PIL import Image
import cv2
import onnxruntime as ort
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ort.set_default_logger_severity(3)

sys.path.insert(0, str(Path(__file__).parent.parent))
from pathUtlis import get_path


class VietOCRTransformerHybrid:
    """
    Hybrid VietOCR Transformer: ONNX Encoder + PyTorch Decoder
    
    Benchmark results:
    - Pure PyTorch: 192ms
    - Pure ONNX: 169ms  
    - Hybrid: 163ms (fastest!)
    
    Usage:
        ocr = VietOCRTransformerHybrid()
        text = ocr("image.png")
    """
    
    def __init__(self, onnx_encoder_path=None, pytorch_weights_path=None):
        """
        Initialize hybrid model
        
        Args:
            onnx_encoder_path: Path to ONNX encoder model
            pytorch_weights_path: Path to PyTorch weights (optional, will download if not provided)
        """
        self.device = 'cpu'
        self.d_model = 256
        self.img_height = 32
        self.img_width = 512
        self.max_seq_len = 128
        
        # Load ONNX Encoder
        if onnx_encoder_path is None:
            onnx_encoder_path = get_path('models') / 'vietocr_transformer_onnx' / 'vietocr_encoder.onnx'
        
        print(f"Loading ONNX Encoder: {onnx_encoder_path}")
        self.encoder_session = ort.InferenceSession(
            str(onnx_encoder_path),
            providers=['CPUExecutionProvider']
        )
        print("[OK] ONNX Encoder loaded")
        
        # Load PyTorch Decoder components
        print("Loading PyTorch Decoder components...")
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = self.device
        
        predictor = Predictor(config)
        self.model = predictor.model
        self.model.eval()
        
        # Extract vocab from predictor
        self.vocab = predictor.vocab
        
        # Get decoder components
        self.embed_tgt = self.model.transformer.embed_tgt
        self.pos_enc = self.model.transformer.pos_enc
        self.transformer_decoder = self.model.transformer.transformer.decoder
        self.fc = self.model.transformer.fc
        self.gen_nopeek_mask = self.model.transformer.gen_nopeek_mask
        
        print("[OK] PyTorch Decoder loaded")
        print("[OK] VietOCR Transformer Hybrid ready!")
    
    def preprocess_image(self, image):
        """Preprocess image for encoder input"""
        # Load image if path
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert('RGB')
        
        # Resize keeping aspect ratio
        w, h = img.size
        new_h = self.img_height
        new_w = int(w * new_h / h)
        
        if new_w > self.img_width:
            new_w = self.img_width
        elif new_w < 32:
            new_w = 32
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Pad to fixed width
        padded_img = Image.new('RGB', (self.img_width, self.img_height), color='white')
        padded_img.paste(img, (0, 0))
        
        # Convert to numpy
        img_array = np.array(padded_img).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, 0)  # Add batch
        
        return img_array
    
    def encode(self, image_array):
        """ONNX Encoder: image -> memory"""
        input_name = self.encoder_session.get_inputs()[0].name
        outputs = self.encoder_session.run(None, {input_name: image_array})
        memory = outputs[0]  # (seq_len, batch, d_model)
        return torch.from_numpy(memory)
    
    def decode_greedy(self, memory):
        """PyTorch Decoder: memory -> text"""
        batch_size = memory.shape[1]
        
        # Start with SOS token
        sos_token = 1
        eos_token = 2
        
        decoded_ids = [sos_token]
        
        with torch.no_grad():
            for step in range(self.max_seq_len - 1):
                # Prepare target sequence
                tgt = torch.tensor(decoded_ids, dtype=torch.long).unsqueeze(1)  # (T, 1)
                
                # Generate causal mask
                tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(memory.device)
                
                # Embed and add positional encoding
                tgt_emb = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
                
                # Decode
                output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # Get logits for last position
                logits = self.fc(output[-1:].transpose(0, 1))  # (1, 1, vocab_size)
                
                # Greedy: take argmax
                next_token = int(torch.argmax(logits[0, 0]).item())
                
                decoded_ids.append(next_token)
                
                # Check EOS
                if next_token == eos_token:
                    break
        
        return decoded_ids
    
    def decode_tokens(self, token_ids):
        """Convert token ids to text"""
        # Remove SOS/EOS
        start = 1 if token_ids[0] == 1 else 0
        try:
            end = token_ids.index(2)
        except ValueError:
            end = len(token_ids)
        
        # Decode using vocab (i2c = index to char)
        chars = []
        for tid in token_ids[start:end]:
            if hasattr(self.vocab, 'i2c') and tid in self.vocab.i2c:
                chars.append(self.vocab.i2c[tid])
            elif hasattr(self.vocab, 'i2s') and tid in self.vocab.i2s:
                chars.append(self.vocab.i2s[tid])
        
        return ''.join(chars)
    
    def __call__(self, image):
        """Run OCR on image"""
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # ONNX Encode
        memory = self.encode(img_array)
        
        # PyTorch Decode
        token_ids = self.decode_greedy(memory)
        
        # Convert to text
        text = self.decode_tokens(token_ids)
        
        return text
    
    def recognize_batch(self, images):
        """OCR batch of images"""
        results = []
        for img in images:
            try:
                text = self(img)
                results.append(text)
            except Exception as e:
                print(f"Error: {e}")
                results.append("")
        return results


def test_hybrid():
    """Test hybrid inference"""
    print("=" * 60)
    print("Testing VietOCR Transformer Hybrid")
    print("=" * 60)
    
    ocr = VietOCRTransformerHybrid()
    
    test_image = get_path('root') / 'image.png'
    if test_image.exists():
        print(f"\nTesting with: {test_image}")
        
        import time
        
        # Warmup
        _ = ocr(str(test_image))
        
        # Benchmark
        n_runs = 5
        start = time.time()
        for _ in range(n_runs):
            result = ocr(str(test_image))
        avg_time = (time.time() - start) / n_runs * 1000
        
        print(f"Result: {result}")
        print(f"Avg time: {avg_time:.1f}ms")
        print("\n[OK] Hybrid inference working!")
    else:
        print(f"Test image not found: {test_image}")


if __name__ == '__main__':
    test_hybrid()
