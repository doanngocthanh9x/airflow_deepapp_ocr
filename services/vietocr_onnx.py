"""
VietOCR ONNX Inference
Uses ONNX Runtime for fast Vietnamese OCR
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import onnxruntime
import os

# Suppress ONNX Runtime warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
onnxruntime.set_default_logger_severity(3)  # 3 = ERROR level only

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathUtlis import get_path


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
            else:
                # Skip invalid token index
                pass
        
        sent = ''.join(chars)
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4


class VietOCRVocabProvider:
    """Load Vietnamese character set from config"""
    @staticmethod
    def get_vocab(config_dir=None):
        """Load vocab from config files - must load base.yml first!"""
        if config_dir is None:
            # Try multiple paths
            paths_to_try = [
                get_path('models') / 'buiquangmanhhp1999' / 'config',  # New path
                get_path('webapp') / 'ConvertVietOcr2Onnx' / 'config',  # Old path
            ]
            
            config_dir = None
            for path in paths_to_try:
                if (path / 'base.yml').exists():
                    config_dir = path
                    break
            
            if config_dir is None:
                raise ValueError(f"Config directory not found! Tried: {paths_to_try}")
        
        import yaml
        
        # Load base.yml first (has vocab)
        base_yml = Path(config_dir) / 'base.yml'
        with open(base_yml, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Merge with vgg-seq2seq.yml if exists
        vgg_yml = Path(config_dir) / 'vgg-seq2seq.yml'
        if vgg_yml.exists():
            with open(vgg_yml, encoding='utf-8') as f:
                vgg_config = yaml.safe_load(f)
            if vgg_config:
                config.update(vgg_config)
        
        chars = config.get('vocab', '')
        if not chars:
            raise ValueError(f"Vocab not found in config! Config keys: {config.keys()}")
        
        return VietOCRVocab(chars)


class VietOCROnnxInference:
    """Vietnamese OCR using ONNX Runtime"""
    
    def __init__(self, cnn_model=None, encoder_model=None, decoder_model=None, vocab=None, config_dir=None):
        """
        Initialize ONNX models
        
        Args:
            cnn_model: Path to CNN ONNX model or model path
            encoder_model: Path to Encoder ONNX model
            decoder_model: Path to Decoder ONNX model
            vocab: VietOCRVocab instance
            config_dir: Path to config directory (for vocab loading)
        """
        # Use default paths if not provided
        if cnn_model is None:
            weight_dir = get_path('models') / 'buiquangmanhhp1999/weights'
            cnn_model = str(weight_dir / 'cnn.onnx')
            encoder_model = str(weight_dir / 'encoder.onnx')
            decoder_model = str(weight_dir / 'decoder.onnx')
        
        # Load ONNX sessions
        self.cnn_session = onnxruntime.InferenceSession(
            cnn_model,
            providers=['CPUExecutionProvider']  # Use CPU to avoid warnings
        )
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model,
            providers=['CPUExecutionProvider']
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model,
            providers=['CPUExecutionProvider']
        )
        
        # Load vocab
        if vocab is None:
            vocab = VietOCRVocabProvider.get_vocab(config_dir)
        self.vocab = vocab
        
        self.sos_token = 1
        self.eos_token = 2
        self.max_seq_length = 128
    
    def preprocess_image(self, image_path):
        """Preprocess image for CNN input"""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path
        
        # Resize to CNN input size (usually 3x32x320 for VietOCR)
        img = img.resize((320, 32), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)
        
        return img_array
    
    def __call__(self, image_path):
        """
        Run OCR on image
        
        Args:
            image_path: Path to image or PIL Image
            
        Returns:
            Recognized text
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # CNN forward pass - extract features
        cnn_input = {self.cnn_session.get_inputs()[0].name: img_array}
        cnn_outputs = self.cnn_session.run(None, cnn_input)
        src = cnn_outputs[0]  # Shape: (1, C, H, W)
        
        # Encoder forward pass - encode visual features
        encoder_input = {self.encoder_session.get_inputs()[0].name: src}
        encoder_outputs = self.encoder_session.run(None, encoder_input)
        encoder_out = encoder_outputs[0]  # encoder outputs
        hidden = encoder_outputs[1]        # hidden state
        
        # Decoding loop
        translated_sentence = [[self.sos_token]]  # Start with SOS token, shape: (1, 1)
        max_length = 0
        
        while max_length <= self.max_seq_length:
            # Prepare decoder input - last token(s)
            tgt_inp = np.array(translated_sentence[-1], dtype=np.int64)
            
            # Decoder forward pass
            decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
            decoder_inputs = {
                decoder_input_names[0]: tgt_inp,        # tgt
                decoder_input_names[1]: hidden,         # hidden
                decoder_input_names[2]: encoder_out    # encoder_outputs
            }
            
            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
            output = decoder_outputs[0]          # (batch, vocab_size) or similar
            hidden = decoder_outputs[1]          # updated hidden state
            
            # Get predicted token
            # output shape should be (1, vocab_size) or similar
            if output.ndim > 1:
                output = output[0]  # Take first element of batch
            
            predicted_token = int(np.argmax(output, axis=-1))
            
            translated_sentence.append([predicted_token])
            max_length += 1
            
            # Stop if EOS token
            if predicted_token == self.eos_token:
                break
        
        # Decode to text
        sentence_ids = np.concatenate(translated_sentence).flatten().tolist()
        text = self.vocab.decode(sentence_ids)
        
        return text


# Convenience function
def ocr_vietocr_onnx(image_path):
    """Quick OCR function using ONNX"""
    ocr = VietOCROnnxInference()
    return ocr(image_path)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Debug: Check vocab
        vocab = VietOCRVocabProvider.get_vocab()
        print(f"Vocab size: {len(vocab)}")
        print(f"i2c size: {len(vocab.i2c)}")
        print(f"First 10 i2c keys: {sorted(list(vocab.i2c.keys())[:10])}")
        
        result = ocr_vietocr_onnx(image_path)
        print(f"OCR Result: {result}")
    else:
        print("Usage: python vietocr_onnx.py <image_path>")
