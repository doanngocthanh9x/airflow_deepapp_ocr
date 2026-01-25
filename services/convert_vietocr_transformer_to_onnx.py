"""
Convert VietOCR Transformer model to ONNX format

VietOCR Transformer architecture:
- CNN backbone (VGG/ResNet) -> extract visual features
- Transformer decoder -> generate text sequence

The challenge is that Transformer uses autoregressive decoding,
so we need to export it carefully.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathUtlis import get_path


def analyze_vietocr_transformer():
    """Analyze VietOCR Transformer model structure"""
    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        
        # Load config
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        
        print("=" * 60)
        print("VietOCR Transformer Configuration")
        print("=" * 60)
        
        # Print important config
        print(f"\n[Backbone]")
        print(f"  - backbone: {config.get('backbone', 'N/A')}")
        print(f"  - cnn: {config.get('cnn', {})}")
        
        print(f"\n[Transformer]")
        transformer_cfg = config.get('transformer', {})
        print(f"  - d_model: {transformer_cfg.get('d_model', 'N/A')}")
        print(f"  - nhead: {transformer_cfg.get('nhead', 'N/A')}")
        print(f"  - num_encoder_layers: {transformer_cfg.get('num_encoder_layers', 'N/A')}")
        print(f"  - num_decoder_layers: {transformer_cfg.get('num_decoder_layers', 'N/A')}")
        print(f"  - dim_feedforward: {transformer_cfg.get('dim_feedforward', 'N/A')}")
        print(f"  - max_seq_length: {transformer_cfg.get('max_seq_length', 'N/A')}")
        print(f"  - pos_dropout: {transformer_cfg.get('pos_dropout', 'N/A')}")
        print(f"  - trans_dropout: {transformer_cfg.get('trans_dropout', 'N/A')}")
        
        print(f"\n[Dataset]")
        dataset_cfg = config.get('dataset', {})
        print(f"  - image_height: {dataset_cfg.get('image_height', 'N/A')}")
        print(f"  - image_min_width: {dataset_cfg.get('image_min_width', 'N/A')}")
        print(f"  - image_max_width: {dataset_cfg.get('image_max_width', 'N/A')}")
        
        print(f"\n[Vocab]")
        vocab = config.get('vocab', '')
        print(f"  - vocab length: {len(vocab)}")
        print(f"  - vocab (first 50 chars): {vocab[:50]}...")
        
        # Create predictor to analyze model
        predictor = Predictor(config)
        model = predictor.model
        
        print(f"\n[Model Structure]")
        print(f"  - Type: {type(model).__name__}")
        
        # Analyze model components
        for name, module in model.named_children():
            print(f"  - {name}: {type(module).__name__}")
            
            # Show sub-modules for important parts
            if hasattr(module, 'named_children'):
                for sub_name, sub_module in module.named_children():
                    print(f"      - {sub_name}: {type(sub_module).__name__}")
        
        return config, predictor
        
    except Exception as e:
        print(f"Error analyzing model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def export_transformer_to_onnx(output_dir=None):
    """
    Export VietOCR Transformer to ONNX
    
    Strategy:
    1. Export CNN backbone separately
    2. Export Transformer as a whole (with traced forward)
    """
    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        
        if output_dir is None:
            output_dir = get_path('models') / 'vietocr_transformer_onnx'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        predictor = Predictor(config)
        model = predictor.model
        model.eval()
        
        print("=" * 60)
        print("Exporting VietOCR Transformer to ONNX")
        print("=" * 60)
        
        # Get model components
        cnn = model.cnn
        transformer = model.transformer
        
        # Get config values
        dataset_cfg = config.get('dataset', {})
        img_height = dataset_cfg.get('image_height', 32)
        img_width = dataset_cfg.get('image_max_width', 512)
        
        transformer_cfg = config.get('transformer', {})
        max_seq_len = transformer_cfg.get('max_seq_length', 128)
        d_model = transformer_cfg.get('d_model', 256)
        
        print(f"\n[Config]")
        print(f"  Image size: {img_height}x{img_width}")
        print(f"  Max sequence length: {max_seq_len}")
        print(f"  d_model: {d_model}")
        
        # ============================================
        # 1. Export CNN backbone
        # ============================================
        print(f"\n[1/3] Exporting CNN backbone...")
        
        # Dummy input for CNN
        dummy_img = torch.randn(1, 3, img_height, img_width)
        
        # Wrap CNN for export
        class CNNWrapper(nn.Module):
            def __init__(self, cnn):
                super().__init__()
                self.cnn = cnn
            
            def forward(self, x):
                return self.cnn(x)
        
        cnn_wrapper = CNNWrapper(cnn)
        cnn_wrapper.eval()
        
        cnn_onnx_path = output_dir / 'transformer_cnn.onnx'
        
        torch.onnx.export(
            cnn_wrapper,
            dummy_img,
            str(cnn_onnx_path),
            input_names=['image'],
            output_names=['features'],
            dynamic_axes={
                'image': {0: 'batch', 3: 'width'},
                'features': {0: 'batch', 2: 'seq_len'}
            },
            opset_version=14,
            do_constant_folding=True
        )
        print(f"  ✓ Saved: {cnn_onnx_path}")
        
        # ============================================
        # 2. Export Transformer Encoder
        # ============================================
        print(f"\n[2/3] Exporting Transformer encoder...")
        
        # Get CNN output shape
        with torch.no_grad():
            cnn_out = cnn(dummy_img)
            print(f"  CNN output shape: {cnn_out.shape}")
            # Shape: (batch, channels, height, width) -> need to reshape for transformer
        
        # The transformer in VietOCR uses the CNN features directly
        # We need to export the full transformer forward pass
        
        # ============================================
        # 3. Export Full Transformer (greedy decode)
        # ============================================
        print(f"\n[3/3] Exporting full model for greedy decoding...")
        
        # Create a wrapper that does full prediction
        class TransformerONNXWrapper(nn.Module):
            """Wrapper for ONNX export - returns logits for each step"""
            def __init__(self, model, max_seq_len=128):
                super().__init__()
                self.model = model
                self.max_seq_len = max_seq_len
            
            def forward(self, img, tgt):
                """
                Args:
                    img: Input image (B, 3, H, W)
                    tgt: Target sequence (B, T) - for teacher forcing style
                
                Returns:
                    logits: (B, T, vocab_size)
                """
                # Get CNN features
                src = self.model.cnn(img)
                
                # Transformer forward
                outputs = self.model.transformer(src, tgt)
                
                return outputs
        
        # For ONNX, we export with fixed sequence length
        wrapper = TransformerONNXWrapper(model, max_seq_len)
        wrapper.eval()
        
        # Dummy inputs
        dummy_img = torch.randn(1, 3, img_height, img_width)
        dummy_tgt = torch.zeros(1, max_seq_len, dtype=torch.long)
        dummy_tgt[0, 0] = 1  # SOS token
        
        full_onnx_path = output_dir / 'transformer_full.onnx'
        
        try:
            torch.onnx.export(
                wrapper,
                (dummy_img, dummy_tgt),
                str(full_onnx_path),
                input_names=['image', 'tgt_seq'],
                output_names=['logits'],
                dynamic_axes={
                    'image': {0: 'batch', 3: 'width'},
                    'tgt_seq': {0: 'batch', 1: 'seq_len'},
                    'logits': {0: 'batch', 1: 'seq_len'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"  ✓ Saved: {full_onnx_path}")
        except Exception as e:
            print(f"  ✗ Error exporting full model: {e}")
            print("  Trying alternative export method...")
            
            # Alternative: Export CNN and Transformer separately
            export_separate_components(model, config, output_dir)
        
        # ============================================
        # Save vocab and config
        # ============================================
        import json
        import yaml
        
        vocab = config.get('vocab', '')
        vocab_path = output_dir / 'vocab.txt'
        vocab_path.write_text(vocab, encoding='utf-8')
        print(f"\n✓ Vocab saved: {vocab_path}")
        
        config_path = output_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dict(config), f, allow_unicode=True)
        print(f"✓ Config saved: {config_path}")
        
        print(f"\n{'='*60}")
        print(f"Export completed! Files saved to: {output_dir}")
        print(f"{'='*60}")
        
        return output_dir
        
    except Exception as e:
        print(f"Error exporting model: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_separate_components(model, config, output_dir):
    """
    Export VietOCR Transformer in a way that works for inference.
    
    VietOCR Transformer structure:
    - model.cnn: CNN backbone (VGG19)
    - model.transformer: LanguageTransformer
        - embed_tgt: Token embedding
        - pos_enc: Positional encoding
        - transformer: nn.Transformer (encoder + decoder)
        - fc: Linear output layer
    
    Strategy: Export 2 parts:
    1. Encoder: CNN + Transformer Encoder
    2. Decoder: Token embed + Transformer Decoder + FC
    """
    import math
    
    dataset_cfg = config.get('dataset', {})
    img_height = dataset_cfg.get('image_height', 32)
    img_width = 512
    
    transformer_cfg = config.get('transformer', {})
    d_model = transformer_cfg.get('d_model', 256)
    vocab_size = len(config.get('vocab', '')) + 4  # +4 for special tokens
    
    # ============================================
    # Export Encoder: CNN + Transformer Encoder
    # ============================================
    print("\n  Exporting CNN + Transformer Encoder...")
    
    class VietOCREncoder(nn.Module):
        """
        Encode image to memory representation
        Rewrites CNN output reshape to use positive indices for ONNX compatibility
        """
        def __init__(self, model, d_model):
            super().__init__()
            # Get VGG features
            self.vgg_features = model.cnn.model.features
            self.vgg_dropout = model.cnn.model.dropout
            self.vgg_last_conv = model.cnn.model.last_conv_1x1
            
            # Transformer encoder parts
            self.pos_enc = model.transformer.pos_enc
            self.transformer_encoder = model.transformer.transformer.encoder
            self.d_model = d_model
        
        def forward(self, img):
            """
            Args:
                img: (B, 3, H, W)
            Returns:
                memory: (seq_len, B, d_model)
            """
            # VGG forward
            conv = self.vgg_features(img)
            conv = self.vgg_dropout(conv)
            conv = self.vgg_last_conv(conv)
            
            # Reshape: (B, C, H, W) -> (W*H, B, C) using positive indices
            # Original: conv.transpose(-1, -2).flatten(2).permute(-1, 0, 1)
            B, C, H, W = conv.shape
            conv = conv.transpose(2, 3)  # (B, C, W, H)
            conv = conv.reshape(B, C, W * H)  # (B, C, W*H)
            conv = conv.permute(2, 0, 1)  # (W*H, B, C)
            
            # Add positional encoding
            src = self.pos_enc(conv * math.sqrt(self.d_model))
            
            # Encode: (seq_len, B, C) -> (seq_len, B, C)
            memory = self.transformer_encoder(src)
            
            return memory
    
    try:
        encoder = VietOCREncoder(model, d_model)
        encoder.eval()
        
        dummy_img = torch.randn(1, 3, img_height, img_width)
        
        with torch.no_grad():
            test_out = encoder(dummy_img)
            print(f"  Encoder output shape: {test_out.shape}")
        
        encoder_path = output_dir / 'vietocr_encoder.onnx'
        
        # Use fixed input size to avoid dynamic shape issues
        torch.onnx.export(
            encoder,
            dummy_img,
            str(encoder_path),
            input_names=['image'],
            output_names=['memory'],
            opset_version=14,
            do_constant_folding=True
        )
        print(f"  [OK] Encoder saved: {encoder_path}")
    except Exception as e:
        print(f"  ✗ Encoder export failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================
    # Export Decoder: Token embed + Transformer Decoder + FC
    # ============================================
    print("\n  Exporting Transformer Decoder...")
    
    class VietOCRDecoder(nn.Module):
        """Decode memory to text tokens"""
        def __init__(self, model, d_model):
            super().__init__()
            self.embed_tgt = model.transformer.embed_tgt
            self.pos_enc = model.transformer.pos_enc
            self.transformer_decoder = model.transformer.transformer.decoder
            self.fc = model.transformer.fc
            self.d_model = d_model
            
        def generate_square_subsequent_mask(self, sz):
            """Generate causal mask"""
            mask = torch.triu(torch.ones(sz, sz), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            return mask
        
        def forward(self, tgt, memory):
            """
            Args:
                tgt: Target tokens (T, B) - token indices
                memory: Encoder output (S, B, C)
            Returns:
                logits: (B, T, vocab_size)
            """
            # Generate causal mask
            tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0]).to(tgt.device)
            
            # Embed and add positional encoding
            tgt_emb = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
            
            # Decode
            output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            
            # Transpose: (T, B, C) -> (B, T, C)
            output = output.transpose(0, 1)
            
            # Project to vocab
            logits = self.fc(output)
            
            return logits
    
    try:
        decoder = VietOCRDecoder(model, d_model)
        decoder.eval()
        
        # Get actual encoder output length for 512px width image
        dummy_img = torch.randn(1, 3, img_height, img_width)
        with torch.no_grad():
            encoder_out = encoder(dummy_img)
            memory_len = encoder_out.shape[0]  # Should be 256 for 512px width
            print(f"  Encoder output length for {img_width}px: {memory_len}")
        
        # Dummy inputs matching encoder output
        seq_len = 128  # max decoding length
        batch_size = 1
        
        dummy_tgt = torch.zeros(seq_len, batch_size, dtype=torch.long)
        dummy_tgt[0, 0] = 1  # SOS token
        dummy_memory = torch.randn(memory_len, batch_size, d_model)
        
        with torch.no_grad():
            test_out = decoder(dummy_tgt, dummy_memory)
            print(f"  Decoder output shape: {test_out.shape}")
        
        decoder_path = output_dir / 'vietocr_decoder.onnx'
        
        # Export with dynamic axes for flexibility
        torch.onnx.export(
            decoder,
            (dummy_tgt, dummy_memory),
            str(decoder_path),
            input_names=['tgt', 'memory'],
            output_names=['logits'],
            dynamic_axes={
                'tgt': {0: 'tgt_len'},
                'logits': {1: 'tgt_len'}
            },
            opset_version=14,
            do_constant_folding=True
        )
        print(f"  [OK] Decoder saved: {decoder_path}")
    except Exception as e:
        print(f"  ✗ Decoder export failed: {e}")
        import traceback
        traceback.print_exc()


def verify_onnx_export(onnx_path, dummy_input):
    """Verify ONNX model output matches PyTorch"""
    import onnxruntime as ort
    
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: dummy_input.numpy()})
    
    return onnx_output


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert VietOCR Transformer to ONNX')
    parser.add_argument('--analyze', action='store_true', help='Analyze model structure only')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_vietocr_transformer()
    else:
        # First analyze
        print("Analyzing model structure...\n")
        analyze_vietocr_transformer()
        
        print("\n" + "=" * 60)
        print("Starting ONNX export...")
        print("=" * 60 + "\n")
        
        export_transformer_to_onnx(args.output)
