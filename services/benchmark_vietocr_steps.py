"""
Benchmark từng bước của VietOCR Transformer: Encoder vs Decoder
So sánh PyTorch và ONNX để tìm hybrid tối ưu
"""
import time
import numpy as np
import torch
from PIL import Image
import onnxruntime as ort
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def benchmark_vietocr_steps():
    # ============================================
    # Setup PyTorch model
    # ============================================
    print('Loading PyTorch model...')
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    predictor = Predictor(config)
    model = predictor.model
    model.eval()

    # ============================================
    # Setup ONNX models
    # ============================================
    print('Loading ONNX models...')
    ort.set_default_logger_severity(3)
    encoder_sess = ort.InferenceSession(
        r'C:\Automation\Airflow\models\vietocr_transformer_onnx\vietocr_encoder.onnx',
        providers=['CPUExecutionProvider']
    )
    decoder_sess = ort.InferenceSession(
        r'C:\Automation\Airflow\models\vietocr_transformer_onnx\vietocr_decoder.onnx',
        providers=['CPUExecutionProvider']
    )

    # ============================================
    # Prepare test input
    # ============================================
    print('Preparing test input...')
    img = Image.open(r'C:\Automation\Airflow\image.png').convert('RGB')
    img = img.resize((512, 32), Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 32, 512)
    img_tensor = torch.from_numpy(img_np)

    n_runs = 10
    d_model = 256

    # ============================================
    # Benchmark ENCODER
    # ============================================
    print()
    print('='*60)
    print('ENCODER Benchmark (CNN + Transformer Encoder)')
    print('='*60)

    # PyTorch Encoder
    with torch.no_grad():
        # Warmup
        src = model.cnn(img_tensor)
        src_enc = model.transformer.pos_enc(src * math.sqrt(d_model))
        memory_pt = model.transformer.transformer.encoder(src_enc)
        
        start = time.time()
        for _ in range(n_runs):
            src = model.cnn(img_tensor)
            src_enc = model.transformer.pos_enc(src * math.sqrt(d_model))
            memory_pt = model.transformer.transformer.encoder(src_enc)
        pt_encoder_time = (time.time() - start) / n_runs * 1000
        print(f'PyTorch Encoder: {pt_encoder_time:.1f}ms')

    # ONNX Encoder
    _ = encoder_sess.run(None, {'image': img_np})  # Warmup
    start = time.time()
    for _ in range(n_runs):
        memory_onnx = encoder_sess.run(None, {'image': img_np})[0]
    onnx_encoder_time = (time.time() - start) / n_runs * 1000
    print(f'ONNX Encoder:    {onnx_encoder_time:.1f}ms')
    
    enc_winner = "ONNX" if onnx_encoder_time < pt_encoder_time else "PyTorch"
    enc_speedup = abs(pt_encoder_time/onnx_encoder_time - 1)*100 if onnx_encoder_time < pt_encoder_time else abs(onnx_encoder_time/pt_encoder_time - 1)*100
    print(f'Winner: {enc_winner} ({enc_speedup:.0f}% faster)')

    # ============================================
    # Benchmark DECODER (10 decode steps)
    # ============================================
    print()
    print('='*60)
    print('DECODER Benchmark (10 decode steps)')
    print('='*60)

    # Use memory from encoder
    memory_pt = memory_pt.detach()
    memory_np = memory_onnx

    # PyTorch Decoder - 10 steps
    with torch.no_grad():
        def pt_decode_step(tgt_tokens, memory):
            tgt = torch.tensor(tgt_tokens).unsqueeze(1).long()  # (T, 1)
            tgt_mask = model.transformer.gen_nopeek_mask(tgt.shape[0])
            tgt_emb = model.transformer.pos_enc(model.transformer.embed_tgt(tgt) * math.sqrt(d_model))
            output = model.transformer.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.transformer.fc(output.transpose(0, 1))
            return logits
        
        # Warmup
        _ = pt_decode_step([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], memory_pt)
        
        start = time.time()
        for _ in range(n_runs):
            for step in range(10):
                tgt = [1] + [0] * 9
                _ = pt_decode_step(tgt, memory_pt)
        pt_decoder_time = (time.time() - start) / n_runs * 1000
        print(f'PyTorch Decoder (10 steps): {pt_decoder_time:.1f}ms')

    # ONNX Decoder - 10 steps (must use full 128 tokens due to fixed shape)
    tgt_full = np.zeros((128, 1), dtype=np.int64)
    tgt_full[0, 0] = 1

    # Warmup
    _ = decoder_sess.run(None, {'tgt': tgt_full, 'memory': memory_np})

    start = time.time()
    for _ in range(n_runs):
        for step in range(10):
            _ = decoder_sess.run(None, {'tgt': tgt_full, 'memory': memory_np})
    onnx_decoder_time = (time.time() - start) / n_runs * 1000
    print(f'ONNX Decoder (10 steps):    {onnx_decoder_time:.1f}ms')
    
    dec_winner = "ONNX" if onnx_decoder_time < pt_decoder_time else "PyTorch"
    dec_speedup = abs(pt_decoder_time/onnx_decoder_time - 1)*100 if onnx_decoder_time < pt_decoder_time else abs(onnx_decoder_time/pt_decoder_time - 1)*100
    print(f'Winner: {dec_winner} ({dec_speedup:.0f}% faster)')

    # ============================================
    # SUMMARY
    # ============================================
    print()
    print('='*60)
    print('SUMMARY')
    print('='*60)
    print(f'Encoder: {enc_winner} is faster ({enc_speedup:.0f}%)')
    print(f'Decoder: {dec_winner} is faster ({dec_speedup:.0f}%)')
    print()
    
    # Calculate totals
    total_pt = pt_encoder_time + pt_decoder_time
    total_onnx = onnx_encoder_time + onnx_decoder_time
    hybrid_onnx_enc_pt_dec = onnx_encoder_time + pt_decoder_time
    hybrid_pt_enc_onnx_dec = pt_encoder_time + onnx_decoder_time
    
    print('Total times:')
    print(f'  Pure PyTorch:              {total_pt:.1f}ms')
    print(f'  Pure ONNX:                 {total_onnx:.1f}ms')
    print(f'  Hybrid (ONNX Enc + PT Dec): {hybrid_onnx_enc_pt_dec:.1f}ms')
    print(f'  Hybrid (PT Enc + ONNX Dec): {hybrid_pt_enc_onnx_dec:.1f}ms')
    print()
    
    # Find best
    options = {
        'Pure PyTorch': total_pt,
        'Pure ONNX': total_onnx,
        'Hybrid (ONNX Encoder + PyTorch Decoder)': hybrid_onnx_enc_pt_dec,
        'Hybrid (PyTorch Encoder + ONNX Decoder)': hybrid_pt_enc_onnx_dec
    }
    best = min(options, key=options.get)
    
    print('='*60)
    print(f'RECOMMENDATION: {best}')
    print(f'Estimated time: {options[best]:.1f}ms')
    print('='*60)
    
    return options


if __name__ == '__main__':
    benchmark_vietocr_steps()
