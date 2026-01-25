from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
requests.packages.urllib3.disable_warnings()

from PIL import Image
# Fix for Pillow 10+ compatibility (ANTIALIAS -> LANCZOS)
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

import cv2
import numpy as np
import json
import unicodedata
import os
import sys
import math
from pathlib import Path

flag_debug=False
def remove_accents(s: str) -> str:
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def get_cls_text(det_result):
    cls_results = []
    if det_result and det_result[0]:
        for line in det_result[0]:
            if len(line) > 2:
                angle = line[2]
                cls_results.append(angle)
            else:
                cls_results.append(None)
    return cls_results


def crop_image_perspective(image, points):
    """
    Crop image using perspective transform (better than simple crop)
    From service.py - handles rotated text boxes properly
    """
    points = np.array(points, dtype="float32")
    assert len(points) == 4, "shape of points must be 4*2"
    
    crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                         np.linalg.norm(points[2] - points[3])))
    crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                          np.linalg.norm(points[1] - points[2])))
    
    if crop_width == 0 or crop_height == 0:
        return None
        
    pts_std = np.float32([[0, 0],
                          [crop_width, 0],
                          [crop_width, crop_height],
                          [0, crop_height]])
    
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    cropped = cv2.warpPerspective(image, matrix, (crop_width, crop_height),
                                   borderMode=cv2.BORDER_REPLICATE, 
                                   flags=cv2.INTER_CUBIC)
    
    # Rotate if image is vertical (height > 1.5 * width)
    height, width = cropped.shape[0:2]
    if height * 1.0 / width >= 1.5:
        cropped = np.rot90(cropped, k=3)
    
    return cropped


def similarity_score(text1, text2):
    """Calculate similarity between two texts (0-1)"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    t1 = remove_accents(text1.lower().strip())
    t2 = remove_accents(text2.lower().strip())
    
    if not t1 or not t2:
        return 0.0
    
    # Simple character overlap score
    set1 = set(t1.replace(" ", ""))
    set2 = set(t2.replace(" ", ""))
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def find_best_rotation(crop_image, vietocr_predictor, paddle_text, debug_prefix=None, debug_path=None):
    """
    Try multiple rotations and find the one that gives VietOCR result 
    most similar to PaddleOCR result.
    
    Args:
        crop_image: Cropped image (RGB)
        vietocr_predictor: VietOCR predictor
        paddle_text: Reference text from PaddleOCR
        debug_prefix: Prefix for debug images
        debug_path: Path to save debug images
        
    Returns:
        (best_text, best_angle, best_score, all_results)
    """
    rotations = [
        (0, None),                    # No rotation
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    
    results = []
    
    for angle, rotate_code in rotations:
        # Rotate image
        if rotate_code is not None:
            rotated = cv2.rotate(crop_image, rotate_code)
        else:
            rotated = crop_image.copy()
        
        # Convert to PIL and run OCR
        pil_img = Image.fromarray(rotated)
        try:
            vietocr_text = vietocr_predictor.predict(pil_img)
        except Exception as e:
            vietocr_text = ""
        
        # Calculate similarity with PaddleOCR
        score = similarity_score(vietocr_text, paddle_text)
        
        results.append({
            'angle': angle,
            'vietocr': vietocr_text,
            'paddle': paddle_text,
            'score': score,
            'image': rotated
        })
        
        # Save debug image if requested
        if flag_debug:
            if debug_path and debug_prefix:
                img_path = debug_path / f"{debug_prefix}_rot{angle:03d}.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
    
    # Find best rotation (highest similarity score)
    best = max(results, key=lambda x: x['score'])
    
    return best['vietocr'], best['angle'], best['score'], results


def batch_find_best_rotation(cropped_images, vietocr_predictor, paddle_texts, 
                              debug_path=None, batch_size=16):
    """
    OPTIMIZED: Batch process all rotations for all images at once.
    
    Instead of running VietOCR 4 times per image (4 rotations),
    we prepare ALL rotated images first, then batch predict once.
    
    Args:
        cropped_images: List of cropped images (RGB)
        vietocr_predictor: VietOCR predictor
        paddle_texts: List of reference texts from PaddleOCR
        debug_path: Path to save debug images
        batch_size: Batch size for VietOCR prediction
        
    Returns:
        List of (best_text, best_angle, best_score, all_results) for each image
    """
    ROTATIONS = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    
    num_images = len(cropped_images)
    num_rotations = len(ROTATIONS)
    
    print(f"\n[BATCH] Preparing {num_images} images × {num_rotations} rotations = {num_images * num_rotations} total")
    
    # Step 1: Prepare all rotated images
    all_pil_images = []  # Flat list: [img1_rot0, img1_rot90, img1_rot180, img1_rot270, img2_rot0, ...]
    all_rotated_cv = []  # Keep cv2 images for debug saving
    
    for i, crop in enumerate(cropped_images):
        for angle, rotate_code in ROTATIONS:
            if rotate_code is not None:
                rotated = cv2.rotate(crop, rotate_code)
            else:
                rotated = crop.copy()
            
            all_rotated_cv.append(rotated)
            all_pil_images.append(Image.fromarray(rotated))
            
            # Save debug images
            if flag_debug:
             if debug_path:
                 img_path = debug_path / f"region_{i+1:02d}_rot{angle:03d}.png"
                 cv2.imwrite(str(img_path), cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
    
    # Step 2: Batch predict with VietOCR
    print(f"[BATCH] Running VietOCR batch prediction ({len(all_pil_images)} images)...")
    
    all_texts = []
    for batch_start in range(0, len(all_pil_images), batch_size):
        batch_end = min(batch_start + batch_size, len(all_pil_images))
        batch_images = all_pil_images[batch_start:batch_end]
        
        # Try batch predict if available, else fall back to single
        try:
            # VietOCR's predict_batch if available
            if hasattr(vietocr_predictor, 'predict_batch'):
                batch_texts = vietocr_predictor.predict_batch(batch_images)
            else:
                # Fallback: predict one by one
                batch_texts = [vietocr_predictor.predict(img) for img in batch_images]
        except Exception as e:
            print(f"[BATCH] Error in batch {batch_start}-{batch_end}: {e}")
            batch_texts = [""] * len(batch_images)
        
        all_texts.extend(batch_texts)
        print(f"[BATCH] Processed {batch_end}/{len(all_pil_images)} images")
    
    # Step 3: Organize results by image and find best rotation
    results_per_image = []
    
    for i in range(num_images):
        base_idx = i * num_rotations
        paddle_text = paddle_texts[i]
        
        rotations_results = []
        for j, (angle, _) in enumerate(ROTATIONS):
            idx = base_idx + j
            vietocr_text = all_texts[idx] if idx < len(all_texts) else ""
            score = similarity_score(vietocr_text, paddle_text)
            
            rotations_results.append({
                'angle': angle,
                'vietocr': vietocr_text,
                'paddle': paddle_text,
                'score': score
            })
        
        # Find best rotation
        best = max(rotations_results, key=lambda x: x['score'])
        
        results_per_image.append({
            'best_text': best['vietocr'],
            'best_angle': best['angle'],
            'best_score': best['score'],
            'all_rotations': rotations_results
        })
        
        # Print debug
        print(f"\n[Region {i+1}] PaddleOCR: '{paddle_text}'")
        for r in rotations_results:
            marker = " <-- BEST" if r['angle'] == best['angle'] else ""
            vtext = r['vietocr'][:30] + '...' if len(r['vietocr']) > 30 else r['vietocr']
            print(f"  {r['angle']:3d}°: VietOCR='{vtext}' score={r['score']:.3f}{marker}")
    
    return results_per_image


# ============================================================================
# CLASSIFICATION MODEL (from service.py) - DISABLED, using rotation search instead
# ============================================================================

class Classification:
    """Classify text orientation (0° or 180°) and rotate if needed"""
    
    def __init__(self, onnx_path):
        from onnxruntime import InferenceSession
        self.session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.threshold = 0.5  # Lower threshold to rotate more aggressively
        self.labels = ['0', '180']

    @staticmethod
    def resize(image):
        input_c = 3
        input_h = 48
        input_w = 192
        h = image.shape[0]
        w = image.shape[1]
        ratio = w / float(h)
        if math.ceil(input_h * ratio) > input_w:
            resized_w = input_w
        else:
            resized_w = int(math.ceil(input_h * ratio))
        resized_image = cv2.resize(image, (resized_w, input_h))

        if input_c == 1:
            resized_image = resized_image[np.newaxis, :]

        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padded_image = np.zeros((input_c, input_h, input_w), dtype=np.float32)
        padded_image[:, :, 0:resized_w] = resized_image
        return padded_image

    def __call__(self, images):
        """Classify and rotate images if needed"""
        num_images = len(images)
        results = [['', 0.0]] * num_images
        raw_outputs_all = [None] * num_images  # Store raw outputs for debug
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))

        batch_size = 6
        for i in range(0, num_images, batch_size):
            norm_images = []
            for j in range(i, min(num_images, i + batch_size)):
                norm_img = self.resize(images[indices[j]])
                norm_img = norm_img[np.newaxis, :]
                norm_images.append(norm_img)
            norm_images = np.concatenate(norm_images)

            raw_outputs = self.session.run(None, {self.inputs.name: norm_images})[0]
            
            # Debug: print raw outputs
            print(f"\n[CLS DEBUG] Batch {i//batch_size + 1}:")
            print(f"  Raw output shape: {raw_outputs.shape}")
            print(f"  Labels: {self.labels}")
            
            for k in range(len(raw_outputs)):
                orig_idx = indices[i + k]
                prob_0 = raw_outputs[k, 0]
                prob_180 = raw_outputs[k, 1]
                pred_idx = raw_outputs[k].argmax()
                pred_label = self.labels[pred_idx]
                pred_conf = raw_outputs[k, pred_idx]
                
                print(f"  Region {orig_idx+1}: prob_0°={prob_0:.4f}, prob_180°={prob_180:.4f} -> {pred_label}° (conf={pred_conf:.4f})")
                
                results[orig_idx] = [pred_label, float(pred_conf)]
                raw_outputs_all[orig_idx] = (prob_0, prob_180)
                
                # Rotate if 180° with high confidence
                if pred_label == '180' and pred_conf > self.threshold:
                    print(f"    -> ROTATING image {orig_idx+1}")
                    images[orig_idx] = cv2.rotate(images[orig_idx], cv2.ROTATE_180)
        
        return images, results
          

def run_ocr():
    from paddleocr import PaddleOCR
    
    # Add project root to path for imports
    sys.path.insert(0, '/opt/airflow')
    from services.vietcache import get_predictor
    
    # Use paths
    models_path = Path('/opt/airflow/models')
    dags_path = Path('/opt/airflow/dags')
    logs_path = Path('/opt/airflow/logs')
    
    img_path = str(dags_path / '333.png')
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    print(f"Processing image: {img_path}")
    
    # PaddleOCR for detection + recognition
    print("Loading PaddleOCR...")
    paddle_ocr = PaddleOCR(lang='en', use_gpu=False)
    det_result = paddle_ocr.ocr(img_path, det=True, rec=True, cls=True)
    image = cv2.imread(img_path)

    # VietOCR from local cache (no download!)
    print("Loading VietOCR from local cache...")
    vietocr = get_predictor(name='vgg_transformer', device='cpu')
    print("VietOCR loaded successfully!")

    final_results = []
    cls_results = get_cls_text(det_result)
    
    # Create debug folder for saving cropped images
    debug_path = logs_path / 'debug_crops'
    debug_path.mkdir(parents=True, exist_ok=True)
    print(f"Debug images will be saved to: {debug_path}")
    
    print("is debug?: " + str(flag_debug))

    if det_result and det_result[0]:
        # Step 1: Crop all images using perspective transform
        cropped_images = []
        boxes = []
        paddle_texts = []
        
        # Convert image to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for i, line in enumerate(det_result[0]):
            box = line[0]
            pts = np.array(box, dtype="float32")
            
            # Use perspective transform instead of simple crop
            crop = crop_image_perspective(image_rgb, pts)
            if crop is None or crop.size == 0:
                continue
                
            cropped_images.append(crop)
            boxes.append(box)
            paddle_texts.append(line[1][0] if line[1] else "")
        
        print(f"Detected {len(cropped_images)} text regions")
        
        # Step 2: BATCH find best rotation for all images
        print("\n=== BATCH ROTATION SEARCH (VietOCR vs PaddleOCR comparison) ===")
        
        batch_results = batch_find_best_rotation(
            cropped_images, 
            vietocr, 
            paddle_texts,
            debug_path=debug_path,
            batch_size=16  # Process 16 images at a time
        )
        
        # Extract results
        vietocr_texts = [r['best_text'] for r in batch_results]
        rotation_info = batch_results
        
        # Step 3: Compile results
        for i in range(len(cropped_images)):
            paddle_cls = cls_results[i] if i < len(cls_results) else None
            rot_info = rotation_info[i]
            
            result = {
                "box": boxes[i], 
                "VietOCR": vietocr_texts[i], 
                "PaddleOCR": paddle_texts[i],
                "rotation": {
                    "best_angle": rot_info['best_angle'],
                    "similarity_score": rot_info['best_score'],
                    "all_rotations": rot_info['all_rotations']
                },
                "PaddleCLS": str(paddle_cls) if paddle_cls else None
            }
            final_results.append(result)
            
            print(f"\n[FINAL Region {i+1}]")
            print(f"  VietOCR: {vietocr_texts[i]}")
            print(f"  PaddleOCR: {paddle_texts[i]}")
            print(f"  Best Rotation: {rot_info['best_angle']}° (similarity={rot_info['best_score']:.3f})")
    
    # Save results
    output_file = str(logs_path / 'ocr_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"\nOCR processing completed. Results saved to: {output_file}")
    
    return final_results


with DAG(
    dag_id="com.batch.ocr.vietocr.classification",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ocr", "vietocr", "classification"],
) as dag:
    ocr_task = PythonOperator(
        task_id="run_ocr_task",
        python_callable=run_ocr
    )