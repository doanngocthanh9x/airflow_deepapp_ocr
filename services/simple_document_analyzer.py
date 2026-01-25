"""
Simple Document Analyzer - OPTIMIZED VERSION
✨ Improvements:
- Better bbox expansion for Vietnamese text with diacritics
- Batch OCR processing for speed
- Optimized preprocessing pipeline
- Better table structure detection
- Memory efficient processing
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image

# Import VietOCR options
try:
    from vietocr_onnx_optimized import VietOCROnnxInferenceOptimized
except ImportError:
    VietOCROnnxInferenceOptimized = None

try:
    from vietocr_transformer import VietOCRTransformerInference
except ImportError:
    VietOCRTransformerInference = None


class SimpleLayoutDetector:
    """Layout detection with optimizations"""
    
    LABELS = [
        "title", "text", "reference", "figure", "figure caption",
        "table", "table caption", "equation"
    ]
    
    def __init__(self, model_path: str, device: str = "cpu"):
        providers = ['CPUExecutionProvider']
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Enable graph optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:4]
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Optimized preprocessing"""
        h, w = image.shape[:2]
        target_h, target_w = self.input_shape
        
        # Convert BGR to RGB in one step
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Calculate scale
        r = min(target_h / h, target_w / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        
        # Resize
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        dw, dh = (target_w - new_w) / 2.0, (target_h - new_h) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add padding
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Normalize and transpose in one step
        img_normalized = (img_padded / 255.0).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        
        info = {
            'scale_factor': [w / new_w, h / new_h],
            'pad': [dw, dh],
            'orig_shape': [h, w]
        }
        
        return img_normalized, info
    
    def postprocess(self, outputs: np.ndarray, info: Dict, threshold: float = 0.3, nms_iou: float = 0.6) -> List[Dict]:
        """Optimized postprocessing with better NMS"""
        arr = np.squeeze(outputs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        results = []
        
        if arr.shape[1] == 6:
            # Filter by threshold
            mask = arr[:, 4] >= threshold
            arr = arr[mask]
            
            if arr.size == 0:
                return []
            
            xyxy = arr[:, :4].astype(np.float32)
            scores = arr[:, 4].astype(np.float32)
            cls_ids = arr[:, 5].astype(np.int32)
            
            # Remove padding and rescale
            dw, dh = info['pad']
            sx, sy = info['scale_factor']
            
            xyxy[:, [0, 2]] -= dw
            xyxy[:, [1, 3]] -= dh
            xyxy *= np.array([sx, sy, sx, sy], dtype=np.float32)
            
            # Apply NMS per class
            keep_indices = []
            for c in np.unique(cls_ids):
                idx = np.where(cls_ids == c)[0]
                k = self._nms(xyxy[idx], scores[idx], iou_threshold=nms_iou)
                keep_indices.extend(idx[k])
            
            # Build results
            for i in keep_indices:
                cid = int(cls_ids[i])
                if 0 <= cid < len(self.LABELS):
                    results.append({
                        "type": self.LABELS[cid].lower(),
                        "bbox": [float(x) for x in xyxy[i].tolist()],
                        "score": float(scores[i])
                    })
        
        return results
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
        """Optimized NMS"""
        if len(boxes) == 0:
            return []
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image: np.ndarray, threshold: float = 0.3, nms_iou: float = 0.6) -> List[Dict]:
        """Detect layouts in image"""
        input_tensor, info = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]
        results = self.postprocess(outputs, info, threshold, nms_iou)
        return results


class SimpleOCROptimized:
    """
    Optimized OCR with better Vietnamese support
    
    Improvements:
    - Better bbox expansion for diacritics (dấu thanh điệu)
    - Batch processing for speed
    - Direct PIL image handling (no temp files)
    - Better text filtering
    """
    
    def __init__(self, det_model_path: str, rec_model_path: str = None,
                 dict_path: str = None, device: str = "cpu",
                 bbox_expansion_horizontal: float = 0.12,
                 bbox_expansion_vertical: float = 0.30,
                 use_vietocr: str = "transformer"):
        """
        Args:
            det_model_path: Path to detection ONNX model
            rec_model_path: Path to recognition ONNX model
            dict_path: Path to character dictionary
            device: "cpu" or "cuda"
            bbox_expansion_horizontal: Horizontal expansion (default: 12%)
            bbox_expansion_vertical: Vertical expansion (default: 30% for Vietnamese diacritics)
            use_vietocr: Recognition model type:
                - "transformer": Use PyTorch Transformer (.pth) - default, best quality
                - "onnx": Use ONNX model - faster inference
                - False: Use standard recognition model
        """
        # Setup providers with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Load detection model
        self.det_session = ort.InferenceSession(det_model_path, sess_options=sess_options, providers=providers)
        
        # Load recognition model based on choice
        self.use_vietocr = use_vietocr
        
        if use_vietocr == "transformer":
            # Use PyTorch Transformer model - best quality for Vietnamese
            if VietOCRTransformerInference is None:
                raise ImportError("VietOCRTransformerInference not available. Install vietocr package.")
            self.vietocr = VietOCRTransformerInference(device=device)
            self.rec_session = None
            self.char_dict = None
            
        elif use_vietocr == "onnx":
            # Use ONNX model - faster inference
            if VietOCROnnxInferenceOptimized is None:
                raise ImportError("VietOCROnnxInferenceOptimized not available.")
            self.vietocr = VietOCROnnxInferenceOptimized()
            self.rec_session = None
            self.char_dict = None
            
        else:
            # Use standard recognition model
            self.vietocr = None
            self.rec_session = ort.InferenceSession(rec_model_path, sess_options=sess_options, providers=providers)
            self.rec_input_shape = [3, 48, 320]
            self.char_dict = self._load_dict(dict_path)
        
        self.drop_score = 0.5
        
        # Better expansion for Vietnamese text
        self.bbox_expansion_horizontal = bbox_expansion_horizontal
        self.bbox_expansion_vertical = bbox_expansion_vertical
    
    def _load_dict(self, dict_path: str) -> List[str]:
        """Load character dictionary"""
        chars = ['blank']
        with open(dict_path, 'rb') as f:
            for line in f.readlines():
                char = line.decode('utf-8').strip('\n').strip('\r\n')
                chars.append(char)
        chars.append(' ')
        return chars
    
    def _detect_text(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect text regions with optimized bbox expansion for Vietnamese
        """
        h, w = image.shape[:2]
        
        # Preprocess
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (960, 960))
        img = (img / 255.0 - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        
        # Run detection
        input_name = self.det_session.get_inputs()[0].name
        outputs = self.det_session.run(None, {input_name: img})[0]
        
        # Postprocess
        pred = outputs[0, 0, :, :]
        segmentation = pred > 0.3
        
        # Find contours
        contours, _ = cv2.findContours(
            (segmentation * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = self._order_points(box)
            
            # Scale back
            box[:, 0] = box[:, 0] * w / 960
            box[:, 1] = box[:, 1] * h / 960
            
            # === OPTIMIZED BBOX EXPANSION FOR VIETNAMESE ===
            # Calculate current box size
            box_width = np.linalg.norm(box[1] - box[0])
            box_height = np.linalg.norm(box[3] - box[0])
            
            # Increased vertical padding for Vietnamese diacritics
            # - Dấu thanh điệu phía trên: á, ắ, ấ, ậ, etc.
            # - Dấu câu phía dưới: phẩy, chấm, ngoặc
            # - Descenders: g, j, p, q, y, ỵ
            padding_x = box_width * self.bbox_expansion_horizontal
            padding_y = box_height * self.bbox_expansion_vertical  # 30% for better coverage
            
            # Unit vectors
            vec_h = box[1] - box[0]
            vec_h_norm = vec_h / (np.linalg.norm(vec_h) + 1e-6)
            
            vec_v = box[3] - box[0]
            vec_v_norm = vec_v / (np.linalg.norm(vec_v) + 1e-6)
            
            # Expand box
            expanded_box = box.copy()
            expanded_box[0] -= vec_h_norm * padding_x + vec_v_norm * padding_y
            expanded_box[1] += vec_h_norm * padding_x - vec_v_norm * padding_y
            expanded_box[2] += vec_h_norm * padding_x + vec_v_norm * padding_y
            expanded_box[3] -= vec_h_norm * padding_x - vec_v_norm * padding_y
            
            # Clip to image bounds
            expanded_box[:, 0] = np.clip(expanded_box[:, 0], 0, w - 1)
            expanded_box[:, 1] = np.clip(expanded_box[:, 1], 0, h - 1)
            
            # Filter small boxes
            if self._box_area(expanded_box) > 100:
                boxes.append(expanded_box.astype(np.float32))
        
        return self._sort_boxes(boxes)
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points clockwise"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _box_area(self, box: np.ndarray) -> float:
        """Calculate box area"""
        return cv2.contourArea(box)
    
    def _sort_boxes(self, boxes: List[np.ndarray]) -> List[np.ndarray]:
        """Sort boxes top-to-bottom, left-to-right"""
        if not boxes:
            return []
        return sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    
    def _get_rotate_crop_image(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Crop and rotate text region"""
        img_crop_width = int(max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        ))
        img_crop_height = int(max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        ))
        
        pts_std = np.float32([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ])
        
        M = cv2.getPerspectiveTransform(box, pts_std)
        dst_img = cv2.warpPerspective(
            img, M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        
        return dst_img
    
    def _recognize_text_batch(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Batch recognition for speed
        """
        if not img_list:
            return []
        
        if self.use_vietocr:
            # Convert numpy arrays to PIL Images for VietOCR
            pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in img_list]
            
            # Batch process
            texts = self.vietocr.recognize_batch(pil_images)
            
            # Return with confidence scores
            return [(text, 0.9 if text else 0.0) for text in texts]
        else:
            return self._recognize_text_standard(img_list)
    
    def _recognize_text_standard(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Standard recognition (fallback)"""
        if not img_list:
            return []
        
        # Preprocess batch
        batch_images = []
        for img in img_list:
            imgC, imgH, imgW = self.rec_input_shape
            h, w = img.shape[:2]
            
            ratio = w / float(h)
            resized_w = imgW if int(imgH * ratio) > imgW else int(imgH * ratio)
            
            resized_image = cv2.resize(img, (resized_w, imgH))
            resized_image = resized_image.astype('float32')
            
            resized_image = resized_image.transpose((2, 0, 1)) / 255
            resized_image = (resized_image - 0.5) / 0.5
            
            padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
            padding_im[:, :, 0:resized_w] = resized_image
            
            batch_images.append(padding_im)
        
        # Run batch inference
        batch_tensor = np.stack(batch_images, axis=0)
        input_name = self.rec_session.get_inputs()[0].name
        preds = self.rec_session.run(None, {input_name: batch_tensor})[0]
        
        return self._decode_predictions(preds)
    
    def _decode_predictions(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        """Decode CTC predictions"""
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        
        results = []
        for i in range(len(preds_idx)):
            char_list = []
            conf_list = []
            
            prev_idx = -1
            for j, idx in enumerate(preds_idx[i]):
                if idx != 0 and idx != prev_idx:
                    if idx < len(self.char_dict):
                        char_list.append(self.char_dict[idx])
                        conf_list.append(preds_prob[i][j])
                prev_idx = idx
            
            text = ''.join(char_list)
            score = np.mean(conf_list) if conf_list else 0.0
            
            results.append((text, float(score)))
        
        return results
    
    def ocr(self, image: np.ndarray) -> List[Dict]:
        """
        Perform OCR on image with batch processing
        """
        # Detect text boxes
        dt_boxes = self._detect_text(image)
        
        if not dt_boxes:
            return []
        
        # Crop text regions
        img_crop_list = [self._get_rotate_crop_image(image, box) for box in dt_boxes]
        
        # Batch recognize
        rec_results = self._recognize_text_batch(img_crop_list)
        
        # Build results
        results = []
        for box, (text, score) in zip(dt_boxes, rec_results):
            if score >= self.drop_score:
                x_coords = box[:, 0]
                y_coords = box[:, 1]
                bbox = [
                    float(x_coords.min()),
                    float(y_coords.min()),
                    float(x_coords.max()),
                    float(y_coords.max())
                ]
                
                results.append({
                    "text": text,
                    "bbox": bbox,
                    "score": score
                })
        
        return results


class SimpleDocumentAnalyzerOptimized:
    """
    Optimized Document Analyzer with batch processing and better Vietnamese support
    """
    
    def __init__(self, model_dir: str, device: str = "cpu",
                 bbox_expansion_horizontal: float = 0.12,
                 bbox_expansion_vertical: float = 0.30,
                 use_vietocr: Union[str, bool] = "transformer"):
        """
        Args:
            model_dir: Directory containing ONNX models
            device: Device to run on ("cpu" or "cuda")
            bbox_expansion_horizontal: Horizontal expansion (12% for better coverage)
            bbox_expansion_vertical: Vertical expansion (30% for Vietnamese diacritics)
            use_vietocr: VietOCR mode - "transformer" (PyTorch), "onnx", or False
        """
        self.layout_detector = SimpleLayoutDetector(
            model_path=os.path.join(model_dir, "layout.onnx"),
            device=device
        )
        
        self.ocr = SimpleOCROptimized(
            det_model_path=os.path.join(model_dir, "det.onnx"),
            rec_model_path=os.path.join(model_dir, "rec.onnx"),
            dict_path=os.path.join(model_dir, "ocr.res"),
            device=device,
            bbox_expansion_horizontal=bbox_expansion_horizontal,
            bbox_expansion_vertical=bbox_expansion_vertical,
            use_vietocr=use_vietocr
        )
    
    def analyze(self, image: np.ndarray, layout_threshold: float = 0.2, nms_iou: float = 0.6) -> Dict:
        """Analyze document with optimized processing"""
        # Detect layouts
        layouts = self.layout_detector.detect(image, threshold=layout_threshold, nms_iou=nms_iou)
        
        # Perform OCR
        ocr_results = self.ocr.ocr(image)
        
        # Analyze table structure
        table_structure = self.analyze_table_structure(ocr_results)
        
        return {
            "layouts": layouts,
            "ocr_results": ocr_results,
            "table_structure": table_structure
        }
    
    def analyze_image_file(self, image_path: str, layout_threshold: float = 0.2, nms_iou: float = 0.6) -> Dict:
        """Analyze document from file"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        return self.analyze(image, layout_threshold, nms_iou)
    
    def analyze_table_structure(self, ocr_results: List[Dict], tolerance: int = 15) -> Dict:
        """
        Enhanced table structure analysis with better row/column detection
        """
        if not ocr_results:
            return {"rows": [], "columns": [], "cells": []}
        
        # Sort by Y coordinate
        sorted_by_y = sorted(ocr_results, key=lambda x: x['bbox'][1])
        
        # Group into rows with adaptive tolerance
        rows = []
        current_row = [sorted_by_y[0]]
        current_y = sorted_by_y[0]['bbox'][1]
        
        for item in sorted_by_y[1:]:
            y = item['bbox'][1]
            if abs(y - current_y) <= tolerance:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x['bbox'][0])
                rows.append(current_row)
                current_row = [item]
                current_y = y
        
        if current_row:
            current_row.sort(key=lambda x: x['bbox'][0])
            rows.append(current_row)
        
        # Detect column boundaries
        all_x_coords = []
        for result in ocr_results:
            all_x_coords.extend([result['bbox'][0], result['bbox'][2]])
        
        all_x_coords = sorted(set(all_x_coords))
        
        # Group similar X coordinates
        columns = []
        if all_x_coords:
            current_col = [all_x_coords[0]]
            for x in all_x_coords[1:]:
                if abs(x - current_col[-1]) <= tolerance:
                    current_col.append(x)
                else:
                    columns.append(sum(current_col) / len(current_col))
                    current_col = [x]
            if current_col:
                columns.append(sum(current_col) / len(current_col))
        
        return {
            "rows": rows,
            "num_rows": len(rows),
            "num_columns": len(columns),
            "columns": columns
        }


def visualize_results(image: np.ndarray, result: Dict, output_path: str = None):
    """Visualize analysis results"""
    img_vis = image.copy()
    
    colors = {
        "title": (255, 0, 0),
        "text": (0, 255, 0),
        "table": (0, 0, 255),
        "figure": (255, 255, 0),
        "equation": (255, 0, 255),
    }
    
    # Draw layouts
    for layout in result["layouts"]:
        bbox = [int(v) for v in layout["bbox"]]
        x1, y1, x2, y2 = bbox
        
        color = colors.get(layout["type"], (128, 128, 128))
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        label = f"{layout['type']}: {layout['score']:.2f}"
        cv2.putText(img_vis, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw OCR boxes
    for ocr in result["ocr_results"]:
        bbox = [int(v) for v in ocr["bbox"]]
        x1, y1, x2, y2 = bbox
        
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        if ocr["text"]:
            cv2.putText(img_vis, ocr["text"][:30], (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, img_vis)
        print(f"✓ Saved visualization to {output_path}")
    
    return img_vis


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python simple_document_analyzer_optimized.py <model_dir> <image_path> [output_path]")
        print("\n✨ OPTIMIZED VERSION with:")
        print("  - Better Vietnamese diacritics support (30% vertical expansion)")
        print("  - Batch processing for speed")
        print("  - No temp file I/O")
        print("  - Optimized preprocessing pipeline")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output_optimized.jpg"
    
    print("🚀 Loading OPTIMIZED models...")
    analyzer = SimpleDocumentAnalyzerOptimized(
        model_dir,
        device="cpu",
        bbox_expansion_horizontal=0.12,  # 12%
        bbox_expansion_vertical=0.30     # 30% for Vietnamese
    )
    
    print(f"📄 Analyzing {image_path}...")
    result = analyzer.analyze_image_file(image_path, layout_threshold=0.2, nms_iou=0.6)
    
    print(f"\n=== Results ===")
    print(f"✓ Layouts: {len(result['layouts'])}")
    print(f"✓ OCR boxes: {len(result['ocr_results'])}")
    
    # Show first few OCR results
    for i, ocr in enumerate(result['ocr_results'][:5], 1):
        print(f"{i}. '{ocr['text']}'")
    
    # Visualize
    image = cv2.imread(image_path)
    visualize_results(image, result, output_path)