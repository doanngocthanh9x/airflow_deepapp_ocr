"""
Test Script - Tìm tham số tối ưu cho Layout Detection
"""

import sys
import cv2
from simple_document_analyzer import SimpleDocumentAnalyzer


def test_parameters(analyzer, image_path, threshold_values, nms_iou_values):
    """
    Test với nhiều bộ tham số khác nhau
    """
    print("="*80)
    print("TESTING DIFFERENT PARAMETERS FOR LAYOUT DETECTION")
    print("="*80)
    
    # Load image once
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return
    
    results_summary = []
    
    for threshold in threshold_values:
        for nms_iou in nms_iou_values:
            print(f"\n{'='*80}")
            print(f"Testing: threshold={threshold}, nms_iou={nms_iou}")
            print(f"{'='*80}")
            
            result = analyzer.analyze(image, layout_threshold=threshold, nms_iou=nms_iou)
            
            num_layouts = len(result['layouts'])
            num_ocr = len(result['ocr_results'])
            
            print(f"Results:")
            print(f"  - Detected {num_layouts} layout regions")
            print(f"  - Detected {num_ocr} OCR text boxes")
            
            if num_layouts > 0:
                print(f"  - Layout types:")
                layout_types = {}
                for layout in result['layouts']:
                    ltype = layout['type']
                    layout_types[ltype] = layout_types.get(ltype, 0) + 1
                for ltype, count in layout_types.items():
                    print(f"    * {ltype}: {count}")
            
            # Table structure
            table_info = result.get('table_structure', {})
            print(f"  - Table structure: {table_info.get('num_rows', 0)} rows, {table_info.get('num_columns', 0)} columns")
            
            results_summary.append({
                'threshold': threshold,
                'nms_iou': nms_iou,
                'num_layouts': num_layouts,
                'num_ocr': num_ocr,
                'num_rows': table_info.get('num_rows', 0),
                'num_cols': table_info.get('num_columns', 0)
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*80}")
    print(f"{'Threshold':<12} {'NMS IoU':<10} {'Layouts':<10} {'OCR Boxes':<12} {'Rows':<8} {'Cols':<8}")
    print("-"*80)
    for r in results_summary:
        print(f"{r['threshold']:<12.2f} {r['nms_iou']:<10.2f} {r['num_layouts']:<10} {r['num_ocr']:<12} {r['num_rows']:<8} {r['num_cols']:<8}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Find best for layout detection (most layouts)
    best_layout = max(results_summary, key=lambda x: x['num_layouts'])
    print(f"For MOST layout detections:")
    print(f"  Use: threshold={best_layout['threshold']}, nms_iou={best_layout['nms_iou']}")
    print(f"  Result: {best_layout['num_layouts']} layouts")
    
    # Find best for table analysis (most rows detected)
    best_table = max(results_summary, key=lambda x: x['num_rows'])
    print(f"\nFor BEST table structure:")
    print(f"  Use: threshold={best_table['threshold']}, nms_iou={best_table['nms_iou']}")
    print(f"  Result: {best_table['num_rows']} rows, {best_table['num_cols']} columns")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_layout_params.py <model_dir> <image_path>")
        print("Example: python test_layout_params.py ./models/weights test_image.jpg")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    image_path = sys.argv[2]
    
    # Initialize analyzer
    print("Loading models...")
    analyzer = SimpleDocumentAnalyzer(model_dir, device="cpu")
    print("Models loaded!\n")
    
    # Test with different parameters
    threshold_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    nms_iou_values = [0.5, 0.6, 0.7, 0.8]
    
    test_parameters(analyzer, image_path, threshold_values, nms_iou_values)