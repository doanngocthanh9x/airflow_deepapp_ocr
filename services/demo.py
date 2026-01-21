"""
Demo script để test Simple Document Analyzer
"""

import cv2
import os
import json
from datetime import datetime
from simple_document_analyzer import (
    SimpleDocumentAnalyzerOptimized,
    SimpleLayoutDetector,
    SimpleOCROptimized,
    visualize_results
)
from ocr_export import OCRResultExporter


def save_to_json(result, output_path):
    """Save analysis results to JSON file"""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "layouts": result['layouts'],
        "ocr_results": result['ocr_results']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ JSON saved to: {output_path}")


def save_to_markdown(result, output_path):
    """Save analysis results to Markdown file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Document Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Layout Results
        f.write("## Layout Detection Results\n\n")
        f.write(f"Found **{len(result['layouts'])}** layout regions:\n\n")
        
        for i, layout in enumerate(result['layouts'], 1):
            f.write(f"### {i}. {layout['type'].upper()}\n")
            f.write(f"- **Score:** {layout['score']:.4f}\n")
            f.write(f"- **BBox:** {layout.get('bbox', 'N/A')}\n\n")
        
        # OCR Results
        f.write("## OCR Results\n\n")
        f.write(f"Found **{len(result['ocr_results'])}** text boxes:\n\n")
        
        for i, ocr in enumerate(result['ocr_results'], 1):
            f.write(f"### {i}. Text Region\n")
            f.write(f"- **Text:** {ocr['text']}\n")
            f.write(f"- **Score:** {ocr['score']:.4f}\n")
            f.write(f"- **BBox:** {ocr.get('bbox', 'N/A')}\n\n")
    
    print(f"   ✓ Markdown saved to: {output_path}")


def demo_full_analysis():
    """Demo phân tích đầy đủ (Layout + OCR) với VietOCR Transformer"""
    print("\n" + "="*60)
    print("DEMO: Full Document Analysis (Layout + VietOCR Transformer)")
    print("="*60)
    
    # Đường dẫn models
    model_dir = r"C:\Automation\Airflow\models\hoaivannguyen\weights"
    
    # Khởi tạo analyzer với VietOCR transformer
    print(f"\n1. Loading models from: {model_dir}")
    print("   Using VietOCR Transformer for recognition...")
    analyzer = SimpleDocumentAnalyzerOptimized(
        model_dir, 
        device="cpu", 
        bbox_expansion_horizontal=0.20,
        bbox_expansion_vertical=0.30,
        use_vietocr="transformer"  # Sử dụng VietOCR transformer
    )
    print("   ✓ Models loaded successfully!")
    
    # Đọc ảnh test
    test_image = r"C:\Automation\Airflow\a15.png"
    if not os.path.exists(test_image):
        print(f"\n   ⚠ Test image not found: {test_image}")
        print("   Please provide a test image and update the path.")
        return
    
    print(f"\n2. Reading image: {test_image}")
    image = cv2.imread(test_image)
    print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Phân tích
    print(f"\n3. Analyzing document...")
    result = analyzer.analyze(image, layout_threshold=0.3)
    print(f"   ✓ Analysis complete!")
    
    # In kết quả layout
    print(f"\n4. Layout Detection Results:")
    print(f"   Found {len(result['layouts'])} layout regions")
    for i, layout in enumerate(result['layouts'], 1):
        print(f"   {i}. {layout['type'].upper()}: score={layout['score']:.3f}")
    
    # In kết quả OCR
    print(f"\n5. OCR Results:")
    print(f"   Found {len(result['ocr_results'])} text boxes")
    for i, ocr in enumerate(result['ocr_results'][:5], 1):  # Chỉ in 5 đầu tiên
        print(f"   {i}. '{ocr['text'][:50]}...' (score={ocr['score']:.3f})")
    if len(result['ocr_results']) > 5:
        print(f"   ... and {len(result['ocr_results']) - 5} more")
    
    # Visualize
    output_path = "output_full_analysis.jpg"
    print(f"\n6. Visualizing results to: {output_path}")
    visualize_results(image, result, output_path)
    print("   ✓ Visualization saved!")
    
    # Export results using OCRResultExporter
    print(f"\n7. Exporting results to Markdown and JSON...")
    exporter = OCRResultExporter()
    
    # Save Markdown with layout organization
    md_path = exporter.save_markdown(result, "output_full_analysis.md", image_path=test_image)
    print(f"   ✓ Markdown saved: {md_path}")
    
    # Save JSON
    json_path = exporter.save_json(result, "output_full_analysis.json", image_path=test_image)
    print(f"   ✓ JSON saved: {json_path}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("Files generated:")
    print(f"  - Visualization: {output_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {json_path}")
    print("="*60 + "\n")


def demo_layout_only():
    """Demo chỉ phát hiện layout"""
    print("\n" + "="*60)
    print("DEMO: Layout Detection Only")
    print("="*60)
    
    model_path = r"C:\Automation\Airflow\models\hoaivannguyen\weights\layout.onnx"
    
    print(f"\n1. Loading layout model: {model_path}")
    detector = SimpleLayoutDetector(model_path, device="cpu")
    print("   ✓ Model loaded!")
    
    test_image = "test_document.jpg"
    if not os.path.exists(test_image):
        print(f"\n   ⚠ Test image not found: {test_image}")
        return
    
    print(f"\n2. Reading image: {test_image}")
    image = cv2.imread(test_image)
    print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    print(f"\n3. Detecting layouts...")
    layouts = detector.detect(image, threshold=0.3)
    print(f"   ✓ Detection complete!")
    
    print(f"\n4. Results:")
    print(f"   Found {len(layouts)} layout regions:")
    for i, layout in enumerate(layouts, 1):
        bbox = layout['bbox']
        print(f"   {i}. {layout['type'].upper()}")
        print(f"      Score: {layout['score']:.3f}")
        print(f"      BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    print("\n" + "="*60 + "\n")


def demo_ocr_only():
    """Demo chỉ OCR"""
    print("\n" + "="*60)
    print("DEMO: OCR Only")
    print("="*60)
    
    model_dir = r"C:\Automation\Airflow\models\hoaivannguyen\weights"
    
    print(f"\n1. Loading OCR models from: {model_dir}")
    ocr = SimpleOCROptimized(
        det_model_path=os.path.join(model_dir, "det.onnx"),
        rec_model_path=os.path.join(model_dir, "rec.onnx"),
        dict_path=os.path.join(model_dir, "ocr.res"),
        device="cpu",
        use_vietocr="transformer" 
    )
    
    print("   ✓ Models loaded!")
    
    test_image = r"C:\Automation\Airflow\a14.png"
    if not os.path.exists(test_image):
        print(f"\n   ⚠ Test image not found: {test_image}")
        return
    
    print(f"\n2. Reading image: {test_image}")
    image = cv2.imread(test_image)
    print(f"   ✓ Image loaded!")
    
    print(f"\n3. Performing OCR...")
    ocr_results = ocr.ocr(image)
    print(f"   ✓ OCR complete!")
    
    print(f"\n4. Results:")
    print(f"   Found {len(ocr_results)} text boxes:")
    for i, result in enumerate(ocr_results[:10], 1):  # In 10 đầu tiên
        print(f"   {i}. Text: '{result['text']}'")
        print(f"      Score: {result['score']:.3f}")
    if len(ocr_results) > 10:
        print(f"   ... and {len(ocr_results) - 10} more")
    
    print("\n" + "="*60 + "\n")


def demo_batch_processing():
    """Demo xử lý nhiều ảnh với VietOCR Transformer"""
    print("\n" + "="*60)
    print("DEMO: Batch Processing with VietOCR Transformer")
    print("="*60)
    
    model_dir = r"C:\Automation\Airflow\models\hoaivannguyen\weights"
    
    print(f"\n1. Loading models...")
    print("   Using VietOCR Transformer for recognition...")
    analyzer = SimpleDocumentAnalyzerOptimized(
        model_dir, 
        device="cpu",
        bbox_expansion_horizontal=0.10,
        bbox_expansion_vertical=0.25,
        use_vietocr="transformer"  # Sử dụng VietOCR transformer
    )
    print("   ✓ Models loaded!")
    
    # Giả sử có nhiều ảnh trong thư mục
    import glob
    image_files = glob.glob(r"C:\Automation\Airflow\deepdoc_vietocr\img\*.jpg")
    
    if not image_files:
        print("\n   ⚠ No test images found in test_images/")
        print("   Create test_images/ folder and add some .jpg files")
        return
    
    print(f"\n2. Found {len(image_files)} images to process")
    
    print(f"\n3. Processing images...")
    for i, img_path in enumerate(image_files, 1):
        print(f"\n   [{i}/{len(image_files)}] Processing: {img_path}")
        
        try:
            result = analyzer.analyze_image_file(img_path)
            print(f"       ✓ Layouts: {len(result['layouts'])}, OCR: {len(result['ocr_results'])}")
            
            # Save visualization
            output_path = f"output_batch_{i}.jpg"
            image = cv2.imread(img_path)
            visualize_results(image, result, output_path)
            
        except Exception as e:
            print(f"       ✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")


def show_menu():
    """Hiển thị menu"""
    print("\n" + "="*60)
    print("Simple Document Analyzer - Demo Menu")
    print("="*60)
    print("\nChọn demo:")
    print("  1. Full Analysis (Layout + OCR)")
    print("  2. Layout Detection Only")
    print("  3. OCR Only")
    print("  4. Batch Processing")
    print("  5. Exit")
    print("="*60)
    
    choice = input("\nNhập lựa chọn (1-5): ").strip()
    return choice


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  Simple Document Analyzer - Demo Scripts               #")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    # Kiểm tra thư mục models
    model_dir = r"C:\Automation\Airflow\models\hoaivannguyen\weights"
    if not os.path.exists(model_dir):
        print(f"\n⚠ WARNING: Model directory not found!")
        print(f"   Expected: {model_dir}")
        print(f"   Please update the path in this script.")
        
        # Cho phép user nhập đường dẫn
        custom_path = input("\nEnter your model directory path (or press Enter to exit): ").strip()
        if custom_path:
            model_dir = custom_path
            # Update paths in demo functions
            print(f"\n✓ Using custom path: {model_dir}")
        else:
            print("\nExiting...")
            exit(0)
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            demo_full_analysis()
        elif choice == "2":
            demo_layout_only()
        elif choice == "3":
            demo_ocr_only()
        elif choice == "4":
            demo_batch_processing()
        elif choice == "5":
            print("\nGoodbye!")
            break
        else:
            print("\n⚠ Invalid choice. Please select 1-5.")
        
        input("\nPress Enter to continue...")