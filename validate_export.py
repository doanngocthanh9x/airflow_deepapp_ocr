#!/usr/bin/env python3
"""
Validation script for OCR export functionality
Ensures all components are in place and working
"""

import sys
import os
from pathlib import Path

def check_imports():
    """Check if all required imports are available"""
    print("\n" + "="*60)
    print("Checking Imports...")
    print("="*60)
    
    required_modules = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'PIL': 'Pillow',
        'json': 'JSON (built-in)',
        'pathlib': 'pathlib (built-in)',
        'datetime': 'datetime (built-in)'
    }
    
    missing = []
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name:20} available")
        except ImportError:
            print(f"✗ {name:20} MISSING")
            missing.append(name)
    
    return len(missing) == 0


def check_files():
    """Check if all required files exist"""
    print("\n" + "="*60)
    print("Checking Files...")
    print("="*60)
    
    required_files = {
        'services/ocr_export.py': 'OCRResultExporter class',
        'services/demo.py': 'Demo with export integration',
        'services/test_export.py': 'Export tests',
        'services/simple_document_analyzer.py': 'Document analyzer',
        'services/vietocr_transformer.py': 'VietOCR Transformer',
        'services/vietocr_onnx.py': 'VietOCR ONNX',
        'pathUtlis.py': 'Path management',
        'EXPORT_GUIDE.md': 'Export documentation'
    }
    
    base_path = Path('c:\\Automation\\Airflow')
    missing = []
    
    for filepath, description in required_files.items():
        full_path = base_path / filepath
        if full_path.exists():
            print(f"✓ {description:30} ({filepath})")
        else:
            print(f"✗ {description:30} ({filepath}) MISSING")
            missing.append(filepath)
    
    return len(missing) == 0


def check_ocr_export_class():
    """Check if OCRResultExporter has all required methods"""
    print("\n" + "="*60)
    print("Checking OCRResultExporter Class...")
    print("="*60)
    
    sys.path.insert(0, r'c:\Automation\Airflow\services')
    
    try:
        from ocr_export import OCRResultExporter
        
        required_methods = [
            'to_markdown',
            'to_json',
            'save_markdown',
            'save_json',
            'save_both'
        ]
        
        all_ok = True
        for method_name in required_methods:
            if hasattr(OCRResultExporter, method_name):
                method = getattr(OCRResultExporter, method_name)
                print(f"✓ {method_name:20} available")
            else:
                print(f"✗ {method_name:20} MISSING")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"✗ Error loading OCRResultExporter: {e}")
        return False


def check_demo_integration():
    """Check if demo.py has export integration"""
    print("\n" + "="*60)
    print("Checking Demo Integration...")
    print("="*60)
    
    demo_path = Path('c:\\Automation\\Airflow\\services\\demo.py')
    
    try:
        with open(demo_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'from ocr_export import OCRResultExporter': 'Import statement',
            'exporter = OCRResultExporter()': 'Exporter instantiation',
            'save_markdown': 'Markdown save',
            'save_json': 'JSON save',
            'demo_full_analysis': 'Main demo function'
        }
        
        all_ok = True
        for check_str, description in checks.items():
            if check_str in content:
                print(f"✓ {description:30} found in demo.py")
            else:
                print(f"✗ {description:30} NOT found in demo.py")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"✗ Error reading demo.py: {e}")
        return False


def check_functionality():
    """Test basic functionality"""
    print("\n" + "="*60)
    print("Testing Functionality...")
    print("="*60)
    
    sys.path.insert(0, r'c:\Automation\Airflow\services')
    
    try:
        from ocr_export import OCRResultExporter
        
        # Create test data
        test_result = {
            "layouts": [
                {"type": "title", "score": 0.95, "bbox": [10, 10, 500, 50]},
                {"type": "text", "score": 0.87, "bbox": [10, 60, 500, 400]}
            ],
            "ocr_results": [
                {"text": "Test Title", "score": 0.95},
                {"text": "Test content", "score": 0.88}
            ]
        }
        
        # Test markdown generation
        exporter = OCRResultExporter()
        md = exporter.to_markdown(test_result, "test.jpg")
        if "OCR Analysis Report" in md and "Layout Detection" in md:
            print("✓ Markdown generation works")
        else:
            print("✗ Markdown generation incomplete")
            return False
        
        # Test JSON generation
        js = exporter.to_json(test_result, "test.jpg")
        if '"timestamp"' in js and '"layouts"' in js and '"ocr_results"' in js:
            print("✓ JSON generation works")
        else:
            print("✗ JSON generation incomplete")
            return False
        
        # Test file saving
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = exporter.save_markdown(test_result, f"{tmpdir}\\test.md", "test.jpg")
            json_path = exporter.save_json(test_result, f"{tmpdir}\\test.json", "test.jpg")
            
            if Path(md_path).exists():
                print("✓ Markdown file saving works")
            else:
                print("✗ Markdown file saving failed")
                return False
            
            if Path(json_path).exists():
                print("✓ JSON file saving works")
            else:
                print("✗ JSON file saving failed")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks"""
    print("\n" + "="*70)
    print("OCR EXPORT FUNCTIONALITY - VALIDATION")
    print("="*70)
    
    results = {
        'Imports': check_imports(),
        'Files': check_files(),
        'OCRResultExporter Class': check_ocr_export_class(),
        'Demo Integration': check_demo_integration(),
        'Functionality': check_functionality()
    }
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All validation checks passed!")
        print("\nThe export functionality is ready to use.")
        print("\nNext steps:")
        print("1. Run: python services/test_export.py")
        print("2. Run: python services/demo.py")
        print("3. Check output files (*.md and *.json)")
        print("4. Integrate into Airflow DAG")
    else:
        print("✗ Some validation checks failed.")
        print("Please fix the issues above before proceeding.")
        return 1
    
    print("="*70 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
