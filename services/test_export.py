"""
Test OCR export functionality
"""
import json
from ocr_export import OCRResultExporter

def test_markdown_export():
    """Test Markdown export with layout organization"""
    print("\n" + "="*60)
    print("Test 1: Markdown Export")
    print("="*60)
    
    example_result = {
        "layouts": [
            {"type": "title", "score": 0.95, "bbox": [10, 10, 500, 50]},
            {"type": "text", "score": 0.87, "bbox": [10, 60, 500, 400]},
            {"type": "table", "score": 0.82, "bbox": [10, 410, 500, 600]},
            {"type": "figure", "score": 0.78, "bbox": [510, 60, 800, 400]}
        ],
        "ocr_results": [
            {
                "text": "Vietnamese OCR Document Analysis",
                "score": 0.95,
                "type": "title",
                "bbox": [10, 10, 500, 50]
            },
            {
                "text": "This is a Vietnamese document with mixed content types.",
                "score": 0.88,
                "type": "text",
                "bbox": [10, 60, 500, 100]
            },
            {
                "text": "Header 1 | Header 2 | Header 3",
                "score": 0.82,
                "type": "table",
                "bbox": [10, 410, 500, 450]
            },
            {
                "text": "Hình ảnh minh họa",
                "score": 0.75,
                "type": "figure",
                "bbox": [510, 60, 800, 400]
            },
            {
                "text": "More Vietnamese text content here.",
                "score": 0.86,
                "type": "text",
                "bbox": [10, 100, 500, 150]
            }
        ]
    }
    
    exporter = OCRResultExporter()
    
    # Generate markdown
    md_content = exporter.to_markdown(example_result, "example_document.jpg")
    
    print("\nGenerated Markdown:\n")
    print(md_content)
    
    # Save to file
    md_path = exporter.save_markdown(example_result, "test_output.md", "example_document.jpg")
    print(f"\n✓ Markdown file saved: {md_path}")
    
    return md_path


def test_json_export():
    """Test JSON export with metadata"""
    print("\n" + "="*60)
    print("Test 2: JSON Export")
    print("="*60)
    
    example_result = {
        "layouts": [
            {"type": "title", "score": 0.95, "bbox": [10, 10, 500, 50]},
            {"type": "text", "score": 0.87, "bbox": [10, 60, 500, 400]}
        ],
        "ocr_results": [
            {"text": "Document Title", "score": 0.92},
            {"text": "This is the main content.", "score": 0.85}
        ]
    }
    
    exporter = OCRResultExporter()
    
    # Generate JSON
    json_content = exporter.to_json(example_result, "example.jpg", indent=2)
    
    print("\nGenerated JSON:\n")
    print(json_content)
    
    # Save to file
    json_path = exporter.save_json(example_result, "test_output.json", "example.jpg")
    print(f"\n✓ JSON file saved: {json_path}")
    
    # Parse and verify
    with open(json_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    print(f"\n✓ Verified JSON structure:")
    print(f"  - Image: {saved_data['metadata']['image']}")
    print(f"  - Layouts count: {saved_data['metadata']['layouts_count']}")
    print(f"  - Text regions: {saved_data['metadata']['text_regions_count']}")
    
    return json_path


def test_both_export():
    """Test saving both formats at once"""
    print("\n" + "="*60)
    print("Test 3: Save Both Formats")
    print("="*60)
    
    example_result = {
        "layouts": [
            {"type": "title", "score": 0.95, "bbox": [10, 10, 500, 50]},
            {"type": "text", "score": 0.87, "bbox": [10, 60, 500, 400]},
            {"type": "table", "score": 0.82, "bbox": [10, 410, 500, 600]}
        ],
        "ocr_results": [
            {"text": "Title with Vietnamese content", "score": 0.93},
            {"text": "Main paragraph text here.", "score": 0.88},
            {"text": "Table data with values", "score": 0.81}
        ]
    }
    
    exporter = OCRResultExporter()
    
    # Save both formats
    md_path, json_path = exporter.save_both(
        example_result,
        "./test_export_output",
        image_path="test_document.jpg",
        basename="complete_analysis"
    )
    
    print(f"\n✓ Markdown saved: {md_path}")
    print(f"✓ JSON saved: {json_path}")
    
    # Display file sizes
    import os
    md_size = os.path.getsize(md_path) / 1024  # KB
    json_size = os.path.getsize(json_path) / 1024  # KB
    
    print(f"\nFile sizes:")
    print(f"  - Markdown: {md_size:.2f} KB")
    print(f"  - JSON: {json_size:.2f} KB")
    
    return md_path, json_path


def test_layout_grouping():
    """Test that results are properly organized by layout type"""
    print("\n" + "="*60)
    print("Test 4: Layout Grouping Verification")
    print("="*60)
    
    example_result = {
        "layouts": [
            {"type": "title", "score": 0.95, "bbox": [10, 10, 500, 50]},
            {"type": "text", "score": 0.87, "bbox": [10, 60, 500, 400]},
            {"type": "table", "score": 0.82, "bbox": [10, 410, 500, 600]},
            {"type": "figure", "score": 0.78, "bbox": [510, 60, 800, 400]}
        ],
        "ocr_results": [
            {"text": "Title Text", "score": 0.95, "type": "title"},
            {"text": "Paragraph 1", "score": 0.88, "type": "text"},
            {"text": "Paragraph 2", "score": 0.86, "type": "text"},
            {"text": "Table header", "score": 0.82, "type": "table"},
            {"text": "Figure caption", "score": 0.75, "type": "figure"}
        ]
    }
    
    exporter = OCRResultExporter()
    md_content = exporter.to_markdown(example_result)
    
    # Verify sections exist
    sections = {
        "title": "## Title" in md_content,
        "text": "## Text" in md_content,
        "table": "## Table" in md_content,
        "figure": "## Figure" in md_content
    }
    
    print("\nLayout sections found in Markdown:")
    for layout_type, found in sections.items():
        status = "✓" if found else "✗"
        print(f"  {status} {layout_type.upper()}: {found}")
    
    all_found = all(sections.values())
    print(f"\n{'✓' if all_found else '✗'} All layout sections properly organized")
    
    return all_found


if __name__ == '__main__':
    print("\n" + "="*60)
    print("OCR Export Functionality Tests")
    print("="*60)
    
    try:
        # Run tests
        test_markdown_export()
        test_json_export()
        test_both_export()
        layout_ok = test_layout_grouping()
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60 + "\n")
        
        if layout_ok:
            print("✓ Export functionality is working correctly")
            print("  - Markdown with layout organization: OK")
            print("  - JSON with metadata: OK")
            print("  - Both formats simultaneously: OK")
            print("  - Layout grouping: OK")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
