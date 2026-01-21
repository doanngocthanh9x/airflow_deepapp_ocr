# OCR Export Feature - Complete Documentation

## 🎯 Overview

The OCR Export feature provides seamless conversion of OCR analysis results into two professional formats:

1. **Markdown** - Human-readable reports organized by layout type
2. **JSON** - Machine-readable structured data for databases and APIs

Both formats are automatically organized, include comprehensive metadata, and support Vietnamese text with proper UTF-8 encoding.

## 📦 What's Included

### Core Component
- **ocr_export.py** - OCRResultExporter class with 5 export methods

### Supporting Files
- **demo.py** - Updated with export integration example
- **test_export.py** - Comprehensive test suite (4 test functions)
- **validate_export.py** - System validation script

### Documentation
- **EXPORT_GUIDE.md** - Full user guide with examples
- **EXPORT_QUICKREF.md** - Quick reference card
- **EXPORT_IMPLEMENTATION_SUMMARY.md** - Implementation details

## 🚀 Quick Start

### 1. Basic Export

```python
from ocr_export import OCRResultExporter

exporter = OCRResultExporter()

# Export to Markdown
exporter.save_markdown(result, "output.md", image_path="doc.jpg")

# Export to JSON
exporter.save_json(result, "output.json", image_path="doc.jpg")

# Export both at once
md_path, json_path = exporter.save_both(result, "./output", image_path="doc.jpg")
```

### 2. With Document Analyzer

```python
from simple_document_analyzer import SimpleDocumentAnalyzerOptimized
from ocr_export import OCRResultExporter
import cv2

# Analyze
analyzer = SimpleDocumentAnalyzerOptimized(model_dir, device="cpu")
image = cv2.imread("document.jpg")
result = analyzer.analyze(image, layout_threshold=0.3)

# Export
exporter = OCRResultExporter()
md_path, json_path = exporter.save_both(
    result,
    output_dir="./results",
    image_path="document.jpg",
    basename="document_analysis"
)

print(f"✓ Markdown: {md_path}")
print(f"✓ JSON: {json_path}")
```

### 3. Batch Processing

```python
from pathlib import Path

docs_dir = Path("./documents")
results_dir = Path("./results")

for img_path in docs_dir.glob("*.jpg"):
    image = cv2.imread(str(img_path))
    result = analyzer.analyze(image)
    
    exporter.save_both(
        result,
        str(results_dir),
        image_path=str(img_path),
        basename=img_path.stem
    )
```

## 📊 API Reference

### OCRResultExporter Class

#### Static Methods

##### `to_markdown(result, image_path=None) -> str`
Converts OCR result to Markdown string organized by layout type.

```python
md_string = exporter.to_markdown(result, "image.jpg")
# Returns: Markdown formatted string
```

##### `to_json(result, image_path=None, indent=2) -> str`
Converts OCR result to JSON string with metadata.

```python
json_string = exporter.to_json(result, "image.jpg", indent=2)
# Returns: JSON formatted string
```

##### `save_markdown(result, output_path, image_path=None) -> str`
Saves OCR result as Markdown file.

```python
path = exporter.save_markdown(result, "output.md", "image.jpg")
# Returns: Path to saved file
# Creates: output.md with Markdown content
```

##### `save_json(result, output_path, image_path=None) -> str`
Saves OCR result as JSON file.

```python
path = exporter.save_json(result, "output.json", "image.jpg")
# Returns: Path to saved file
# Creates: output.json with JSON content
```

##### `save_both(result, output_dir, image_path=None, basename="ocr_result") -> tuple`
Saves both Markdown and JSON files simultaneously.

```python
md_path, json_path = exporter.save_both(
    result,
    "./output",
    image_path="image.jpg",
    basename="analysis"
)
# Returns: Tuple of (markdown_path, json_path)
# Creates: output/analysis.md and output/analysis.json
# Automatically creates output directory if needed
```

## 📋 Output Formats

### Markdown Example

```markdown
# OCR Analysis Report

## Metadata
- **Generated**: 2026-01-20 10:30:45
- **Image**: document.jpg
- **Layouts Detected**: 4
- **Text Regions**: 12

## Layout Detection

### 1. TITLE (Score: 95%)
- **BBox**: [10, 10, 500, 50]

### 2. TEXT (Score: 87%)
- **BBox**: [10, 60, 500, 400]

### 3. TABLE (Score: 82%)
- **BBox**: [10, 410, 500, 600]

### 4. FIGURE (Score: 78%)
- **BBox**: [510, 60, 800, 400]

## OCR Results by Layout Type

### Title
1. Vietnamese Document Title (confidence: 0.95)

### Text
1. This is the main content paragraph... (confidence: 0.88)
2. Another paragraph with more text... (confidence: 0.86)

### Table
1. Header | Value | Unit (confidence: 0.82)

### Figure
1. Figure caption or description (confidence: 0.75)
```

### JSON Example

```json
{
  "timestamp": "2026-01-20T10:30:45.123456",
  "metadata": {
    "image": "document.jpg",
    "layouts_count": 4,
    "text_regions_count": 12
  },
  "layouts": [
    {
      "type": "title",
      "score": 0.95,
      "bbox": [10, 10, 500, 50]
    },
    {
      "type": "text",
      "score": 0.87,
      "bbox": [10, 60, 500, 400]
    }
  ],
  "ocr_results": [
    {
      "text": "Vietnamese Document Title",
      "score": 0.95,
      "type": "title"
    },
    {
      "text": "Main content paragraph...",
      "score": 0.88,
      "type": "text"
    }
  ]
}
```

## 🧪 Testing & Validation

### Validate Installation
```bash
python validate_export.py
```

Checks:
- ✓ Required imports
- ✓ All files present
- ✓ OCRResultExporter methods
- ✓ demo.py integration
- ✓ Basic functionality

### Run Test Suite
```bash
python services/test_export.py
```

Tests:
1. Markdown export with layout organization
2. JSON export with metadata
3. Saving both formats simultaneously
4. Layout grouping verification

### Try Demo
```bash
python services/demo.py
```

Generates:
- `output_full_analysis.jpg` - Visualization
- `output_full_analysis.md` - Markdown report
- `output_full_analysis.json` - JSON data

## 🔄 Integration Examples

### With Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from services.simple_document_analyzer import SimpleDocumentAnalyzerOptimized
from services.ocr_export import OCRResultExporter
import cv2

def export_ocr_results(image_path, **context):
    analyzer = SimpleDocumentAnalyzerOptimized(model_dir, device="cpu")
    image = cv2.imread(image_path)
    result = analyzer.analyze(image)
    
    exporter = OCRResultExporter()
    md_path, json_path = exporter.save_both(
        result,
        "./dags/data/results",
        image_path=image_path
    )
    
    context['task_instance'].xcom_push(key='markdown', value=md_path)
    context['task_instance'].xcom_push(key='json', value=json_path)

with DAG('document_processing', start_date=datetime(2026, 1, 20)) as dag:
    export_task = PythonOperator(
        task_id='export_results',
        python_callable=export_ocr_results,
        op_kwargs={'image_path': '/path/to/image.jpg'}
    )
```

### With FastAPI

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from services.simple_document_analyzer import SimpleDocumentAnalyzerOptimized
from services.ocr_export import OCRResultExporter
import cv2
import numpy as np

app = FastAPI()
analyzer = SimpleDocumentAnalyzerOptimized(model_dir, device="cpu")
exporter = OCRResultExporter()

@app.post("/analyze-export")
async def analyze_and_export(file: UploadFile):
    # Read file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Analyze
    result = analyzer.analyze(image)
    
    # Export
    md_path, json_path = exporter.save_both(
        result,
        "./results",
        image_path=file.filename
    )
    
    return {
        "markdown": md_path,
        "json": json_path,
        "status": "success"
    }
```

### With Database Storage

```python
import json
import mysql.connector

def save_to_database(result, image_path):
    # Prepare JSON
    json_content = exporter.to_json(result, image_path)
    json_data = json.loads(json_content)
    
    # Save to MySQL
    conn = mysql.connector.connect(host="localhost", user="root", database="ocr_db")
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO analysis (image_path, result_json, timestamp) VALUES (%s, %s, %s)",
        (image_path, json.dumps(json_data), json_data['timestamp'])
    )
    
    conn.commit()
    cursor.close()
    conn.close()
```

## 💾 File Organization

```
c:\Automation\Airflow\
├── services/
│   ├── ocr_export.py              ← Main export class
│   ├── demo.py                    ← Updated with export integration
│   ├── test_export.py             ← Test suite
│   └── simple_document_analyzer.py ← Used by export
├── validate_export.py             ← Validation script
├── EXPORT_GUIDE.md                ← Full guide
├── EXPORT_QUICKREF.md             ← Quick reference
└── EXPORT_IMPLEMENTATION_SUMMARY.md ← Implementation details
```

## 🐛 Troubleshooting

### Issue: FileNotFoundError
**Solution**: Ensure output directory exists or use `save_both()` which auto-creates it.

```python
# Recommended
md_path, json_path = exporter.save_both(result, "./output")

# Or create directory manually
import os
os.makedirs("./output", exist_ok=True)
```

### Issue: Encoding Problems with Vietnamese Text
**Solution**: All methods use UTF-8 encoding by default.

```python
# Verify encoding when reading files
with open(md_path, 'r', encoding='utf-8') as f:
    content = f.read()
```

### Issue: Large JSON Files
**Solution**: JSON is larger than Markdown due to structure. This is normal.

```python
# For large documents, consider filtering or compressing
json_content = exporter.to_json(result, image_path)
# Use gzip if needed
import gzip
with gzip.open("output.json.gz", 'wt', encoding='utf-8') as f:
    f.write(json_content)
```

## ✨ Features

✅ **Layout Organization** - Results grouped by type (Title, Text, Table, Figure)
✅ **Metadata Tracking** - Timestamp, image path, element counts
✅ **Vietnamese Support** - Full UTF-8 encoding
✅ **Confidence Scores** - All scores included and preserved
✅ **Bounding Boxes** - Spatial information retained
✅ **Batch Processing** - Easy multi-file handling
✅ **Flexible API** - Use individual methods or combined `save_both()`
✅ **Auto Validation** - Included validation script
✅ **Comprehensive Tests** - Test suite provided
✅ **Professional Output** - Publication-ready Markdown and clean JSON

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| EXPORT_GUIDE.md | Complete user guide with advanced examples |
| EXPORT_QUICKREF.md | Quick reference for common tasks |
| EXPORT_IMPLEMENTATION_SUMMARY.md | What was implemented and how |
| README (this file) | Main documentation |

## 🎓 Learning Path

1. **Start Here**: Read this README
2. **Quick Tasks**: Check EXPORT_QUICKREF.md
3. **Full Details**: Read EXPORT_GUIDE.md
4. **Implementation**: Review EXPORT_IMPLEMENTATION_SUMMARY.md
5. **Hands-On**: Run `validate_export.py` and `test_export.py`
6. **Practice**: Run `python services/demo.py`

## 🔗 Related Files

- `c:\Automation\Airflow\services\simple_document_analyzer.py` - OCR analysis
- `c:\Automation\Airflow\services\vietocr_transformer.py` - Vietnamese OCR model
- `c:\Automation\Airflow\services\vietocr_onnx.py` - ONNX-based OCR
- `c:\Automation\Airflow\docker-compose.yaml` - Docker setup
- `c:\Automation\Airflow\requirements.txt` - Dependencies

## 🚀 Next Steps

1. ✓ Run `python validate_export.py` - Verify setup
2. ✓ Run `python services/test_export.py` - Test functionality
3. ✓ Run `python services/demo.py` - See working example
4. → Integrate into Airflow DAG
5. → Create FastAPI endpoint
6. → Display results in web UI
7. → Store in MySQL database

## 📞 Support

For issues:
1. Check EXPORT_GUIDE.md troubleshooting section
2. Run `validate_export.py` to verify setup
3. Review test results from `test_export.py`
4. Check demo output in `output_full_analysis.*` files

---

**Status**: ✅ READY FOR USE

All functionality implemented, tested, and documented.

