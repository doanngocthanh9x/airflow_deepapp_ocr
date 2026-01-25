"""
Export utilities for OCR results
Convert OCR results to Markdown and JSON formats
"""
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np


class OCRResultExporter:
    """Export OCR results to various formats"""
    
    @staticmethod
    def _is_point_in_bbox(point: Tuple[float, float], bbox: List[float], margin: float = 10) -> bool:
        """Check if a point is inside a bounding box with margin"""
        x, y = point
        x1, y1, x2, y2 = bbox
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)
    
    @staticmethod
    def _get_ocr_center(ocr_bbox: List) -> Tuple[float, float]:
        """Get center point of OCR bounding box"""
        if isinstance(ocr_bbox, list) and len(ocr_bbox) >= 4:
            # Handle polygon format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if isinstance(ocr_bbox[0], list):
                xs = [p[0] for p in ocr_bbox]
                ys = [p[1] for p in ocr_bbox]
                return (sum(xs) / len(xs), sum(ys) / len(ys))
            # Handle [x1, y1, x2, y2] format
            else:
                x1, y1, x2, y2 = ocr_bbox[:4]
                return ((x1 + x2) / 2, (y1 + y2) / 2)
        return (0, 0)
    
    @staticmethod
    def _match_ocr_to_layouts(result: Dict) -> Dict[int, List[Dict]]:
        """
        Match OCR results to layout regions based on bounding box overlap
        
        Returns:
            Dict mapping layout index to list of OCR results
        """
        layouts = result.get('layouts', [])
        ocr_results = result.get('ocr_results', [])
        
        # Dict to store matched OCR for each layout
        layout_ocr_map = {i: [] for i in range(len(layouts))}
        unmatched_ocr = []
        
        for ocr in ocr_results:
            ocr_bbox = ocr.get('bbox', [])
            center = OCRResultExporter._get_ocr_center(ocr_bbox)
            
            matched = False
            for i, layout in enumerate(layouts):
                layout_bbox = layout.get('bbox', [])
                if layout_bbox and OCRResultExporter._is_point_in_bbox(center, layout_bbox):
                    layout_ocr_map[i].append(ocr)
                    matched = True
                    break
            
            if not matched:
                unmatched_ocr.append(ocr)
        
        # Add unmatched as special key -1
        layout_ocr_map[-1] = unmatched_ocr
        
        return layout_ocr_map
    
    @staticmethod
    def _format_table_markdown(ocr_list: List[Dict]) -> str:
        """Format OCR results as markdown table (attempt to detect rows/columns)"""
        if not ocr_list:
            return "*No text detected in table*"
        
        # Sort by Y position first, then X position
        def get_y_position(ocr):
            bbox = ocr.get('bbox', [])
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], list):
                    return bbox[0][1]  # First point Y
                elif len(bbox) >= 2:
                    return bbox[1]  # y1 in [x1, y1, x2, y2]
            return 0
        
        def get_x_position(ocr):
            bbox = ocr.get('bbox', [])
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], list):
                    return bbox[0][0]  # First point X
                elif len(bbox) >= 1:
                    return bbox[0]  # x1 in [x1, y1, x2, y2]
            return 0
        
        # Sort OCR results by position
        sorted_ocr = sorted(ocr_list, key=lambda x: (get_y_position(x), get_x_position(x)))
        
        # Group by rows (similar Y position)
        rows = []
        current_row = []
        last_y = None
        y_threshold = 20  # Threshold to consider same row
        
        for ocr in sorted_ocr:
            y = get_y_position(ocr)
            if last_y is None or abs(y - last_y) < y_threshold:
                current_row.append(ocr)
            else:
                if current_row:
                    # Sort row by X position
                    current_row.sort(key=get_x_position)
                    rows.append(current_row)
                current_row = [ocr]
            last_y = y
        
        if current_row:
            current_row.sort(key=get_x_position)
            rows.append(current_row)
        
        # Build markdown table
        if not rows:
            return "*No text detected in table*"
        
        # Determine max columns
        max_cols = max(len(row) for row in rows)
        
        md_lines = []
        
        # Table header (first row)
        if rows:
            header_texts = [ocr.get('text', '').strip() for ocr in rows[0]]
            while len(header_texts) < max_cols:
                header_texts.append('')
            md_lines.append("| " + " | ".join(header_texts) + " |")
            md_lines.append("|" + "|".join(["---"] * max_cols) + "|")
        
        # Table body (remaining rows)
        for row in rows[1:]:
            row_texts = [ocr.get('text', '').strip() for ocr in row]
            while len(row_texts) < max_cols:
                row_texts.append('')
            md_lines.append("| " + " | ".join(row_texts) + " |")
        
        return "\n".join(md_lines)
    
    @staticmethod
    def to_markdown(result: Dict, image_path: str = None) -> str:
        """
        Convert OCR result to Markdown format organized by layout
        
        Args:
            result: OCR analysis result dict
            image_path: Path to the analyzed image
            
        Returns:
            Markdown string
        """
        md = []
        
        # Header
        md.append("# OCR Analysis Report")
        md.append("")
        
        # Metadata
        md.append("## Metadata")
        md.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if image_path:
            md.append(f"- **Image**: {image_path}")
        md.append(f"- **Layouts Detected**: {len(result.get('layouts', []))}")
        md.append(f"- **Text Regions**: {len(result.get('ocr_results', []))}")
        md.append("")
        
        # Match OCR to layouts
        layout_ocr_map = OCRResultExporter._match_ocr_to_layouts(result)
        
        # Layout sections with matched OCR content
        layouts = result.get('layouts', [])
        if layouts:
            md.append("## Document Content")
            md.append("")
            
            for i, layout in enumerate(layouts):
                layout_type = layout.get('type', 'unknown').lower()
                score = layout.get('score', 0)
                bbox = layout.get('bbox', [])
                matched_ocr = layout_ocr_map.get(i, [])
                
                # Format layout header
                md.append(f"### {i+1}. {layout_type.upper()} (Score: {score:.2%})")
                md.append(f"> Bounding Box: `{[round(x, 1) for x in bbox]}`")
                md.append("")
                
                if not matched_ocr:
                    md.append("*No text detected in this region*")
                    md.append("")
                    continue
                
                # Format content based on layout type
                if layout_type == 'table':
                    md.append(OCRResultExporter._format_table_markdown(matched_ocr))
                    md.append("")
                    
                elif layout_type == 'title':
                    # Combine all text as title
                    title_text = " ".join([ocr.get('text', '').strip() for ocr in matched_ocr])
                    md.append(f"**{title_text}**")
                    md.append("")
                    
                elif layout_type in ['text', 'reference', 'figure caption', 'table caption']:
                    # Combine as paragraph
                    paragraph_text = " ".join([ocr.get('text', '').strip() for ocr in matched_ocr])
                    md.append(paragraph_text)
                    md.append("")
                    
                elif layout_type in ['figure', 'equation']:
                    md.append(f"*[{layout_type.upper()} detected]*")
                    if matched_ocr:
                        text = " ".join([ocr.get('text', '').strip() for ocr in matched_ocr])
                        if text:
                            md.append(f"Text: {text}")
                    md.append("")
                    
                else:
                    # Default: list all text
                    for ocr in matched_ocr:
                        text = ocr.get('text', '').strip()
                        if text:
                            md.append(f"- {text}")
                    md.append("")
        
        # Unmatched OCR results
        unmatched = layout_ocr_map.get(-1, [])
        if unmatched:
            md.append("## Other Detected Text")
            md.append("")
            for i, ocr in enumerate(unmatched, 1):
                text = ocr.get('text', '').strip()
                score = ocr.get('score', 0)
                if text:
                    md.append(f"{i}. {text} *(confidence: {score:.2%})*")
            md.append("")
        
        return "\n".join(md)
    
    @staticmethod
    def to_json(result: Dict, image_path: str = None, indent: int = 2) -> str:
        """
        Convert OCR result to JSON format with layout-matched OCR
        
        Args:
            result: OCR analysis result dict
            image_path: Path to the analyzed image
            indent: JSON indentation level
            
        Returns:
            JSON string
        """
        # Match OCR to layouts
        layout_ocr_map = OCRResultExporter._match_ocr_to_layouts(result)
        
        # Build structured layout data with matched OCR
        layouts_with_content = []
        for i, layout in enumerate(result.get('layouts', [])):
            matched_ocr = layout_ocr_map.get(i, [])
            layout_data = {
                "index": i + 1,
                "type": layout.get('type', 'unknown'),
                "score": layout.get('score', 0),
                "bbox": layout.get('bbox', []),
                "ocr_content": [
                    {
                        "text": ocr.get('text', ''),
                        "score": ocr.get('score', 0),
                        "bbox": ocr.get('bbox', [])
                    }
                    for ocr in matched_ocr
                ],
                "combined_text": " ".join([ocr.get('text', '').strip() for ocr in matched_ocr])
            }
            
            # For tables, add table structure
            if layout.get('type', '').lower() == 'table':
                layout_data["table_structure"] = OCRResultExporter._extract_table_structure(matched_ocr)
            
            layouts_with_content.append(layout_data)
        
        # Unmatched OCR
        unmatched = layout_ocr_map.get(-1, [])
        
        export_data = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "image_path": str(image_path) if image_path else None,
                "layouts_count": len(result.get('layouts', [])),
                "text_regions_count": len(result.get('ocr_results', []))
            },
            "document_content": layouts_with_content,
            "unmatched_ocr": [
                {
                    "text": ocr.get('text', ''),
                    "score": ocr.get('score', 0),
                    "bbox": ocr.get('bbox', [])
                }
                for ocr in unmatched
            ],
            "raw_layouts": result.get('layouts', []),
            "raw_ocr_results": result.get('ocr_results', [])
        }
        
        # Convert floats to be JSON serializable
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(v) for v in obj]
            elif isinstance(obj, float):
                return round(obj, 4)
            return obj
        
        export_data = convert_floats(export_data)
        
        return json.dumps(export_data, ensure_ascii=False, indent=indent)
    
    @staticmethod
    def _extract_table_structure(ocr_list: List[Dict]) -> Dict:
        """Extract table structure from OCR results (rows and columns)"""
        if not ocr_list:
            return {"rows": [], "row_count": 0, "col_count": 0}
        
        def get_y_position(ocr):
            bbox = ocr.get('bbox', [])
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], list):
                    return bbox[0][1]
                elif len(bbox) >= 2:
                    return bbox[1]
            return 0
        
        def get_x_position(ocr):
            bbox = ocr.get('bbox', [])
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], list):
                    return bbox[0][0]
                elif len(bbox) >= 1:
                    return bbox[0]
            return 0
        
        sorted_ocr = sorted(ocr_list, key=lambda x: (get_y_position(x), get_x_position(x)))
        
        rows = []
        current_row = []
        last_y = None
        y_threshold = 20
        
        for ocr in sorted_ocr:
            y = get_y_position(ocr)
            if last_y is None or abs(y - last_y) < y_threshold:
                current_row.append(ocr)
            else:
                if current_row:
                    current_row.sort(key=get_x_position)
                    rows.append([ocr.get('text', '').strip() for ocr in current_row])
                current_row = [ocr]
            last_y = y
        
        if current_row:
            current_row.sort(key=get_x_position)
            rows.append([ocr.get('text', '').strip() for ocr in current_row])
        
        max_cols = max(len(row) for row in rows) if rows else 0
        
        return {
            "rows": rows,
            "row_count": len(rows),
            "col_count": max_cols
        }
    
    @staticmethod
    def save_markdown(result: Dict, output_path: str, image_path: str = None) -> str:
        """
        Save OCR result as Markdown file
        
        Args:
            result: OCR analysis result dict
            output_path: Path to save markdown file
            image_path: Path to the analyzed image
            
        Returns:
            Path to saved file
        """
        markdown_content = OCRResultExporter.to_markdown(result, image_path)
        
        output_path = Path(output_path)
        output_path.write_text(markdown_content, encoding='utf-8')
        
        print(f"✓ Markdown saved: {output_path}")
        return str(output_path)
    
    @staticmethod
    def save_json(result: Dict, output_path: str, image_path: str = None) -> str:
        """
        Save OCR result as JSON file
        
        Args:
            result: OCR analysis result dict
            output_path: Path to save JSON file
            image_path: Path to the analyzed image
            
        Returns:
            Path to saved file
        """
        json_content = OCRResultExporter.to_json(result, image_path)
        
        output_path = Path(output_path)
        output_path.write_text(json_content, encoding='utf-8')
        
        print(f"✓ JSON saved: {output_path}")
        return str(output_path)
    
    @staticmethod
    def save_both(result: Dict, output_dir: str, image_path: str = None, basename: str = "ocr_result") -> tuple:
        """
        Save OCR result in both Markdown and JSON formats
        
        Args:
            result: OCR analysis result dict
            output_dir: Directory to save files
            image_path: Path to the analyzed image
            basename: Base filename (without extension)
            
        Returns:
            Tuple of (markdown_path, json_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        md_path = output_dir / f"{basename}.md"
        json_path = output_dir / f"{basename}.json"
        
        md_saved = OCRResultExporter.save_markdown(result, str(md_path), image_path)
        json_saved = OCRResultExporter.save_json(result, str(json_path), image_path)
        
        return md_saved, json_saved


if __name__ == '__main__':
    # Example usage with table
    example_result = {
        "layouts": [
            {"type": "title", "score": 0.95, "bbox": [10, 10, 500, 50]},
            {"type": "table", "score": 0.92, "bbox": [10, 60, 500, 300]},
            {"type": "text", "score": 0.87, "bbox": [10, 320, 500, 400]}
        ],
        "ocr_results": [
            # Title OCR - inside title bbox
            {"text": "Document Title", "score": 0.92, "bbox": [15, 15, 200, 45]},
            
            # Table OCR - inside table bbox, simulating rows
            {"text": "Name", "score": 0.95, "bbox": [20, 70, 100, 90]},
            {"text": "Age", "score": 0.93, "bbox": [150, 70, 200, 90]},
            {"text": "City", "score": 0.94, "bbox": [280, 70, 350, 90]},
            {"text": "John", "score": 0.92, "bbox": [20, 120, 100, 140]},
            {"text": "25", "score": 0.91, "bbox": [150, 120, 200, 140]},
            {"text": "Ha Noi", "score": 0.90, "bbox": [280, 120, 380, 140]},
            {"text": "Mary", "score": 0.93, "bbox": [20, 170, 100, 190]},
            {"text": "30", "score": 0.92, "bbox": [150, 170, 200, 190]},
            {"text": "HCM", "score": 0.91, "bbox": [280, 170, 350, 190]},
            
            # Text OCR - inside text bbox
            {"text": "This is the footer text of the document.", "score": 0.85, "bbox": [15, 330, 450, 380]}
        ]
    }
    
    exporter = OCRResultExporter()
    
    # Export to markdown
    md = exporter.to_markdown(example_result, "example.jpg")
    print("=== MARKDOWN ===")
    print(md)
    print()
    
    # Export to JSON
    js = exporter.to_json(example_result, "example.jpg")
    print("=== JSON ===")
    print(js)
