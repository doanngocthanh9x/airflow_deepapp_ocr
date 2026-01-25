import cv2
import numpy as np
from paddleocr import PaddleOCR

# ============= 1. OCR =============
ocr = PaddleOCR(
    lang='vi',
    use_angle_cls=True,
    det=True,
    rec=True
)

# ============= 2. Warp một bbox về hình chữ nhật =============
def crop_line_from_quad(img, box):
    box = np.array(box, dtype="float32")

    width_top = np.linalg.norm(box[1] - box[0])
    width_bottom = np.linalg.norm(box[2] - box[3])
    maxWidth = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(box[3] - box[0])
    height_right = np.linalg.norm(box[2] - box[1])
    maxHeight = int(max(height_left, height_right))

    if maxWidth <= 0 or maxHeight <= 0:
        return None, 0, 0

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped, maxWidth, maxHeight

# ============= 3. THUẬT TOÁN: TOP-TO-BOTTOM, LEFT-TO-RIGHT =============
def simple_reading_order(boxes, line_height_ratio=0.5):
    """
    Thuật toán đơn giản: Sắp xếp theo thứ tự đọc tự nhiên
    1. Gom các boxes có cùng Y (cùng dòng) 
    2. Sort từng dòng từ trái sang phải
    3. Sort các dòng từ trên xuống dưới
    
    Tham số:
    - line_height_ratio: Tỷ lệ để xác định 2 boxes có cùng line hay không
    """
    if not boxes:
        return []
    
    # Thu thập thông tin boxes
    box_info = []
    for i, box in enumerate(boxes):
        min_y = float(np.min(box[:, 1]))
        max_y = float(np.max(box[:, 1]))
        min_x = float(np.min(box[:, 0]))
        max_x = float(np.max(box[:, 0]))
        
        cy = (min_y + max_y) / 2
        cx = (min_x + max_x) / 2
        h = max_y - min_y
        w = max_x - min_x
        
        box_info.append({
            'idx': i,
            'min_y': min_y,
            'max_y': max_y,
            'min_x': min_x,
            'max_x': max_x,
            'cy': cy,
            'cx': cx,
            'h': h,
            'w': w
        })
    
    # Tính median height
    heights = [info['h'] for info in box_info]
    median_h = np.median(heights)
    threshold = median_h * line_height_ratio
    
    # Sort theo Y trước
    box_info.sort(key=lambda x: x['cy'])
    
    # Gom thành lines
    lines = []
    current_line = [box_info[0]]
    
    for i in range(1, len(box_info)):
        curr = box_info[i]
        prev_line_cy = np.mean([b['cy'] for b in current_line])
        
        # Nếu khoảng cách Y nhỏ -> cùng line
        if abs(curr['cy'] - prev_line_cy) <= threshold:
            current_line.append(curr)
        else:
            # Sort line hiện tại theo X, lưu lại và bắt đầu line mới
            current_line.sort(key=lambda x: x['cx'])
            lines.append([b['idx'] for b in current_line])
            current_line = [curr]
    
    # Xử lý line cuối
    if current_line:
        current_line.sort(key=lambda x: x['cx'])
        lines.append([b['idx'] for b in current_line])
    
    return lines

# ============= 4. OCR theo từng line đã gom =============
def ocr_by_grouped_lines(img, boxes, ocr_texts, lines):
    """
    Nhận diện text theo từng line đã được gom
    
    Args:
        img: Ảnh gốc
        boxes: List các bounding boxes từ PaddleOCR
        ocr_texts: List text tương ứng với mỗi box
        lines: List các lines (mỗi line là list các box indices)
    
    Returns:
        List các dòng text đã được gom
    """
    line_texts = []
    
    for line_idx, line in enumerate(lines):
        words = []
        
        for box_idx in line:
            if box_idx < len(ocr_texts):
                text = ocr_texts[box_idx]
                words.append(text)
        
        # Ghép các words thành một dòng
        line_text = ' '.join(words)
        line_texts.append(line_text)
    
    return line_texts

# ============= 5. Dựng canvas theo reading order =============
def build_layout_from_reading_order(img, boxes, lines,
                                     line_spacing=10,
                                     word_spacing=5,
                                     margin=20):
    """
    Dựng canvas dựa trên reading order đã được sắp xếp
    
    lines: [[idx1, idx2], [idx3, idx4], ...]
    """
    # Warp tất cả boxes
    warped_cache = {}
    for i, box in enumerate(boxes):
        warped, w, h = crop_line_from_quad(img, box)
        if warped is not None:
            warped_cache[i] = (warped, w, h)
    
    # Tính kích thước từng line
    line_heights = []
    line_widths = []
    
    for line in lines:
        max_h = 0
        total_w = 0
        valid_count = 0
        
        for idx in line:
            if idx in warped_cache:
                _, w, h = warped_cache[idx]
                max_h = max(max_h, h)
                total_w += w
                valid_count += 1
        
        if valid_count > 1:
            total_w += (valid_count - 1) * word_spacing
        
        line_heights.append(max_h)
        line_widths.append(total_w)
    
    # Tính canvas size
    if not line_heights or not line_widths:
        return None
    
    canvas_w = max(line_widths) + 2 * margin
    canvas_h = sum(line_heights) + (len(lines) - 1) * line_spacing + 2 * margin
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # Dán boxes lên canvas
    cur_y = margin
    
    for line_idx, line in enumerate(lines):
        line_h = line_heights[line_idx]
        cur_x = margin
        
        for idx in line:
            if idx not in warped_cache:
                continue
            
            warped, w, h = warped_cache[idx]
            
            # Căn giữa theo chiều cao
            y_offset = (line_h - h) // 2
            y_pos = cur_y + y_offset
            
            canvas[y_pos:y_pos + h, cur_x:cur_x + w] = warped
            cur_x += w + word_spacing
        
        cur_y += line_h + line_spacing
    
    return canvas

# ============= 6. Visualize reading order =============
def visualize_reading_order(img, boxes, lines):
    """
    Vẽ số thứ tự đọc lên ảnh
    """
    vis_img = img.copy()
    
    order = 1
    for line_idx, line in enumerate(lines):
        for idx in line:
            box = boxes[idx]
            
            # Vẽ box
            pts = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
            
            # Ghi số thứ tự
            cx = int(np.mean(box[:, 0]))
            cy = int(np.mean(box[:, 1]))
            
            cv2.putText(vis_img, str(order), (cx - 10, cy + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            order += 1
    
    return vis_img

# ============= 7. Main =============
def main():
    image_path = "/home/dnthanh/airflow_deepapp_ocr/Screenshot 2025-05-06 113653.png"
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh:", image_path)
        return

    print("Đang OCR ảnh...")
    ocr_result = ocr.ocr(image_path, cls=True)
    if not ocr_result or len(ocr_result[0]) == 0:
        print("Không có bbox nào.")
        return

    # Thu thập boxes và texts
    boxes = []
    texts = []
    confidences = []
    
    for line in ocr_result[0]:
        box = np.array(line[0], dtype=float)
        text = line[1][0]  # Text
        conf = line[1][1]  # Confidence
        
        boxes.append(box)
        texts.append(text)
        confidences.append(conf)

    print(f"Tổng số boxes: {len(boxes)}")
    print("\n" + "="*60)
    
    # Gom boxes thành lines theo thứ tự đọc
    print("Đang gom boxes thành lines...")
    lines = simple_reading_order(boxes, line_height_ratio=0.5)
    
    print(f"Số lines được gom: {len(lines)}")
    print("\n" + "="*60)
    
    # OCR theo từng line
    print("\nText theo từng line:")
    line_texts = ocr_by_grouped_lines(img, boxes, texts, lines)
    
    for i, (line_indices, line_text) in enumerate(zip(lines, line_texts), 1):
        print(f"\nLine {i} ({len(line_indices)} boxes):")
        print(f"  Text: {line_text}")
        
        # In chi tiết từng box trong line
        print(f"  Boxes:")
        for j, box_idx in enumerate(line_indices):
            print(f"    [{j+1}] {texts[box_idx]} (conf: {confidences[box_idx]:.2f})")
    
    print("\n" + "="*60)
    
    # Lưu text vào file
    output_txt_path = "/home/dnthanh/airflow_deepapp_ocr/ocr_result.txt"
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for i, line_text in enumerate(line_texts, 1):
            f.write(f"Line {i}: {line_text}\n")
    print(f"\nĐã lưu text vào: {output_txt_path}")
    
    # Build canvas
    print("\nĐang dựng layout canvas...")
    canvas = build_layout_from_reading_order(
        img, boxes, lines,
        line_spacing=10,
        word_spacing=5,
        margin=20
    )
    
    # Visualize
    print("Đang tạo visualization...")
    vis = visualize_reading_order(img, boxes, lines)
    
    # Hiển thị kết quả
    if canvas is not None:
        cv2.imshow("Layout Result", canvas)
        cv2.imshow("Reading Order", vis)
        cv2.imshow("Original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Lưu kết quả
        cv2.imwrite("/home/claude/layout_result.png", canvas)
        cv2.imwrite("/home/claude/reading_order.png", vis)
        print("\nĐã lưu ảnh vào:")
        print("  - /home/claude/layout_result.png")
        print("  - /home/claude/reading_order.png")
    else:
        print("Không thể tạo canvas.")

if __name__ == "__main__":
    main()