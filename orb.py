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

# ============= 3. Thuật toán Simple Reading Order =============
def simple_reading_order(boxes, line_height_ratio=0.5):
    """
    Sắp xếp boxes theo thứ tự đọc tự nhiên: Top->Bottom, Left->Right
    """
    if not boxes:
        return []
    
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
    
    heights = [info['h'] for info in box_info]
    median_h = np.median(heights)
    threshold = median_h * line_height_ratio
    
    box_info.sort(key=lambda x: x['cy'])
    
    lines = []
    current_line = [box_info[0]]
    
    for i in range(1, len(box_info)):
        curr = box_info[i]
        prev_line_cy = np.mean([b['cy'] for b in current_line])
        
        if abs(curr['cy'] - prev_line_cy) <= threshold:
            current_line.append(curr)
        else:
            current_line.sort(key=lambda x: x['cx'])
            lines.append([b['idx'] for b in current_line])
            current_line = [curr]
    
    if current_line:
        current_line.sort(key=lambda x: x['cx'])
        lines.append([b['idx'] for b in current_line])
    
    return lines

# ============= 4. ORB: Tìm vị trí tương đối giữa các cropped boxes =============
def find_relative_positions_with_orb(img, boxes, warped_boxes, 
                                     min_matches=10,
                                     distance_threshold=50):
    """
    Dùng ORB để tìm vị trí tương đối của các cropped boxes trong ảnh gốc
    
    Returns:
        dict: {box_idx: (x, y, w, h)} - Vị trí và kích thước của mỗi box
    """
    print("\n=== Đang dùng ORB để tìm vị trí các boxes ===")
    
    # Convert ảnh gốc sang grayscale
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # ORB detector cho ảnh gốc
    orb = cv2.ORB_create(nfeatures=2000)
    kp_img, des_img = orb.detectAndCompute(gray_img, None)
    
    if des_img is None:
        print("Không tìm thấy features trong ảnh gốc!")
        return None
    
    positions = {}
    
    for idx, (box, warped) in enumerate(zip(boxes, warped_boxes)):
        if warped is None:
            continue
        
        # Convert warped box sang grayscale
        if len(warped.shape) == 3:
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray_warped = warped
        
        # Detect features trong warped box
        kp_box, des_box = orb.detectAndCompute(gray_warped, None)
        
        if des_box is None or len(kp_box) < 5:
            # Fallback: dùng vị trí từ bbox gốc
            min_x = float(np.min(box[:, 0]))
            min_y = float(np.min(box[:, 1]))
            max_x = float(np.max(box[:, 0]))
            max_y = float(np.max(box[:, 1]))
            positions[idx] = {
                'x': min_x,
                'y': min_y,
                'w': max_x - min_x,
                'h': max_y - min_y,
                'method': 'bbox'
            }
            continue
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_box, des_img, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) >= min_matches:
            # Tìm homography
            src_pts = np.float32([kp_box[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # Transform corners của warped box về ảnh gốc
                    h, w = gray_warped.shape
                    corners = np.float32([
                        [0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]
                    ]).reshape(-1, 1, 2)
                    
                    transformed_corners = cv2.perspectiveTransform(corners, M)
                    
                    # Tính bounding box
                    x_coords = transformed_corners[:, 0, 0]
                    y_coords = transformed_corners[:, 0, 1]
                    
                    min_x = float(np.min(x_coords))
                    max_x = float(np.max(x_coords))
                    min_y = float(np.min(y_coords))
                    max_y = float(np.max(y_coords))
                    
                    positions[idx] = {
                        'x': min_x,
                        'y': min_y,
                        'w': max_x - min_x,
                        'h': max_y - min_y,
                        'method': 'orb',
                        'matches': len(good_matches)
                    }
                    
                    print(f"  Box {idx}: ORB matched ({len(good_matches)} good matches)")
                    continue
            except:
                pass
        
        # Fallback: dùng bbox gốc
        min_x = float(np.min(box[:, 0]))
        min_y = float(np.min(box[:, 1]))
        max_x = float(np.max(box[:, 0]))
        max_y = float(np.max(box[:, 1]))
        positions[idx] = {
            'x': min_x,
            'y': min_y,
            'w': max_x - min_x,
            'h': max_y - min_y,
            'method': 'bbox'
        }
        print(f"  Box {idx}: Fallback to bbox (insufficient matches)")
    
    return positions

# ============= 5. Dựng lại layout giống ảnh gốc =============
def reconstruct_layout_with_original_positions(img, boxes, lines, 
                                                positions=None,
                                                margin=20,
                                                scale_factor=1.0):
    """
    Dựng lại layout với vị trí tương đối giống ảnh gốc
    
    Args:
        positions: dict vị trí từ ORB hoặc None (dùng bbox gốc)
        scale_factor: Tỷ lệ scale (< 1 để thu nhỏ, > 1 để phóng to)
    """
    # Warp tất cả boxes
    warped_boxes = []
    for box in boxes:
        warped, w, h = crop_line_from_quad(img, box)
        warped_boxes.append(warped)
    
    # Nếu không có positions từ ORB, dùng bbox gốc
    if positions is None:
        positions = {}
        for idx, box in enumerate(boxes):
            min_x = float(np.min(box[:, 0]))
            min_y = float(np.min(box[:, 1]))
            max_x = float(np.max(box[:, 0]))
            max_y = float(np.max(box[:, 1]))
            positions[idx] = {
                'x': min_x,
                'y': min_y,
                'w': max_x - min_x,
                'h': max_y - min_y
            }
    
    # Tính canvas size dựa trên positions
    all_positions = list(positions.values())
    min_x = min(p['x'] for p in all_positions)
    min_y = min(p['y'] for p in all_positions)
    max_x = max(p['x'] + p['w'] for p in all_positions)
    max_y = max(p['y'] + p['h'] for p in all_positions)
    
    # Apply scale
    canvas_w = int((max_x - min_x) * scale_factor) + 2 * margin
    canvas_h = int((max_y - min_y) * scale_factor) + 2 * margin
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # Dán các boxes lên canvas theo vị trí tương đối
    for idx in positions:
        if warped_boxes[idx] is None:
            continue
        
        pos = positions[idx]
        
        # Tính vị trí trên canvas
        x = int((pos['x'] - min_x) * scale_factor) + margin
        y = int((pos['y'] - min_y) * scale_factor) + margin
        
        warped = warped_boxes[idx]
        h, w = warped.shape[:2]
        
        # Scale warped nếu cần
        if scale_factor != 1.0:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            warped = cv2.resize(warped, (new_w, new_h))
            h, w = new_h, new_w
        
        # Kiểm tra boundary
        if y + h > canvas_h or x + w > canvas_w:
            continue
        
        canvas[y:y+h, x:x+w] = warped
    
    return canvas

# ============= 6. Visualize ORB matches =============
def visualize_orb_matches(img, warped_boxes, boxes, positions, num_show=5):
    """
    Vẽ visualization của ORB matches
    """
    vis_img = img.copy()
    
    # Chọn một số boxes để visualize
    orb_boxes = [idx for idx, pos in positions.items() 
                 if pos.get('method') == 'orb']
    
    if not orb_boxes:
        print("Không có box nào dùng ORB!")
        return vis_img
    
    # Lấy random một số boxes
    import random
    show_boxes = random.sample(orb_boxes, min(num_show, len(orb_boxes)))
    
    print(f"\nVisualizing ORB matches cho {len(show_boxes)} boxes...")
    
    for idx in show_boxes:
        warped = warped_boxes[idx]
        if warped is None:
            continue
        
        pos = positions[idx]
        
        # Vẽ bounding box trên ảnh gốc
        x, y, w, h = int(pos['x']), int(pos['y']), int(pos['w']), int(pos['h'])
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Ghi số matches
        matches_count = pos.get('matches', 0)
        cv2.putText(vis_img, f"Box {idx}: {matches_count} matches", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
    
    return vis_img

# ============= 7. So sánh 2 layouts =============
def compare_layouts(original_img, reconstructed_img):
    """
    Đặt 2 ảnh cạnh nhau để so sánh
    """
    h1, w1 = original_img.shape[:2]
    h2, w2 = reconstructed_img.shape[:2]
    
    # Resize về cùng chiều cao
    target_h = max(h1, h2)
    
    if h1 != target_h:
        scale = target_h / h1
        original_img = cv2.resize(original_img, (int(w1 * scale), target_h))
    
    if h2 != target_h:
        scale = target_h / h2
        reconstructed_img = cv2.resize(reconstructed_img, (int(w2 * scale), target_h))
    
    # Ghép 2 ảnh
    comparison = np.hstack([original_img, reconstructed_img])
    
    # Vẽ đường phân cách
    h, w = comparison.shape[:2]
    mid = original_img.shape[1]
    cv2.line(comparison, (mid, 0), (mid, h), (0, 0, 255), 3)
    
    # Label
    cv2.putText(comparison, "ORIGINAL", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(comparison, "RECONSTRUCTED", (mid + 20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    return comparison

# ============= 8. Main =============
def main():
    image_path = "/home/dnthanh/airflow_deepapp_ocr/Screenshot 2025-05-06 113653.png"
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh:", image_path)
        return

    print("=== Bước 1: OCR ảnh ===")
    ocr_result = ocr.ocr(image_path, cls=True)
    if not ocr_result or len(ocr_result[0]) == 0:
        print("Không có bbox nào.")
        return

    # Thu thập boxes và texts
    boxes = []
    texts = []
    
    for line in ocr_result[0]:
        box = np.array(line[0], dtype=float)
        text = line[1][0]
        boxes.append(box)
        texts.append(text)

    print(f"Tổng số boxes: {len(boxes)}")
    
    print("\n=== Bước 2: Crop các boxes ===")
    warped_boxes = []
    for i, box in enumerate(boxes):
        warped, w, h = crop_line_from_quad(img, box)
        warped_boxes.append(warped)
        if warped is not None:
            print(f"  Box {i}: {w}x{h}px - '{texts[i]}'")
    
    print("\n=== Bước 3: Sắp xếp theo reading order ===")
    lines = simple_reading_order(boxes, line_height_ratio=0.5)
    print(f"Số lines: {len(lines)}")
    for i, line in enumerate(lines):
        line_texts = [texts[idx] for idx in line]
        print(f"  Line {i+1}: {' '.join(line_texts)}")
    
    print("\n=== Bước 4: Dùng ORB tìm vị trí tương đối ===")
    positions = find_relative_positions_with_orb(
        img, boxes, warped_boxes,
        min_matches=10,
        distance_threshold=50
    )
    
    # Thống kê
    orb_count = sum(1 for p in positions.values() if p.get('method') == 'orb')
    bbox_count = len(positions) - orb_count
    print(f"\nThống kê:")
    print(f"  - ORB matched: {orb_count}/{len(positions)} boxes")
    print(f"  - Fallback to bbox: {bbox_count}/{len(positions)} boxes")
    
    print("\n=== Bước 5: Dựng lại layout ===")
    
    # Thử nhiều scale factors
    scale_factors = [1.0, 0.8, 0.5]
    
    for scale in scale_factors:
        print(f"\nDựng layout với scale = {scale}")
        
        reconstructed = reconstruct_layout_with_original_positions(
            img, boxes, lines,
            positions=positions,
            margin=20,
            scale_factor=scale
        )
        
        if reconstructed is not None:
            # Lưu kết quả
            output_path = f"/home/claude/reconstructed_layout_scale_{scale}.png"
            cv2.imwrite(output_path, reconstructed)
            print(f"  Đã lưu: {output_path}")
    
    # Visualize ORB matches
    print("\n=== Bước 6: Visualize ORB matches ===")
    orb_vis = visualize_orb_matches(img, warped_boxes, boxes, positions, num_show=5)
    cv2.imwrite("/home/claude/orb_matches_visualization.png", orb_vis)
    
    # So sánh original vs reconstructed
    print("\n=== Bước 7: So sánh layouts ===")
    reconstructed_1x = reconstruct_layout_with_original_positions(
        img, boxes, lines, positions=positions, margin=20, scale_factor=1.0
    )
    
    if reconstructed_1x is not None:
        comparison = compare_layouts(img, reconstructed_1x)
        cv2.imwrite("/home/claude/comparison_original_vs_reconstructed.png", comparison)
        print("Đã lưu: comparison_original_vs_reconstructed.png")
    
    print("\n=== Hoàn thành! ===")
    print("\nCác file đã tạo:")
    print("  1. reconstructed_layout_scale_*.png - Layout với các scale khác nhau")
    print("  2. orb_matches_visualization.png - Visualization ORB matches")
    print("  3. comparison_original_vs_reconstructed.png - So sánh gốc vs dựng lại")
    
    # Hiển thị
    if reconstructed_1x is not None:
        cv2.imshow("Original", img)
        cv2.imshow("Reconstructed (scale=1.0)", reconstructed_1x)
        cv2.imshow("ORB Matches", orb_vis)
        cv2.imshow("Comparison", comparison)
        print("\nNhấn phím bất kỳ để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()