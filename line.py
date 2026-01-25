import cv2
import numpy as np
from paddleocr import PaddleOCR

# ============= 1. Khởi tạo PaddleOCR =============
ocr = PaddleOCR(
    lang='en',
    use_angle_cls=True,
    det=True,
    rec=True
)

# ============= 2. Crop + duỗi thẳng 1 bbox =============
def crop_line_from_quad(img, box):
    """
    box: np.array/list 4 điểm [p0,p1,p2,p3] (trên-trái, trên-phải, dưới-phải, dưới-trái)
    Trả về:
      - warped: ảnh line đã duỗi thẳng (H x W x 3)
      - maxWidth, maxHeight: kích thước warped
    """
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

# ============= 3. Xây canvas theo layout bbox =============
def build_layout_text_image(img, ocr_result, margin=10):
    """
    Giữ nguyên layout của ảnh gốc:
    - Canvas trắng đủ lớn (theo min/max của tất cả bbox).
    - Mỗi bbox được duỗi thẳng rồi đặt lại gần vị trí cũ (có margin).
    """

    if not ocr_result or len(ocr_result[0]) == 0:
        return None

    h_img, w_img = img.shape[:2]

    boxes = []
    for line in ocr_result[0]:
        box = np.array(line[0], dtype=float)
        boxes.append(box)

    # Tính min/max theo tọa độ bbox để biết vùng cần phủ
    all_pts = np.concatenate(boxes, axis=0)
    min_x = max(int(np.floor(np.min(all_pts[:, 0])) - margin), 0)
    min_y = max(int(np.floor(np.min(all_pts[:, 1])) - margin), 0)
    max_x = min(int(np.ceil(np.max(all_pts[:, 0])) + margin), w_img)
    max_y = min(int(np.ceil(np.max(all_pts[:, 1])) + margin), h_img)

    # Kích thước canvas gần tương đương vùng text
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    if canvas_w <= 0 or canvas_h <= 0:
        return None

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Với mỗi bbox:
    for box in boxes:
        # crop + straighten
        warped, w_line, h_line = crop_line_from_quad(img, box)
        if warped is None:
            continue

        # Tính vị trí đặt trên canvas:
        # lấy tâm bbox trong hệ gốc, rồi tịnh tiến về hệ canvas
        cx = np.mean(box[:, 0]) - min_x
        cy = np.mean(box[:, 1]) - min_y

        # góc trên-trái để dán (căn giữa bbox)
        x0 = int(cx - w_line / 2)
        y0 = int(cy - h_line / 2)

        # cắt cho không tràn biên
        x1 = max(x0, 0)
        y1 = max(y0, 0)
        x2 = min(x0 + w_line, canvas_w)
        y2 = min(y0 + h_line, canvas_h)

        # tương ứng vùng trong warped
        wx1 = x1 - x0
        wy1 = y1 - y0
        wx2 = wx1 + (x2 - x1)
        wy2 = wy1 + (y2 - y1)

        if x1 < x2 and y1 < y2:
            canvas[y1:y2, x1:x2] = warped[wy1:wy2, wx1:wx2]

    return canvas

# ============= 4. main =============
def main():
    image_path = "/home/dnthanh/airflow_deepapp_ocr/01HM00013708_300005_image_126.png"  # đổi thành ảnh của bạn
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh:", image_path)
        return

    ocr_result = ocr.ocr(image_path, cls=True)

    layout_img = build_layout_text_image(img, ocr_result, margin=10)

    if layout_img is None:
        print("Không tạo được layout image.")
        return

    cv2.imshow("original", img)
    cv2.imshow("layout_straight", layout_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
