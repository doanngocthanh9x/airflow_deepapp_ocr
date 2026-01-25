import cv2
import numpy as np
import math
from paddleocr import PaddleOCR

# ======================
# 1. Cấu hình PaddleOCR
# ======================
# nếu dùng tiếng Việt đổi lang='vi'
ocr_det = PaddleOCR(lang='en', use_angle_cls=True, det=True, rec=False)
ocr_full = PaddleOCR(lang='en', use_angle_cls=True, det=True, rec=True)

# ======================
# 2. Tính góc từ bbox
# ======================
def compute_angle_from_box(box):
    """
    box: np.array shape (4, 2) hoặc list 4 điểm
    Mặc định dùng cạnh đáy p3 -> p2.
    """
    (x3, y3) = box[3]
    (x2, y2) = box[2]

    dx = x2 - x3
    dy = y2 - y3

    if dx == 0:
        angle_rad = math.pi / 2.0
    else:
        angle_rad = math.atan(dy / dx)

    angle_deg = angle_rad * 180.0 / math.pi
    return angle_deg

# ======================
# 3. Xoay ảnh theo góc
# ======================
def rotate_image_by_angle(img, angle_deg):
    """
    Xoay toàn bộ ảnh quanh tâm.
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

# ======================
# 4. Chọn line dài nhất làm thước
# ======================
def get_longest_line_box(result):
    """
    result: output từ ocr.ocr(img_path, cls=True)
    Trả về bbox (4x2) của line có cạnh đáy dài nhất.
    """
    if not result or len(result[0]) == 0:
        return None

    max_len = -1
    best_box = None

    for line in result[0]:
        box = np.array(line[0], dtype=float)  # 4 điểm
        # cạnh đáy: p3 -> p2
        p3 = box[3]
        p2 = box[2]
        length = np.linalg.norm(p2 - p3)
        if length > max_len:
            max_len = length
            best_box = box

    return best_box

# ======================
# 5. Pipeline: detect line -> xoay -> OCR
# ======================
def process_image(img_path):
    # đọc ảnh gốc
    img = cv2.imread(img_path)
    if img is None:
        print("Không đọc được ảnh:", img_path)
        return

    # bước 1: detect line để lấy bbox
    det_result = ocr_det.ocr(img_path, cls=True)

    box = get_longest_line_box(det_result)
    if box is None:
        print("Không tìm được line nào.")
        return

    # bước 2: tính góc từ bbox
    angle = compute_angle_from_box(box)

    # tùy hướng bạn muốn xoay, có thể dùng:
    # angle = -angle
    # hoặc xử lý thêm:
    # if angle < -45: angle += 180

    print("Góc tính được:", angle)

    # bước 3: xoay ảnh
    img_rotated = rotate_image_by_angle(img, angle)

    # bước 4 (tuỳ chọn): chạy OCR trên ảnh đã xoay
    rot_result = ocr_full.ocr(img_rotated, cls=True)

    # in kết quả text
    print("Kết quả OCR trên ảnh đã xoay:")
    for line in rot_result[0]:
        txt = line[1][0]
        conf = line[1][1]
        print(f"{txt} (conf={conf:.3f})")

    # hiển thị
    cv2.imshow("original", img)
    cv2.imshow("rotated", img_rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # thay bằng path ảnh của bạn
    process_image("/home/dnthanh/airflow_deepapp_ocr/01HM00017214_4.png")
