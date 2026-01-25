import cv2
import numpy as np
from paddleocr import PaddleOCR

# =========================
# 1. Khởi tạo PaddleOCR
# =========================
# Nếu bạn dùng tiếng Việt thì để lang='vi'
ocr = PaddleOCR(
    lang='vi',
    use_angle_cls=True,
    det=True,
    rec=True
)

# =========================
# 2. Hàm crop + duỗi thẳng 1 line từ bbox 4 điểm
# =========================
def crop_line_from_quad(img, box):
    """
    box: list/np.array 4 điểm [p0,p1,p2,p3]
         theo chuẩn PaddleOCR: trên-trái, trên-phải, dưới-phải, dưới-trái.
    Trả về: ảnh line đã được warp về hình chữ nhật (thẳng).
    """
    box = np.array(box, dtype="float32")

    # chiều rộng = max(độ dài 2 cạnh trên/dưới)
    width_top = np.linalg.norm(box[1] - box[0])
    width_bottom = np.linalg.norm(box[2] - box[3])
    maxWidth = int(max(width_top, width_bottom))

    # chiều cao = max(độ dài 2 cạnh trái/phải)
    height_left = np.linalg.norm(box[3] - box[0])
    height_right = np.linalg.norm(box[2] - box[1])
    maxHeight = int(max(height_left, height_right))

    if maxWidth <= 0 or maxHeight <= 0:
        return None

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # duỗi thẳng (perspective transform)
    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped

# =========================
# 3. Dán line vào canvas trắng
# =========================
def paste_line(canvas, line_img, x, y):
    """
    Dán line_img lên canvas tại (x, y) (góc trên-trái).
    Không kiểm tra tràn biên, giả sử canvas đủ lớn.
    """
    h, w = line_img.shape[:2]
    canvas[y:y+h, x:x+w] = line_img

# =========================
# 4. Xây dựng ảnh mới: nền trắng + các line đã straighten
# =========================
def build_straight_text_image(img, ocr_result,
                              line_spacing=10,
                              margin_x=20,
                              margin_y=20):
    """
    img: ảnh gốc (BGR)
    ocr_result: kết quả từ ocr.ocr(image_path, cls=True)
    Trả về: ảnh trắng, trong đó mỗi line text đã được duỗi thẳng
            và xếp lần lượt từ trên xuống.
    """

    if not ocr_result or len(ocr_result[0]) == 0:
        return None

    # Bước 1: lấy tất cả bbox line
    boxes = []
    for line in ocr_result[0]:
        box = np.array(line[0], dtype=float)  # 4 điểm
        boxes.append(box)

    # Bước 2: sort theo toạ độ y trung bình để xếp từ trên xuống
    boxes_sorted = sorted(
        boxes,
        key=lambda b: float(np.mean(b[:, 1]))
    )

    # Bước 3: crop & straighten từng line, tính trước kích thước canvas
    line_images = []
    max_width = 0
    total_height = margin_y  # bắt đầu từ margin trên

    for box in boxes_sorted:
        line_img = crop_line_from_quad(img, box)
        if line_img is None:
            continue
        h, w = line_img.shape[:2]
        line_images.append(line_img)
        max_width = max(max_width, w)
        total_height += h + line_spacing

    if not line_images:
        return None

    total_height += margin_y  # margin dưới
    canvas_width = max_width + 2 * margin_x

    # Bước 4: tạo canvas trắng
    canvas = np.ones(
        (total_height, canvas_width, 3),
        dtype=np.uint8
    ) * 255

    # Bước 5: lần lượt dán line vào canvas
    cur_y = margin_y
    for line_img in line_images:
        h, w = line_img.shape[:2]
        paste_line(canvas, line_img, margin_x, cur_y)
        cur_y += h + line_spacing

    return canvas

# =========================
# 5. Chương trình chính
# =========================
def main():
    image_path = "/home/dnthanh/airflow_deepapp_ocr/01HM00014343_300005_image_96.png"  # đổi thành ảnh của bạn
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh:", image_path)
        return

    # detect + rec (rec không bắt buộc, nhưng tiện nếu muốn xem text)
    ocr_result = ocr.ocr(image_path, cls=True)

    # Xây dựng ảnh text đã được straighten
    new_img = build_straight_text_image(img, ocr_result,
                                        line_spacing=10,
                                        margin_x=20,
                                        margin_y=20)

    if new_img is None:
        print("Không tạo được ảnh text mới (không có line hợp lệ).")
        return

    # Hiển thị
    cv2.imshow("original", img)
    cv2.imshow("straight_lines", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #press key to close the image windows
    input("Nhấn Enter để đóng chương trình...")


if __name__ == "__main__":
    main()
    #press enter to close system
    

