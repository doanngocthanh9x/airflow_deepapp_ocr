import cv2
import numpy as np
from paddleocr import PaddleOCR

# 1. Khởi tạo PaddleOCR: chỉ detect text, không rec
ocr = PaddleOCR(lang='en', use_angle_cls=True, det=True, rec=False)  
# nếu có tiếng Việt: lang='vi'

def detect_lines(image_path):
    """
    Dùng PaddleOCR detect, trả về list:
    [ (box, line_crop_image) ]
    box: np.ndarray shape (4, 2)
    """
    img = cv2.imread(image_path)
    result = ocr.ocr(image_path, cls=True)
    lines = []

    for line in result[0]:
        box = np.array(line[0]).astype(int)  # 4 điểm
        x_min = np.min(box[:, 0])
        x_max = np.max(box[:, 0])
        y_min = np.min(box[:, 1])
        y_max = np.max(box[:, 1])

        line_crop = img[y_min:y_max, x_min:x_max].copy()
        lines.append((box, line_crop))

    return lines

def preprocess_line(line_img):
    """
    Tiền xử lý ảnh line.
    """
    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphology để xóa nhiễu nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def split_characters(line_img):
    """
    Tách ký tự bằng contour.
    Trả về list ảnh ký tự (đã crop) theo thứ tự trái -> phải.
    """
    bin_img = preprocess_line(line_img)

    # Tìm contour
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    char_boxes = []
    h_line = bin_img.shape[0]

    # Lọc contour theo kích thước
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Các ngưỡng này bạn phải chỉnh theo dữ liệu thực tế
        if h < h_line * 0.4:         # quá thấp -> noise
            continue
        if w < 3:                    # quá hẹp -> noise
            continue

        char_boxes.append((x, y, w, h))

    # Sắp xếp trái -> phải
    char_boxes = sorted(char_boxes, key=lambda b: b[0])

    char_images = []
    for (x, y, w, h) in char_boxes:
        char_crop = line_img[y:y+h, x:x+w].copy()
        char_images.append(char_crop)

    return char_images, char_boxes

# (Tùy chọn) 3. Model nhận dạng ký tự
# Ở đây demo đơn giản: dùng luôn rec của PaddleOCR trên từng char_crop
char_recognizer = PaddleOCR(lang='en', det=False, rec=True, use_angle_cls=True)

def recognize_characters(char_images):
    """
    Nhận dạng list ảnh ký tự.
    Ở đây dùng PaddleOCR rec cho từng ảnh.
    """
    texts = []
    for img in char_images:
        # PaddleOCR cần file path hoặc np.ndarray (tùy version),
        # ở đây xử lý qua cv2.imencode -> bytes
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            texts.append("")
            continue
        img_bytes = buf.tobytes()

        rec_result = char_recognizer.ocr(img_bytes, det=False, rec=True)
        if rec_result and len(rec_result[0]) > 0:
            txt = rec_result[0][0][0]
        else:
            txt = ""
        texts.append(txt)

    return texts

def join_characters_to_line(texts, char_boxes, space_ratio=0.5):
    """
    Ghép chuỗi ký tự lại thành 1 line hoàn chỉnh.
    Có chèn khoảng trắng dựa trên khoảng cách giữa 2 box.
    """
    if not texts:
        return ""

    final_text = texts[0]
    for i in range(1, len(texts)):
        prev_x, _, prev_w, _ = char_boxes[i-1]
        cur_x, _, _, _ = char_boxes[i]

        gap = cur_x - (prev_x + prev_w)
        avg_width = np.mean([b[2] for b in char_boxes])

        # nếu khoảng cách lớn hơn space_ratio * avg_width thì xem là khoảng trắng
        if gap > space_ratio * avg_width:
            final_text += " "
        final_text += texts[i]

    return final_text

def process_image(image_path):
    """
    Pipeline tổng:
    1. Detect line bằng PaddleOCR
    2. Crop từng line
    3. Tách ký tự
    4. Nhận dạng từng ký tự
    5. Ghép thành text cho từng line
    """
    lines = detect_lines(image_path)
    all_lines_text = []

    for idx, (box, line_img) in enumerate(lines):
        char_images, char_boxes = split_characters(line_img)
        if not char_images:
            all_lines_text.append("")  # không tách được ký tự
            continue

        char_texts = recognize_characters(char_images)
        line_text = join_characters_to_line(char_texts, char_boxes)
        all_lines_text.append(line_text)

    return all_lines_text

if __name__ == "__main__":
    img_path = "/home/dnthanh/airflow_deepapp_ocr/image.png"  # ảnh bất kỳ có nhiều line text
    lines_text = process_image(img_path)
    for i, t in enumerate(lines_text):
        print(f"Line {i+1}: {t}")
