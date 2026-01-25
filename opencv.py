import cv2
import numpy as np
import math

im0 = None      # ảnh gốc
im1 = None      # ảnh đang hiển thị (vẽ line)
pts = []        # lưu 2 điểm [P1, P2]

def straighten_image(pts):
    global im0
    # pts[0], pts[1] là (x, y)
    x1, y1 = pts[0]
    x2, y2 = pts[1]

    # tính góc (độ) giống C++
    # chú ý tránh chia cho 0
    if (x1 - x2) == 0:
        angle_rad = math.pi / 2.0
    else:
        angle_rad = math.atan((y1 - y2) / (x1 - x2))

    angle = angle_rad * 180.0 / math.pi

    # tâm xoay là tâm ảnh
    h, w = im0.shape[:2]
    center = (w / 2.0, h / 2.0)

    # ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # warpAffine
    dst = cv2.warpAffine(im0, M, (w, h))

    cv2.imshow("dst", dst)

def on_mouse(event, x, y, flags, param):
    global im0, im1, pts

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) == 0:
            # điểm đầu
            pts.append((x, y))
        elif len(pts) == 1:
            # điểm cuối -> xoay ảnh
            pts.append((x, y))
            straighten_image(pts)
        elif len(pts) == 2:
            # reset để chọn lại
            im1 = im0.copy()
            pts = []

    elif event == cv2.EVENT_MOUSEMOVE and len(pts) == 1:
        # đang kéo chuột để thấy line tạm thời
        im1 = im0.copy()
        cv2.line(im1, pts[0], (x, y), (0, 0, 255), 2)

    if im1 is not None:
        cv2.imshow("src", im1)

def main():
    global im0, im1
    im0 = cv2.imread("/home/dnthanh/airflow_deepapp_ocr/Screenshot 2025-07-23 110229.png")  # đổi thành ảnh của bạn
    if im0 is None:
        print("Không load được ảnh")
        return

    im1 = im0.copy()

    cv2.namedWindow("src", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("src", on_mouse, None)
    cv2.imshow("src", im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
