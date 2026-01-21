# 🖼️ Remove Background DAG

> **DAG ID:** `com.image.remove_background`  
> **Phiên bản:** 1.0.0

---

## 📖 Giới thiệu

DAG này sử dụng AI để **tự động xóa background** khỏi hình ảnh. Hỗ trợ nhiều loại model khác nhau tùy vào yêu cầu về chất lượng và tốc độ.

**Thư viện sử dụng:** [rembg](https://github.com/danielgatis/rembg) - dựa trên U²-Net, IS-Net

---

## 🚀 Cách sử dụng

### Bước 1: Đặt ảnh vào thư mục input

```
dags/data/remove_bg_input/
├── image1.jpg
├── image2.png
├── photo.webp
└── ...
```

### Bước 2: Trigger DAG

- Mở Airflow UI → DAG `com.image.remove_background`
- Click **Trigger DAG**

### Bước 3: Lấy kết quả

```
dags/data/remove_bg_output/
├── image1.png      # Ảnh đã xóa nền (transparent)
├── image2.png
├── photo.png
└── report.json     # Báo cáo xử lý
```

---

## 🔄 Pipeline

```
setup_environment >> remove_background >> generate_report
```

| Task | Chức năng |
|------|-----------|
| **setup_environment** | Tạo thư mục, quét ảnh đầu vào |
| **remove_background** | Xóa nền với AI model |
| **generate_report** | Tạo báo cáo JSON |

---

## 🧠 Các Model hỗ trợ

| Model | Chất lượng | Tốc độ | Ghi chú |
|-------|------------|--------|---------|
| `u2net` | ⭐⭐⭐⭐⭐ | ⚡⚡ | Mặc định, chất lượng tốt nhất |
| `u2netp` | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Nhanh hơn, vẫn tốt |
| `u2net_human_seg` | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Tối ưu cho người |
| `silueta` | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ | Rất nhanh |
| `isnet-general-use` | ⭐⭐⭐⭐ | ⚡⚡⚡ | Đa dụng |
| `isnet-anime` | ⭐⭐⭐⭐ | ⚡⚡⚡ | Tối ưu cho anime |

---

## ⚙️ Cấu hình

Chỉnh sửa trong `config.py`:

```python
REMBG_CONFIG = {
    'model': 'u2net',           # Model AI sử dụng
    'alpha_matting': False,     # Xử lý viền mịn hơn
    'output_format': 'png',     # Format output
    'bgcolor': None,            # None = transparent
}
```

### Alpha Matting

Bật `alpha_matting=True` để có viền mịn hơn (tốn thêm thời gian):

```python
REMBG_CONFIG = {
    'model': 'u2net',
    'alpha_matting': True,
    'alpha_matting_fg_threshold': 240,
    'alpha_matting_bg_threshold': 10,
    'alpha_matting_erode_size': 10,
}
```

### Thay đổi màu nền

```python
# Nền trắng
'bgcolor': (255, 255, 255, 255)

# Nền đỏ
'bgcolor': (255, 0, 0, 255)

# Transparent (mặc định)
'bgcolor': None
```

---

## 📁 Cấu trúc Output

```
dags/data/remove_bg_output/
├── image1.png          # Ảnh đã xóa nền
├── image2.png
├── ...
└── report.json         # Báo cáo chi tiết
```

### Nội dung report.json

```json
{
  "generated_at": "2026-01-21T10:30:00",
  "config": {
    "model": "u2net",
    "alpha_matting": false
  },
  "summary": {
    "total_images": 10,
    "success": 9,
    "errors": 1
  },
  "results": [
    {
      "input": "/opt/airflow/dags/data/remove_bg_input/image1.jpg",
      "output": "/opt/airflow/dags/data/remove_bg_output/image1.png",
      "status": "success"
    }
  ]
}
```

---

## 📋 Định dạng hỗ trợ

### Input
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### Output
- `.png` (với alpha channel cho transparency)

---

## 🛠️ Yêu cầu

### Dependencies

```bash
pip install rembg[gpu] onnxruntime-gpu
# hoặc CPU only:
pip install rembg onnxruntime
```

### Thêm vào requirements.txt

```
rembg>=2.0.50
onnxruntime>=1.15.0
# hoặc onnxruntime-gpu cho GPU
```

---

## 📊 Performance

| Model | Thời gian/ảnh (CPU) | Thời gian/ảnh (GPU) |
|-------|---------------------|---------------------|
| u2net | ~3-5s | ~0.5-1s |
| u2netp | ~1-2s | ~0.2-0.5s |
| silueta | ~0.5-1s | ~0.1-0.3s |

---

## 📝 Lưu ý

1. **Lần đầu chạy** sẽ download model (~170MB cho u2net)
2. **GPU** sẽ nhanh hơn đáng kể (5-10x)
3. **Ảnh lớn** sẽ tốn nhiều RAM hơn
4. **Output luôn là PNG** để giữ transparency

---

## 🔗 Liên kết

- [rembg GitHub](https://github.com/danielgatis/rembg)
- [U²-Net Paper](https://arxiv.org/abs/2005.09007)
- [IS-Net Paper](https://arxiv.org/abs/2203.16257)
