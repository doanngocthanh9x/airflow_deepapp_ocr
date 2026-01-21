# 📋 Document Classifier Training DAG

> **DAG ID:** `com.train.document_classifier`  
> **Phiên bản:** 1.0.0  
> **Cập nhật:** Tháng 1/2026

---

## 📖 Giới thiệu

Hiện tại, việc nhiều loại giấy tờ doanh nghiệp chưa được phân loại tự động gây khó khăn trong quy trình xử lý tài liệu. DAG này được thiết kế để **huấn luyện mô hình phân loại hình ảnh** nhận diện các loại giấy tờ phổ biến như:

- 🪪 CCCD (Căn cước công dân) các phiên bản
- 🆔 CMND (Chứng minh nhân dân)
- 📄 Giấy khai sinh
- 📑 Các loại giấy tờ khác...

### Ví dụ hình ảnh giấy tờ

![Ví dụ giấy tờ](image/example1.png)
---

## 🎯 Mục tiêu

| Mục tiêu | Mô tả |
|----------|-------|
| **Tự động phân loại** | Nhận diện loại giấy tờ từ ảnh chụp/scan |
| **Độ chính xác cao** | Đạt >95% accuracy trên tập validation |
| **Tốc độ inference nhanh** | <50ms/ảnh trên CPU |
| **Dễ triển khai** | Export model dạng inference để dùng trong production |

---

## 📂 Cấu trúc Dataset

```
dags/data/
├── cccd_12_number_back/     # CCCD 12 số - Mặt sau
├── cccd_12_number_front/    # CCCD 12 số - Mặt trước
├── cccd_new_back/           # CCCD mới - Mặt sau
├── cccd_new_front/          # CCCD mới - Mặt trước
├── cccd_qr_back/            # CCCD có QR - Mặt sau
├── cccd_qr_front/           # CCCD có QR - Mặt trước
├── cmnd_12_number_front/    # CMND 12 số - Mặt trước
├── cmnd_back/               # CMND - Mặt sau
├── cmnd_front/              # CMND - Mặt trước
└── giay_khai_sinh/          # Giấy khai sinh
```

### Yêu cầu dữ liệu

- ✅ Mỗi folder là **một nhãn (class)**
- ✅ Chứa các file ảnh định dạng: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- ✅ Tên folder = tên class (không dấu, dùng underscore)
- ✅ Khuyến nghị: **>100 ảnh/class** để model học tốt

---

## 🧠 Kiến trúc Model

### MobileNetV3 Small

Model sử dụng để fine-tune là **MobileNetV3 Small (x1.0)** - một kiến trúc CNN hiện đại, tối ưu cho thiết bị edge/mobile.

| Thông số | Giá trị |
|----------|---------|
| **Backbone** | MobileNetV3 Small |
| **Input size** | 224 × 224 × 3 |
| **Parameters** | ~2.5M |
| **FLOPs** | ~65M |
| **ImageNet Top-1** | ~67.4% |

### ✅ Ưu điểm

| Ưu điểm | Chi tiết |
|---------|----------|
| **🚀 Nhẹ và nhanh** | Số tham số ít, FLOPs thấp, phù hợp mobile/edge device, Raspberry Pi |
| **🎯 Generalize tốt** | Pretrain trên ImageNet, trích xuất đặc trưng hiệu quả cho dataset nhỏ/vừa |
| **📦 Dễ triển khai** | Hỗ trợ sẵn trong PaddlePaddle, PyTorch, TensorFlow. Export ONNX, TFLite dễ dàng |

### ⚠️ Hạn chế

| Hạn chế | Chi tiết |
|---------|----------|
| **📉 Accuracy thấp hơn** | So với ResNet50, EfficientNet-B3 do ưu tiên tốc độ |
| **🔍 Biểu diễn hạn chế** | Có thể underfit với bài toán fine-grained hoặc nhiều class phức tạp |

### 💡 Khi nào nên dùng?

- ✅ Ứng dụng mobile/IoT, yêu cầu latency thấp, real-time
- ✅ Dataset vừa phải (10-100 classes), ảnh không quá phức tạp
- ✅ Ưu tiên tốc độ và tài nguyên hơn độ chính xác tuyệt đối

---

## ⚙️ Cấu hình Training

```python
# Model Config
MODEL_CONFIG = {
    'backbone': 'mobilenet_v3_small',
    'image_size': 224,
    'pretrained': True,
}

# Training Config
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'lr_scheduler': 'CosineAnnealingDecay',
    'warmup_epochs': 2,
}

# Data Split
DATA_SPLIT = {
    'train': 0.8,   # 80%
    'val': 0.15,    # 15%
    'test': 0.05,   # 5%
}
```

---

## 🔄 DAG Pipeline

DAG gồm **4 tasks** chạy tuần tự:

```
prepare_dataset >> train_model >> evaluate_model >> export_inference_model
```

---

### 📊 Task 1: Prepare Dataset

Chuẩn bị và chia dữ liệu thành train/val/test sets.

**Chức năng:**

- Quét thư mục data, xác định classes từ tên folder
- Đếm số ảnh mỗi class, kiểm tra cân bằng
- Random split theo tỷ lệ 80/15/5
- Tạo file `train.txt`, `val.txt`, `test.txt`

![Prepare Dataset](image/dag_step1.png)
---

### 🏋️ Task 2: Train Model

Huấn luyện model MobileNetV3 với dữ liệu đã chuẩn bị.

**Chức năng:**

- Load pretrained weights từ ImageNet
- Replace classifier layer cho số classes mới
- Training với data augmentation (resize, flip, rotate, color jitter)
- Learning rate scheduler với warmup
- Save best model theo validation accuracy
- Log metrics qua mỗi epoch

![Train Model](image/dag_step2.png)

---

### 📈 Task 3: Evaluate Model

Đánh giá model trên tập test.

**Chức năng:**

- Load best checkpoint
- Inference trên toàn bộ test set
- Tính toán metrics: Accuracy, Precision, Recall, F1-score
- Tạo Confusion Matrix
- Phân tích per-class performance
- Xuất báo cáo evaluation

![Evaluate Model](image/dag_step3.png)

---

### 💾 Task 4: Export Inference Model

Export model sang định dạng inference để deploy.

**Chức năng:**

- Load best checkpoint
- Chuyển đổi sang static graph
- Export `.pdmodel` + `.pdiparams`
- Tạo `config.json` với preprocessing params
- Tạo `label_map.json` mapping index → class name

![Save Model](image/dag_step4.png)

---

## 📁 Output Structure

```
models/doc_classifier/
├── dataset/
│   ├── train.txt          # Training file list
│   ├── val.txt            # Validation file list
│   ├── test.txt           # Test file list
│   └── class_info.json    # Class statistics
│
├── checkpoints/
│   ├── best_model.pdparams    # Best model weights
│   ├── final_model.pdparams   # Final epoch weights
│   └── training_history.json  # Loss/accuracy per epoch
│
├── evaluation/
│   ├── report.json        # Detailed metrics
│   ├── confusion_matrix.png
│   └── per_class_metrics.json
│
└── inference/
    ├── model.pdmodel      # Inference model
    ├── model.pdiparams    # Model parameters
    ├── config.json        # Preprocessing config
    └── label_map.json     # Class mapping
```

---

## 🚀 Cách sử dụng Model

### Command Line

```bash
cd services
python doc_classifier.py path/to/image.jpg
```

### Python Code

```python
from doc_classifier import create_classifier

# Khởi tạo classifier
classifier = create_classifier()

# Predict
result = classifier.predict("image.jpg")
print(result)
# {
#     'class': 'cccd_qr_front',
#     'label': 'CCCD QR - Mặt trước',
#     'confidence': 0.98,
#     'top_k': [...]
# }

# Batch predict
results = classifier.predict_batch(["img1.jpg", "img2.jpg"])
```

---

## 📊 Kết quả Training

| Metric | Giá trị |
|--------|---------|
| **Best Epoch** | 5 |
| **Best Accuracy** | **99.31%** |
| **Total Classes** | 10 |
| **Total Images** | ~3,938 |
| **Training Time** | ~10 phút |

---

## 📝 Lưu ý

1. **GPU**: Training hỗ trợ cả CPU và GPU. Với GPU sẽ nhanh hơn ~5-10x
2. **Data Balance**: Nên có số lượng ảnh tương đương giữa các class
3. **Augmentation**: Có thể tùy chỉnh trong config để tăng độ đa dạng
4. **Early Stopping**: Model tự lưu checkpoint tốt nhất theo val accuracy

---

## 🔗 Liên kết

- [PaddlePaddle Documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [Apache Airflow](https://airflow.apache.org/)
