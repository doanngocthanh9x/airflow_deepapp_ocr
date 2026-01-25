# 🏷️ Named Entity Recognition (NER) DAG

> **DAG ID:** `com.nlp.ner_from_text`  
> **Model:** PhoBERT-base  
> **Language:** Vietnamese  
> **Version:** 1.0.0

---

## 📖 Giới thiệu

DAG này dùng để **huấn luyện mô hình nhận dạng thực thể được đặt tên (NER)** cho tiếng Việt bằng **PhoBERT** - một mô hình BERT được huấn luyện trước dành riêng cho ngôn ngữ Việt.

### Ứng dụng

- 🏢 Trích xuất tên công ty, tổ chức
- 👤 Xác định tên người
- 📍 Định vị địa danh
- 📅 Nhận dạng thời gian, ngày tháng
- 📄 Trích xuất thông tin từ tài liệu

### Các loại thực thể hỗ trợ

| Loại | Ký hiệu | Ví dụ |
|------|---------|-------|
| **Person** | PER | Nguyễn Văn A, Hồ Chí Minh |
| **Organization** | ORG | Công ty Google, Bộ Giáo dục |
| **Location** | LOC | Hà Nội, Mỹ, Sông Hồng |
| **Date/Time** | DATE | Ngày 1/1/2024, Tháng 3 |
| **Miscellaneous** | MISC | Tiếng Anh, tôn giáo Phật |

---

## 🚀 Cách sử dụng

### Bước 1: Chuẩn bị dữ liệu

Tạo các file theo định dạng **CoNLL** (một token + nhãn trên mỗi dòng):

```
dags/data/ner_text_input/
├── train.txt
├── dev.txt
└── test.txt
```

**Định dạng file (CoNLL):**
```
Nguyễn O
Văn B-PER
A I-PER
làm O
việc O
tại O
Công B-ORG
ty I-ORG
Google I-ORG
. O

(empty line = sentence separator)
```

### Bước 2: Trigger DAG

```
Airflow UI → DAG "com.nlp.ner_from_text" → Trigger DAG
```

### Bước 3: Lấy kết quả

```
dags/data/ner_text_output/
├── model_export/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── label_config.json
└── predictions/
    └── test_predictions.txt
```

---

## 🔄 Pipeline

```
prepare_dataset >> train_ner_model >> evaluate_ner_model >> export_ner_model
```

| Task | Chức năng | Input | Output |
|------|-----------|-------|--------|
| **prepare_dataset** | Tải dữ liệu CoNLL | train.txt, dev.txt, test.txt | Tokenized dataset |
| **train_ner_model** | Huấn luyện PhoBERT | Dataset | Trained model |
| **evaluate_ner_model** | Đánh giá trên test set | Trained model | Metrics (F1, Precision, Recall) |
| **export_ner_model** | Export để inference | Trained model | Model files + label mapping |

---

## 🧠 PhoBERT Model

### Thông số kỹ thuật

| Thông số | Giá trị |
|----------|--------|
| **Model** | vinai/phobert-base |
| **Type** | RoBERTa-based |
| **Parameters** | ~135M |
| **Vocab** | 64K tokens |
| **Max length** | 256 tokens |
| **Pretrained on** | Vietnamese Wikipedia + News |

### Tính năng

- ✅ **Pretrained trên tiếng Việt**: Tối ưu cho ngôn ngữ Việt
- ✅ **Word segmentation**: Tự động cắt từ
- ✅ **Transformer architecture**: Hiệu quả cao
- ✅ **Fine-tuning nhanh**: Hội tụ nhanh trên data nhỏ

---

## ⚙️ Cấu hình Training

Chỉnh sửa trong `config.py`:

```python
NER_CONFIG = {
    'model_name': 'vinai/phobert-base',
    'max_seq_length': 256,      # Độ dài tối đa
    'batch_size': 32,           # Kích thước batch
    'num_epochs': 10,           # Số epoch
    'learning_rate': 5e-5,      # Learning rate
    'warmup_steps': 500,        # Warmup steps
}
```

### Các thông số quan trọng

- **max_seq_length**: Tăng nếu câu dài, giảm để tiết kiệm bộ nhớ
- **batch_size**: Tăng để training nhanh hơn (nếu GPU đủ)
- **num_epochs**: 10-20 epochs thường đủ
- **learning_rate**: 5e-5 hoặc 2e-5 cho fine-tuning

---

## 📊 Định dạng Dữ liệu

### Input Format (CoNLL)

Mỗi dòng = 1 token + 1 nhãn (cách bằng khoảng trắng hoặc tab):

```
word1 label1
word2 label2
...
(empty line for sentence boundary)
```

**Ví dụ:**
```
Công B-ORG
ty I-ORG
Google I-ORG
tuyển O
dụng O
nhân O
viên O
tại O
Hà B-LOC
Nội I-LOC
. O

(empty line here)

Họ B-PER
có O
mức O
lương O
cao O
. O
```

### Label Tags

```
B-LABEL  = Beginning of entity
I-LABEL  = Inside/continuation of entity
O        = Outside any entity
```

### Ví dụ với NER tags

```
Nguyễn B-PER
Văn I-PER
A I-PER
làm O
việc O
tại O
Google B-ORG
. O
```

---

## 📁 Output Structure

```
dags/data/ner_text_output/
├── model_export/
│   ├── pytorch_model.bin          # Model weights
│   ├── config.json                # Model config
│   ├── tokenizer.json             # Tokenizer
│   ├── special_tokens_map.json
│   └── label_config.json          # Label mapping
│
└── predictions/
    ├── test_predictions.txt       # Raw predictions
    └── metrics.json               # F1, Precision, Recall
```

---

## 🛠️ Yêu cầu Dependencies

```bash
pip install transformers>=4.30.0
pip install datasets>=2.10.0
pip install torch>=2.0.0
pip install seqeval              # For NER metrics
pip install fire                 # CLI
```

Thêm vào requirements.txt:

```
transformers>=4.30.0
datasets>=2.10.0
seqeval>=2.2.1
```

---

## 📊 Metrics

DAG sử dụng các metrics NER tiêu chuẩn:

### Token-level metrics
- **Accuracy**: Tỷ lệ token được phân loại đúng

### Entity-level metrics (seqeval)
- **Precision**: P = correct / predicted
- **Recall**: R = correct / gold
- **F1-score**: Harmonic mean của precision và recall

---

## 💡 Tips

1. **Cân bằng dữ liệu**: Nên có số lượng entity tương đương giữa các loại
2. **Data cleaning**: Xóa duplicate, fix encoding trước huấn luyện
3. **Validation set**: Dùng dev.txt để monitor overfitting
4. **Checkpoint**: Model tự save best checkpoint theo F1 score
5. **GPU**: Nên dùng GPU (5-10x nhanh hơn CPU)

---

## 📝 Lưu ý

- PhoBERT không tự động cắt từ, cần word-segmented input
- Labels phải chứa "O" tag ở đầu
- Token được pad với label `-100` (ignored in loss calculation)
- Model được save dưới định dạng PyTorch

---

## 🔗 Liên kết

- [PhoBERT GitHub](https://github.com/VinAIResearch/PhoBERT)
- [Phobert NER Reference](https://github.com/Avi197/Phobert-Named-Entity-Reconigtion)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [seqeval Metrics](https://github.com/chakki-works/seqeval)
- [CoNLL Format](https://www.clips.uantwerpen.be/conll2003/ner/)
