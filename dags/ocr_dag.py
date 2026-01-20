from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
requests.packages.urllib3.disable_warnings()

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2
import numpy as np
import json
import unicodedata
import os

def remove_accents(s: str) -> str:
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def get_cls_text(det_result):
    cls_results = []
    if det_result and det_result[0]:
        for line in det_result[0]:
            if len(line) > 2:
                angle = line[2]
                cls_results.append(angle)
            else:
                cls_results.append(None)
    return cls_results

def run_ocr():
    img_path = '/opt/airflow/333.png'  # Adjust path for container
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    paddle_ocr = PaddleOCR(lang='en', use_gpu=False)
    det_result = paddle_ocr.ocr(img_path, det=True, rec=True, cls=True)
    image = cv2.imread(img_path)

    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    vietocr = Predictor(config)

    final_results = []
    cls_results = get_cls_text(det_result)

    if det_result and det_result[0]:
        for i, line in enumerate(det_result[0]):
            box = line[0]
            pts = np.array(box, dtype="float32")
            x_min = int(min(p[0] for p in pts))
            y_min = int(min(p[1] for p in pts))
            x_max = int(max(p[0] for p in pts))
            y_max = int(max(p[1] for p in pts))

            crop = image[y_min:y_max, x_min:x_max]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            vietocrText = vietocr.predict(crop_pil)
            paddle_ocrText = line[1][0]

            cls_text = cls_results[i] if i < len(cls_results) else None

            final_results.append({"box": box, "VietOCR": vietocrText, "PaddleOCR": paddle_ocrText, "CLS": cls_text})
            
            print(f"Box: {box}\nVietOCR: {vietocrText}\nPaddleOCR: {paddle_ocrText}\nCLS: {cls_text}\n")
    
    # Save results
    with open('/opt/airflow/logs/ocr_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    print("OCR processing completed.")

with DAG(
    dag_id="ocr_processing_dag",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    ocr_task = PythonOperator(
        task_id="run_ocr_task",
        python_callable=run_ocr
    )