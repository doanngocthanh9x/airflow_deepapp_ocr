"""
DAG: Train Document Classifier
Fine-tune PPLCNetV2/MobileNetV3 for Vietnamese ID document classification

Pipeline:
    1. Prepare Dataset - Split train/val/test
    2. Train Model - Fine-tune with pretrained weights
    3. Evaluate Model - Test accuracy and metrics
    4. Export Model - Save inference-ready model

Classes:
    - CCCD (12 số, mới, QR) - front/back
    - CMND - front/back
    - Giấy khai sinh
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path

# Import tasks
from dag_train_classifier.tasks import (
    prepare_dataset,
    train_model,
    evaluate_model,
    export_inference_model,
)

# Load README.md for DAG documentation
_readme_path = Path(__file__).parent / 'README.md'
DAG_DOC_MD = _readme_path.read_text(encoding='utf-8') if _readme_path.exists() else __doc__

# Default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}


# DAG definition
with DAG(
    dag_id='com.train.document_classifier',
    default_args=default_args,
    description='Train document classifier (CCCD, CMND, GKS)',
    schedule=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'classification', 'paddle'],
    doc_md=DAG_DOC_MD,
) as dag:
    
    # Task 1: Prepare Dataset
    task_prepare = PythonOperator(
        task_id='prepare_dataset',
        python_callable=prepare_dataset,
        doc_md="""
        ## Prepare Dataset
        
        - Read images from `dags/data/{class_name}/`
        - Split into train (80%) / val (15%) / test (5%)
        - Create `train_list.txt`, `val_list.txt`, `test_list.txt`
        - Save to `models/doc_classifier/dataset/`
        """,
    )
    
    # Task 2: Train Model
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        doc_md="""
        ## Train Model
        
        - Model: MobileNetV3-Small (or PPLCNetV2 if PaddleClas available)
        - Pretrained: ImageNet weights
        - Fine-tune: 30 epochs with cosine LR decay
        - Save best model based on validation accuracy
        """,
    )
    
    # Task 3: Evaluate Model
    task_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        doc_md="""
        ## Evaluate Model
        
        - Run inference on test set
        - Compute accuracy, precision, recall, F1
        - Generate confusion matrix
        - Save classification report
        """,
    )
    
    # Task 4: Export Inference Model
    task_export = PythonOperator(
        task_id='export_inference_model',
        python_callable=export_inference_model,
        doc_md="""
        ## Export Inference Model
        
        - Convert to static graph (paddle.jit.save)
        - Save model.pdmodel, model.pdiparams
        - Save label_map.json and config.json
        - Ready for deployment
        """,
    )
    
    # Pipeline
    task_prepare >> task_train >> task_evaluate >> task_export
