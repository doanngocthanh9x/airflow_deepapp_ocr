"""
DAG: NER from Text (Named Entity Recognition)
Fine-tune PhoBERT for Vietnamese NER task

Pipeline:
    1. Prepare Dataset - Load and preprocess CoNLL format data
    2. Train NER Model - Fine-tune PhoBERT
    3. Evaluate Model - Test accuracy and metrics
    4. Export Model - Save model for inference

Entity Types:
    - PER: Person
    - ORG: Organization
    - LOC: Location
    - DATE: Date/Time
    - MISC: Miscellaneous
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path

from dag_ner_from_text.tasks import (
    prepare_dataset,
    train_ner_model,
    evaluate_ner_model,
    export_ner_model,
)

# Load README for documentation
_readme_path = Path(__file__).parent / 'README.md'
DAG_DOC_MD = _readme_path.read_text(encoding='utf-8') if _readme_path.exists() else __doc__

# Default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id='com.nlp.ner_from_text',
    default_args=default_args,
    description='NER for Vietnamese text using PhoBERT',
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['nlp', 'ner', 'phobert', 'vietnamese'],
    doc_md=DAG_DOC_MD,
) as dag:
    
    # Task 1: Prepare Dataset
    task_prepare = PythonOperator(
        task_id='prepare_dataset',
        python_callable=prepare_dataset,
        doc_md="""
        ## Prepare Dataset
        
        - Load CoNLL format text files (train.txt, dev.txt, test.txt)
        - Tokenize and align labels
        - Save to HuggingFace dataset format
        """,
    )
    
    # Task 2: Train Model
    task_train = PythonOperator(
        task_id='train_ner_model',
        python_callable=train_ner_model,
        doc_md="""
        ## Train NER Model
        
        - Load PhoBERT model (vinai/phobert-base)
        - Fine-tune on NER task
        - Save best checkpoint based on F1 score
        """,
    )
    
    # Task 3: Evaluate Model
    task_evaluate = PythonOperator(
        task_id='evaluate_ner_model',
        python_callable=evaluate_ner_model,
        doc_md="""
        ## Evaluate Model
        
        - Evaluate on test set
        - Calculate metrics: F1, Precision, Recall
        - Generate error analysis
        """,
    )
    
    # Task 4: Export Model
    task_export = PythonOperator(
        task_id='export_ner_model',
        python_callable=export_ner_model,
        doc_md="""
        ## Export Model
        
        - Export trained model for inference
        - Save label mappings
        - Ready for deployment
        """,
    )
    
    # Pipeline
    task_prepare >> task_train >> task_evaluate >> task_export
