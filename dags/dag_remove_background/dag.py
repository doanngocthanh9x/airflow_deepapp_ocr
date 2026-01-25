"""
DAG: Remove Background
Batch remove background from images using rembg (U2-Net, IS-Net, etc.)

Pipeline:
    1. Setup Environment - Validate directories and scan images
    2. Remove Background - Process images with AI model
    3. Generate Report - Create processing report

Usage:
    1. Put images in: dags/data/remove_bg_input/
    2. Trigger DAG
    3. Get results from: dags/data/remove_bg_output/

Models available:
    - u2net: Best quality, slower
    - u2netp: Faster, good quality
    - u2net_human_seg: Optimized for humans
    - silueta: Fast, general purpose
    - isnet-general-use: Good for various objects
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path

from dag_remove_background.tasks import (
    setup_environment,
    remove_background_batch,
    generate_report,
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
    'retry_delay': timedelta(minutes=2),
}

# DAG definition
with DAG(
    dag_id='com.image.remove_background',
    default_args=default_args,
    description='Remove background from images using AI (rembg)',
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['image', 'processing', 'rembg', 'background'],
    doc_md=DAG_DOC_MD,
) as dag:
    
    # Task 1: Setup
    task_setup = PythonOperator(
        task_id='setup_environment',
        python_callable=setup_environment,
        doc_md="""
        ## Setup Environment
        
        - Create input/output directories
        - Scan for images to process
        - Validate configuration
        """,
    )
    
    # Task 2: Remove Background
    task_remove_bg = PythonOperator(
        task_id='remove_background',
        python_callable=remove_background_batch,
        doc_md="""
        ## Remove Background
        
        - Load rembg model (U2-Net by default)
        - Process each image
        - Save transparent PNG output
        """,
    )
    
    # Task 3: Report
    task_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
        doc_md="""
        ## Generate Report
        
        - Create JSON report with results
        - Summary of success/errors
        - Save to output directory
        """,
    )
    
    # Pipeline
    task_setup >> task_remove_bg >> task_report
