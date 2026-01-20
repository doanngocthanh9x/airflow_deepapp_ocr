from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import requests

def test():
    print(pd.__version__)
    print(requests.get("http://10.160.50.50:5000/").status_code)
    json_response = requests.get("http://10.160.50.50:5000/").json()
    print(json_response)
    print("If you see pandas version and status code 200, requirements are installed correctly.")
with DAG(
    dag_id="test_requirements",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    PythonOperator(
        task_id="run",
        python_callable=test
    )
