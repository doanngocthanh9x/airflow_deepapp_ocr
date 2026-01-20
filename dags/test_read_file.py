from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd

def read_file(**kwargs):
    airflow_home = os.environ.get('AIRFLOW_HOME', os.path.expanduser('~/airflow'))
    file_path = os.path.join(airflow_home, "dags", "data", "example.pkl")
    df = pd.read_pickle(file_path)
    print(df.head()) 

with DAG(
    dag_id="read_file_example",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    PythonOperator(
        task_id="run",
        python_callable=read_file
    ) 
