#========== BASE STAGE: Common setup ==========
FROM apache/airflow:3.1.6-python3.10 
COPY requirements.txt /tmp/
COPY check_venv.sh /opt/airflow/check_venv.sh
USER root
RUN chmod +x /opt/airflow/check_venv.sh
USER airflow
RUN pip install \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --retries 20 \
    --timeout 600 \
    -r /tmp/requirements.txt && \
    pip install \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --retries 20 \
    --timeout 600 \
    apache-airflow-providers-celery \
    apache-airflow-providers-postgres \
    apache-airflow-providers-redis; 
