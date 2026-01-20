FROM apache/airflow:3.1.6-python3.10

USER root

# Install system dependencies first (system layer)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements files
COPY requirements.txt /requirements.txt
COPY requirements_add.txt /requirements_add.txt

# Install Python dependencies (cached separately)
RUN pip install  \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --retries 20 \
    --timeout 300 \
    -r /requirements.txt

RUN pip install  \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --retries 20 \
    --timeout 300 \
    -r /requirements_add.txt
