pip install --no-cache-dir  --trusted-host pypi.org  --trusted-host files.pythonhosted.org  --retries 10  --timeout 120 -r .\requirements.txt

 $env:COMPOSE_DOCKER_CLI_BUILD=1; $env:DOCKER_CLIENT_TIMEOUT=300; 
 $env:COMPOSE_HTTP_TIMEOUT=300; docker compose build


 ```
 wsl
    COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_CLIENT_TIMEOUT=300 COMPOSE_HTTP_TIMEOUT=300 docker compose up -d
 
      COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_CLIENT_TIMEOUT=300 COMPOSE_HTTP_TIMEOUT=300 docker build -t airflow:cpu . --no-cache
    To start the container after building:

    ```
    docker run -d --name airflow-container airflow:cpu
    ```

    To stop the container:

    ```
    docker stop airflow-container
    
 cd /home/dnthanh/airflow_deepapp_ocr && DOCKER_CLIENT_TIMEOUT=300 COMPOSE_HTTP_TIMEOUT=300 docker compose up -d
 ```



```
docker build --no-cache --build-arg BUILD_TYPE=cpu -t airflow:cpu .

docker build --no-cache --build-arg BUILD_TYPE=cpu -t airflow:cpu .
docker run -d --name airflow_cpu airflow:cpu
```
