pip install --no-cache-dir  --trusted-host pypi.org  --trusted-host files.pythonhosted.org  --retries 10  --timeout 120  -r .\requirements.txt

 $env:COMPOSE_DOCKER_CLI_BUILD=1; $env:DOCKER_CLIENT_TIMEOUT=300; 
 $env:COMPOSE_HTTP_TIMEOUT=300; docker compose build