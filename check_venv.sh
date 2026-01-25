#!/bin/bash
# Check if local .venv is mounted (packages exist in site-packages)
if [ -d "/home/airflow/.local/lib/python3.10/site-packages" ] && [ "$(ls -A /home/airflow/.local/lib/python3.10/site-packages)" ]; then
    export AIRFLOW_USE_LOCAL_PACKAGES=yes
    echo "Using local .venv packages, skipping pip install"
else
    export AIRFLOW_USE_LOCAL_PACKAGES=no
    echo "No local packages found, will install from requirements.txt"
fi
# Continue with original entrypoint
exec "$@"