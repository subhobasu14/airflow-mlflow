#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/airflow/.local/lib/python3.8/site-packages

# Initialize the Airflow database
airflow db init

# Create an Airflow user if not already created
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start MLflow server
mlflow server --backend-store-uri sqlite:////opt/airflow/mlflow/mlflow.db --default-artifact-root /opt/airflow/mlflow --host 0.0.0.0 &

# Start Airflow scheduler
airflow scheduler &

# Start Airflow webserver
airflow webserver