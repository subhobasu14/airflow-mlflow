# Start from the official Airflow image
FROM apache/airflow:2.8.1

# Update system packages
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip

# Switch to airflow user for Python package installations
USER airflow

# Install MLflow and Airflow
RUN pip install --user mlflow scikit-learn

# Validate installations - check in isolated Python script
RUN python3 -c "import mlflow; import airflow; from sklearn.datasets import load_iris; print('Packages verified.')"


# Verify installations
RUN pip list --user

# Switch to root to create directories and set permissions
USER root
RUN mkdir -p /opt/airflow/mlflow && chown airflow /opt/airflow/mlflow

# Expose MLflow default port
EXPOSE 5000

# Copy entrypoint script with root permissions
COPY entrypoint.sh /entrypoint.sh

# Make the script executable
RUN chmod +x /entrypoint.sh

# Copy DAGs to Airflow
COPY dags/ /opt/airflow/dags/

# Switch back to airflow user
USER airflow

# Set the Airflow home and working directories
ENV AIRFLOW_HOME=/opt/airflow
WORKDIR $AIRFLOW_HOME

ENTRYPOINT ["/entrypoint.sh"]