from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# MLFlow setup
# mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_tracking_uri("http://localhost:5000")

iris = load_iris()

def prepare_data_v1():
    # Example: Original feature set
    # Assume original feature set
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return (X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist())

def prepare_data_v2():
    # Example: Add feature interaction or transformation
    # Example transformation: interaction term for first two features
    data_with_interaction = np.c_[iris.data, iris.data[:, 0] * iris.data[:, 1]]
    X_train, X_test, y_train, y_test = train_test_split(
        data_with_interaction, iris.target, test_size=0.2, random_state=42
    )
    return (X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist())

def train_model_v1(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    import mlflow
    import mlflow.sklearn
    
    X_train, X_test, y_train, y_test = prepare_data_v1()

    mlflow.set_experiment("RandomForestClassifier")
    mlflow.autolog()
    
    train_dataset = mlflow.data.from_pandas(
    X_train, source="iris", name="iris-train", targets="target"
)
    with mlflow.start_run() as run:
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)
        
        mlflow.log_input(train_dataset, context="training")
        
        mlflow.sklearn.log_model(clf, "model_v1")
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model_v1",
            name="RandomForestClassifier"
        )

def train_model_v2(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier

    X_train, X_test, y_train, y_test = prepare_data_v2()
    
    mlflow.set_experiment("KNeighborsClassifier")
    mlflow.autolog()
    with mlflow.start_run() as run:
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", acc)
        
        # mlflow.log_input(X_train)
        mlflow.sklearn.log_model(clf, "model_v2")
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model_v2",
            name="KNeighborsClassifier"
        )

def load_and_predict_v1(X_test, model_version):
    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/RandomForestClassifier/{model_version}"
    )
    predictions = model.predict(X_test)
    print("Predictions with RandomForestClassifier:", predictions)


def load_and_predict_v2(X_test, model_version):
    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/KNeighborsClassifier/{model_version}"
    )
    predictions = model.predict(X_test)
    print("Predictions with KNeighborsClassifier:", predictions)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG('multitask_airflow_poc', 
         default_args=default_args,
         schedule_interval=None) as dag:

    # Prepare data
    task_prepare_data_v1 = PythonOperator(
        task_id='prepare_data_v1',
        python_callable=prepare_data_v1
    )

    task_prepare_data_v2 = PythonOperator(
        task_id='prepare_data_v2',
        python_callable=prepare_data_v2
    )

    # Train models
    task_train_model_v1 = PythonOperator(
    task_id='train_model_v1',
    python_callable=lambda ti: train_model_v1(
        ti.xcom_pull(task_ids='prepare_data_v1')[0],  # X_train
        ti.xcom_pull(task_ids='prepare_data_v1')[2]   # y_train
    )
)

    task_train_model_v2 = PythonOperator(
        task_id='train_model_v2',
        python_callable=lambda ti: train_model_v2(
        ti.xcom_pull(task_ids='prepare_data_v2')[0],  # X_train
        ti.xcom_pull(task_ids='prepare_data_v2')[2]   # y_train
        )
    )

    # Load and use models
    task_load_and_predict_v1 = PythonOperator(
        task_id='load_and_predict_v1',
        python_callable=lambda ti: load_and_predict_v1(ti.xcom_pull(task_ids='prepare_data_v1')[1], "1")
    )

    task_load_and_predict_v2 = PythonOperator(
        task_id='load_and_predict_v2',
        python_callable=lambda ti: load_and_predict_v2(ti.xcom_pull(task_ids='prepare_data_v2')[1], "1")
    )

    # Setting task dependencies
    task_prepare_data_v1 >> task_train_model_v1 >> task_load_and_predict_v1
    task_prepare_data_v2 >> task_train_model_v2 >> task_load_and_predict_v2