from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    'inference_monitoring_dag',
    default_args=default_args,
    description='Daily model prediction pipeline, dependent on feature store completion',
    schedule_interval=None,   
    start_date=datetime(2024, 9, 1),  
    catchup=True,             
) as dag:
    
    # data pipeline

    inference_start = DummyOperator(task_id="inference_start")

    # 2. Model Prediction Task
    # This task now contains the core logic for model inference
    run_model_prediction = BashOperator(
        task_id="run_model_prediction",      
        bash_command='python /app/scripts/05_model_inference.py',
    )

    # 3. Daily Monitoring Task
    # This script should analyze the latest features and predictions, check data drift, 
    # and calculate prediction stability metrics.
    run_monitoring = BashOperator(
        task_id="run_monitoring",
        bash_command='python /app/scripts/06_model_monitoring.py',
    )

    pipeline_complete = DummyOperator(
        task_id="inference_monitoring_pipeline_complete",
    )

    # Set the dependencies
    inference_start >> run_model_prediction >> run_monitoring >> pipeline_complete
