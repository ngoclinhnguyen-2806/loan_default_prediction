from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

# Wait for upstream data_pipeline to be completed
UPSTREAM_DAG_ID = 'data_pipeline_dag'
UPSTREAM_TASK_ID = 'gold_features'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'daily_inference_dag',
    default_args=default_args,
    description='Daily model prediction pipeline, dependent on feature store completion',
    schedule_interval=None,
    start_date=datetime(2024, 9, 1),
    catchup=True,
) as dag:
    
# 1. Sensor Task: Waits for the feature store to be ready
    wait_for_feature_store = ExternalTaskSensor(
        task_id='wait_for_gold_features',
        external_dag_id=UPSTREAM_DAG_ID,            # The name of the feature creation DAG
        external_task_id=UPSTREAM_TASK_ID,          # The final task in that DAG
        # execution_delta=timedelta(0),             # Optional: Can be used if schedules are offset
        mode='reschedule',                          # Frees up a worker slot while waiting
        timeout=timedelta(hours=6).total_seconds(), # Maximum time to wait
        # Use execution_date_fn for complex schedule dependencies
    )

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
    wait_for_feature_store >> run_model_prediction >> run_monitoring >> pipeline_complete