from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Manual data pipeline for bronze, silver, and gold processing',
    schedule_interval=None,   # 👈 disables automatic scheduling
    start_date=datetime(2025, 1, 1),  # safe placeholder
    catchup=False,             # 👈 ensures no backfill runs
) as dag:
    
    # data pipeline

    pipeline_start = DummyOperator(task_id="pipeline_start")

    # Bronze Processing
    bronze = BashOperator(
        task_id='bronze',
        bash_command='python /app/scripts/01_create_bronze.py',
    )

    # Silver Processing (run in parallel)
    silver_features = BashOperator(
        task_id='silver_features',
        bash_command='python /app/scripts/02_create_silver_features.py',
    )

    silver_loan = BashOperator(
        task_id='silver_loan',
        bash_command='python /app/scripts/02_create_silver_loan.py',
    )

    # Gold Processing
    gold_application = BashOperator(
        task_id="gold_application",        
        bash_command='python /app/scripts/03_create_applications.py all',
    )
    
    gold_features = BashOperator(
        task_id="gold_features",        
        bash_command='python /app/scripts/03_create_features.py',
    )

    gold_labels = BashOperator(
        task_id="gold_labels",        
        bash_command='python /app/scripts/03_create_labels.py',
    )

    pipeline_start >> [
    bronze,
    ] >> DummyOperator(task_id="silver_start") >> [
        silver_features,
        silver_loan
    ] >> DummyOperator(task_id="gold_application_start") >> [
        gold_application,
    ] >> DummyOperator(task_id="gold_feature_label_start") >> [
        gold_features,
        gold_labels
    ]
