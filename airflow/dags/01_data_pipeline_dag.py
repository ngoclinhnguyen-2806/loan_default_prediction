from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    '01_data_pipeline_dag',
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
    bash_command="""
            python /app/scripts/01_create_bronze.py
        """,
    )

    # Silver Processing
    silver = BashOperator(
        task_id='silver',
        bash_command=(
            'python /app/scripts/02_create_silver.py '
        ),
    )

    # Gold Processing
    gold_application = BashOperator(
        task_id="gold_application",        
        bash_command=(
            'python /app/scripts/03_create_applications.py 2024-09-01'
        ),
    )
    
    gold_feature = BashOperator(
        task_id="gold_feature",        
        bash_command=(
            'python /app/scripts/03_create_features.py 2024-09-01'
        ),
    )

    gold_label = BashOperator(
        task_id="gold_label",        
        bash_command=(
            'python /app/scripts/03_create_labels.py 2024-09-01'
        ),
    )

    pipeline_start >> [
    bronze,
    ] >> DummyOperator(task_id="silver_start") >> [
        silver
    ] >> DummyOperator(task_id="gold_start") >> [
        gold_application,
        gold_feature_store,
        gold_label_store
    ]
