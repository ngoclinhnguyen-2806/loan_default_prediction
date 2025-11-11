from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from airflow.utils.task_group import TaskGroup

DUMMY_DATE = "2024-08-01"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="scheduled_training_dag",
    description="One-shot training using 2024-08-01 as a dummy cutoff.",
    default_args=default_args,
    start_date=datetime(2024, 9, 1),  # any past date is fine
    schedule_interval="@once",        # run once (trigger when ready)
    catchup=False,
) as dag:

    start = EmptyOperator(task_id="start")

    with TaskGroup(group_id="train_ML_models") as training:
        trainxgb = BashOperator(
            task_id="model_train_XGB",
            bash_command=(
                f"python /app/scripts/04_model_train_XGB.py "
                f"--snapshotdate {DUMMY_DATE}"
            )
        )
        trainlr = BashOperator(
            task_id="model_train_LR",
            bash_command=(
                f"python /app/scripts/04_model_train_LR_improved.py "
                f"--snapshotdate {DUMMY_DATE}"
            )
        )

    end = EmptyOperator(task_id="end")

    start >> training >> end
