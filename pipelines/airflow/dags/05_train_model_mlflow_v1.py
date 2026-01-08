from __future__ import annotations

import os
from datetime import datetime, timezone
from airflow import DAG
from airflow.operators.python import PythonOperator


def _run(ds: str, **context):
    run_id = context["run_id"].replace(":", "_").replace("+", "_")
    parquet_path = os.getenv("TRAIN_DATA_PARQUET", f"s3a://mlops/silver/training_dataset/dt={ds}/run_id={run_id}")

    from pipelines.training.train_model import TrainConfig, train_and_log
    mlflow_run_id = train_and_log(TrainConfig(parquet_path=parquet_path))
    print(f"[train] mlflow_run_id={mlflow_run_id}")


with DAG(
    dag_id="05_train_model_mlflow_v1",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["mlops", "train"],
) as dag:
    PythonOperator(
        task_id="train_and_log_mlflow",
        python_callable=_run,
    )