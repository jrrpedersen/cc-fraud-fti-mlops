from __future__ import annotations

from datetime import datetime, timezone
from airflow import DAG
from airflow.operators.python import PythonOperator


def _run(ds: str, **context):
    run_id = context["run_id"].replace(":", "_").replace("+", "_")
    from pipelines.spark_jobs.build_training_dataset import build_training_dataset
    out = build_training_dataset(ds=ds, run_id=run_id)
    print(f"[training_dataset] wrote: {out}")


with DAG(
    dag_id="04_build_training_dataset_v1",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["mlops", "train"],
) as dag:
    PythonOperator(
        task_id="build_training_dataset",
        python_callable=_run,
    )