from __future__ import annotations

import os
from datetime import datetime, timezone
from airflow import DAG
from airflow.operators.python import PythonOperator


def _s3_client():
    import boto3
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name=region,
    )

def _read_pointer(bucket: str, key: str) -> str:
    s3 = _s3_client()
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8").strip()

def _run(ds: str, **context):
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    ptr_name = os.getenv("TRAIN_DATA_POINTER", "latest").lower()
    ptr_file = "_LATEST" if ptr_name == "latest" else "_CURRENT"

    pointer_key = f"silver/training_dataset/dt={ds}/{ptr_file}"
    prefix = _read_pointer(bucket, pointer_key)   # e.g. silver/.../run_id=...
    parquet_path = f"s3a://{bucket}/{prefix}"

    from pipelines.training.train_model import TrainConfig, train_and_log
    cfg = TrainConfig(
        parquet_path=parquet_path,
        ds=ds,
        pointer_name=ptr_name,
        pointer_key=pointer_key,
        resolved_prefix=prefix,
    )
    mlflow_run_id = train_and_log(cfg)

    print(f"[train] parquet_path={parquet_path}")
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