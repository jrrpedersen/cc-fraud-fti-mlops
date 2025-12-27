from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator


def _s3_client():
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


def generate_local_file(**context) -> str:
    """
    Generate transactions as JSONL and write to /tmp inside the Airflow container.
    Returns the local path (pushed via XCom).
    """
    # Import from mounted repo directory
    from data_gen.generator.generate import generate_transactions, write_jsonl

    run_id = context["run_id"].replace(":", "_").replace("/", "_")
    ds = context["ds"]  # YYYY-MM-DD
    out_path = Path(f"/tmp/transactions_{ds}_{run_id}.jsonl")

    # Deterministic seed derived from run_id hash (stable per run)
    seed = abs(hash(run_id)) % (2**31)

    start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    events = generate_transactions(
        n=5000,
        start_utc=start,
        duration_minutes=60,
        seed=seed,
        n_cards=500,
        n_merchants=200,
    )
    write_jsonl(events, out_path)

    print(f"Generated file: {out_path} (seed={seed})")
    return str(out_path)


def upload_to_minio(**context) -> str:
    """
    Uploads the generated JSONL file to MinIO (bucket 'mlops') under landing/transactions/.
    Returns the object key.
    """
    client = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    ds = context["ds"]
    run_id = context["run_id"].replace(":", "_").replace("/", "_")
    local_path = context["ti"].xcom_pull(task_ids="generate_transactions")

    # Partition-style layout
    key = f"landing/transactions/dt={ds}/run_id={run_id}/transactions.jsonl"

    client.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} -> s3://{bucket}/{key}")
    print(f"S3 object key: {key}")

    return key


def list_landing_prefix(**context) -> None:
    """
    Lists objects under the run's prefix to verify successful upload.
    """
    client = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    ds = context["ds"]
    run_id = context["run_id"].replace(":", "_").replace("/", "_")
    prefix = f"landing/transactions/dt={ds}/run_id={run_id}/"

    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])

    if not contents:
        raise RuntimeError(f"No objects found under s3://{bucket}/{prefix}")

    print(f"Found {len(contents)} object(s) under s3://{bucket}/{prefix}")
    for obj in contents:
        print(f" - {obj['Key']} ({obj['Size']} bytes)")

def preview_local_file(**context) -> None:
    local_path = context["ti"].xcom_pull(task_ids="generate_transactions")
    path = Path(local_path)

    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"Local file: {path}")
    print(f"Record count: {len(lines)}")
    print("--- first 3 lines ---")
    for i in range(min(3, len(lines))):
        print(lines[i])

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="01_generate_and_land_transactions",
    description="Generate synthetic transactions and land them in MinIO (S3) under landing/transactions/",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,  # manual trigger for now
    catchup=False,
    tags=["phase-1", "feature", "landing"],
) as dag:
    t_gen = PythonOperator(
        task_id="generate_transactions",
        python_callable=generate_local_file,
    )

    t_preview = PythonOperator(
    task_id="preview_sample",
    python_callable=preview_local_file,
    )

    t_upload = PythonOperator(
        task_id="upload_to_minio",
        python_callable=upload_to_minio,
    )

    t_verify = PythonOperator(
        task_id="verify_objects_exist",
        python_callable=list_landing_prefix,
    )

    t_gen >> t_preview >> t_upload >> t_verify