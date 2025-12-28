from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
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


def generate_reference_world(**context) -> dict:
    """
    Generates a stable set of reference entities (banks, merchants, accounts, cards)
    into local JSONL files under /tmp, returning a mapping of dataset->local_path.

    Also generates a world_id and returns it via XCom.
    """
    # With PYTHONPATH=/opt/airflow/repo and the repo mounted at /opt/airflow/repo,
    # we import via the data_gen package.
    from data_gen.ccfraud_gen.bootstrap import generate_reference_world as gen_world

    world_id = f"world_{uuid.uuid4().hex[:12]}"
    seed = int(os.getenv("GEN_SEED", "42"))

    out_dir = Path(f"/tmp/{world_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n_banks": int(os.getenv("N_BANKS", "25")),
        "n_merchants": int(os.getenv("N_MERCHANTS", "2000")),
        "n_accounts": int(os.getenv("N_ACCOUNTS", "20000")),
        "n_cards": int(os.getenv("N_CARDS", "40000")),
    }

    paths = gen_world(out_dir=out_dir, seed=seed, cfg=cfg)

    result = {"world_id": world_id, "paths": {k: str(v) for k, v in paths.items()}}
    print(f"Generated reference world_id={world_id} with datasets={list(paths.keys())}")
    return result


def upload_reference_world(**context) -> str:
    """
    Upload the world JSONL datasets to:
      landing/reference/world_id=<WORLD_ID>/<dataset>.jsonl
    Returns the world_id.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    payload = context["ti"].xcom_pull(task_ids="generate_reference_world")
    world_id = payload["world_id"]
    paths: dict = payload["paths"]

    for dataset, local_path in paths.items():
        key = f"landing/reference/world_id={world_id}/{dataset}.jsonl"
        s3.upload_file(local_path, bucket, key)
        print(f"Uploaded {dataset}: {local_path} -> s3://{bucket}/{key}")

    return world_id


def write_current_world_pointer(**context) -> str:
    """
    Write/overwrite landing/reference/_CURRENT containing the world_id.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    world_id = context["ti"].xcom_pull(task_ids="upload_reference_world")

    key = "landing/reference/_CURRENT"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(world_id + "\n").encode("utf-8"),
        ContentType="text/plain",
    )
    print(f"Wrote pointer s3://{bucket}/{key} -> {world_id}")
    return key


def verify_reference_world(**context) -> None:
    """
    Verify that _CURRENT exists and referenced datasets exist.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    current = (
        s3.get_object(Bucket=bucket, Key="landing/reference/_CURRENT")["Body"]
        .read()
        .decode("utf-8")
        .strip()
    )
    if not current:
        raise RuntimeError("landing/reference/_CURRENT is empty")
    world_id = current

    required = ["banks", "merchants", "accounts", "cards"]
    for dataset in required:
        key = f"landing/reference/world_id={world_id}/{dataset}.jsonl"
        s3.head_object(Bucket=bucket, Key=key)
        print(f"Verified exists: s3://{bucket}/{key}")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="00_bootstrap_reference_world",
    description="Bootstrap reference entities (banks/merchants/accounts/cards) and set landing/reference/_CURRENT",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase-1", "feature", "reference"],
) as dag:
    t_gen = PythonOperator(task_id="generate_reference_world", python_callable=generate_reference_world)
    t_upload = PythonOperator(task_id="upload_reference_world", python_callable=upload_reference_world)
    t_current = PythonOperator(task_id="write_current_world_pointer", python_callable=write_current_world_pointer)
    t_verify = PythonOperator(task_id="verify_reference_world", python_callable=verify_reference_world)

    t_gen >> t_upload >> t_current >> t_verify