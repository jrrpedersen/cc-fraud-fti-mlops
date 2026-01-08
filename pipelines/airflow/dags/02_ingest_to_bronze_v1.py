from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator

from pipelines.spark_jobs.ingest_to_bronze import build_spark, jsonl_to_parquet, stop_spark


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


def _read_text_object(s3, bucket: str, key: str) -> str:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8").strip()


def resolve_landing_latest(**context) -> dict:
    """
    Reads landing _LATEST pointers for transactions and fraud_labels for this DAG run date (ds).
    Returns the landed object keys.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    ds = context["ds"]

    tx_latest = f"landing/transactions/dt={ds}/_LATEST"
    lbl_latest = f"landing/fraud_labels/dt={ds}/_LATEST"

    tx_key = _read_text_object(s3, bucket, tx_latest)
    lbl_key = _read_text_object(s3, bucket, lbl_latest)

    if not tx_key or not lbl_key:
        raise RuntimeError("Missing landing _LATEST pointers. Run 01_generate_and_land_transactions_v1 first.")

    print(f"Resolved landing tx: {tx_key}")
    print(f"Resolved landing labels: {lbl_key}")
    return {"tx_key": tx_key, "lbl_key": lbl_key}


def download_landing_objects(**context) -> dict:
    """
    Downloads landing JSONL objects (transactions + fraud_labels) to /tmp and returns local paths.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    keys = context["ti"].xcom_pull(task_ids="resolve_landing_latest")

    tx_key = keys["tx_key"]
    lbl_key = keys["lbl_key"]

    local_dir = Path(f"/tmp/bronze_staging/{context['ds']}/{context['run_id'].replace(':','_').replace('/','_')}")
    local_dir.mkdir(parents=True, exist_ok=True)

    tx_path = local_dir / "transactions.jsonl"
    lbl_path = local_dir / "fraud_labels.jsonl"

    tx_path.write_bytes(s3.get_object(Bucket=bucket, Key=tx_key)["Body"].read())
    lbl_path.write_bytes(s3.get_object(Bucket=bucket, Key=lbl_key)["Body"].read())

    print(f"Downloaded {tx_key} -> {tx_path}")
    print(f"Downloaded {lbl_key} -> {lbl_path}")

    return {"local_dir": str(local_dir), "tx_path": str(tx_path), "lbl_path": str(lbl_path), "tx_key": tx_key, "lbl_key": lbl_key}


def _extract_run_id_from_key(key: str) -> str:
    marker = "run_id="
    if marker not in key:
        return "unknown_run"
    after = key.split(marker, 1)[1]
    return after.split("/", 1)[0]


def convert_to_bronze_parquet(**context) -> dict:
    """
    Converts downloaded JSONL to local Parquet via Spark (local mode), returns local output dirs + target run_id.
    """
    payload = context["ti"].xcom_pull(task_ids="download_landing_objects")
    tx_path = Path(payload["tx_path"])
    lbl_path = Path(payload["lbl_path"])

    run_id = _extract_run_id_from_key(payload["tx_key"])
    ds = context["ds"]

    out_base = Path(payload["local_dir"]) / "parquet_out"
    out_tx = out_base / "bronze_transactions"
    out_lbl = out_base / "bronze_fraud_labels"

    spark = build_spark(app_name="ccfraud_bronze_ingest")
    try:
        jsonl_to_parquet(tx_path, out_tx, spark=spark)
        jsonl_to_parquet(lbl_path, out_lbl, spark=spark)
    finally:
        stop_spark(spark)

    print(f"Wrote local parquet tx -> {out_tx}")
    print(f"Wrote local parquet labels -> {out_lbl}")

    return {"run_id": run_id, "out_tx": str(out_tx), "out_lbl": str(out_lbl)}


def _upload_directory_to_s3(s3, bucket: str, local_dir: Path, prefix: str) -> int:
    count = 0
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue

        name = path.name
        if name.startswith("_") or name.endswith(".crc"):
            continue  # skip _SUCCESS, _SUCCESS.crc, and crc files

        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}"
        s3.upload_file(str(path), bucket, key)
        count += 1
        
    return count


def upload_bronze_parquet(**context) -> dict:
    """
    Uploads local Parquet directories to MinIO under bronze/.. prefixes.
    Also writes bronze _LATEST pointers for the date.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    ds = context["ds"]

    conv = context["ti"].xcom_pull(task_ids="convert_to_bronze_parquet")
    run_id = conv["run_id"]

    out_tx = Path(conv["out_tx"])
    out_lbl = Path(conv["out_lbl"])

    tx_prefix = f"bronze/transactions/dt={ds}/run_id={run_id}"
    lbl_prefix = f"bronze/fraud_labels/dt={ds}/run_id={run_id}"

    n_tx = _upload_directory_to_s3(s3, bucket, out_tx, tx_prefix)
    n_lbl = _upload_directory_to_s3(s3, bucket, out_lbl, lbl_prefix)

    print(f"Uploaded {n_tx} file(s) to s3://{bucket}/{tx_prefix}/")
    print(f"Uploaded {n_lbl} file(s) to s3://{bucket}/{lbl_prefix}/")

    tx_latest_key = f"bronze/transactions/dt={ds}/_LATEST"
    lbl_latest_key = f"bronze/fraud_labels/dt={ds}/_LATEST"
    s3.put_object(Bucket=bucket, Key=tx_latest_key, Body=(tx_prefix + "\n").encode("utf-8"), ContentType="text/plain")
    s3.put_object(Bucket=bucket, Key=lbl_latest_key, Body=(lbl_prefix + "\n").encode("utf-8"), ContentType="text/plain")

    # Global "current" pointers (stable contract for downstream jobs)
    s3.put_object(
        Bucket=bucket,
        Key="bronze/transactions/_CURRENT",
        Body=(tx_prefix + "\n").encode("utf-8"),
        ContentType="text/plain",
    )
    s3.put_object(
        Bucket=bucket,
        Key="bronze/fraud_labels/_CURRENT",
        Body=(lbl_prefix + "\n").encode("utf-8"),
        ContentType="text/plain",
    )

    return {"tx_prefix": tx_prefix, "lbl_prefix": lbl_prefix}


def verify_bronze_outputs(**context) -> None:
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    ds = context["ds"]

    tx_latest = _read_text_object(s3, bucket, f"bronze/transactions/dt={ds}/_LATEST")
    lbl_latest = _read_text_object(s3, bucket, f"bronze/fraud_labels/dt={ds}/_LATEST")

    for name, prefix in [("transactions", tx_latest), ("fraud_labels", lbl_latest)]:
        if not prefix:
            raise RuntimeError(f"Missing bronze _LATEST for {name} (dt={ds})")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/")
        keys = [c["Key"] for c in resp.get("Contents", [])]
        parquet_keys = [k for k in keys if k.endswith(".parquet")]
        if not parquet_keys:
            raise RuntimeError(f"No parquet files found under s3://{bucket}/{prefix}/")
        print(f"Verified bronze {name}: {len(parquet_keys)} parquet file(s) under s3://{bucket}/{prefix}/")
    
    tx_current = _read_text_object(s3, bucket, "bronze/transactions/_CURRENT")
    lbl_current = _read_text_object(s3, bucket, "bronze/fraud_labels/_CURRENT")

    for name, prefix in [("transactions(_CURRENT)", tx_current), ("fraud_labels(_CURRENT)", lbl_current)]:
        if not prefix:
            raise RuntimeError(f"Missing bronze _CURRENT for {name}")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/")
        keys = [c["Key"] for c in resp.get("Contents", [])]
        parquet_keys = [k for k in keys if k.endswith(".parquet")]
        if not parquet_keys:
            raise RuntimeError(f"No parquet files found under s3://{bucket}/{prefix}/")
        print(f"Verified bronze {name}: {len(parquet_keys)} parquet file(s) under s3://{bucket}/{prefix}/")


default_args = {"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=1)}

with DAG(
    dag_id="02_ingest_to_bronze_v1",
    description="Convert landing JSONL (transactions + fraud_labels) to bronze Parquet using local Spark and upload to MinIO",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase-1", "spark", "bronze"],
) as dag:
    t_resolve = PythonOperator(task_id="resolve_landing_latest", python_callable=resolve_landing_latest)
    t_download = PythonOperator(task_id="download_landing_objects", python_callable=download_landing_objects)
    t_convert = PythonOperator(task_id="convert_to_bronze_parquet", python_callable=convert_to_bronze_parquet)
    t_upload = PythonOperator(task_id="upload_bronze_parquet", python_callable=upload_bronze_parquet)
    t_verify = PythonOperator(task_id="verify_bronze_outputs", python_callable=verify_bronze_outputs)

    t_resolve >> t_download >> t_convert >> t_upload >> t_verify
