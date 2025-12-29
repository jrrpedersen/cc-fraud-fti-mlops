"""
Airflow DAG: Build Silver MIT features from Bronze parquet.

Reads:
- bronze/transactions/dt=<ds>/_LATEST  -> prefix containing parquet parts
- bronze/fraud_labels/dt=<ds>/_LATEST  -> prefix containing parquet parts

Writes:
- silver/training_dataset/dt=<ds>/run_id=<...>/...
- silver/online/card_features/dt=<ds>/run_id=<...>/...
- silver/online/merchant_features/dt=<ds>/run_id=<...>/...

Also writes _LATEST pointers for each, and mirrors the latest online snapshots
to a stable local path for Feast dev:
- /opt/airflow/repo/feature_repo/data/offline/card_features/current/
- /opt/airflow/repo/feature_repo/data/offline/merchant_features/current/
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from pipelines.spark_jobs.build_silver_features import SilverPaths, build_spark, compute_silver, stop_spark

DEFAULT_ARGS = {"owner": "mlops", "retries": 1}

BUCKET = os.environ.get("S3_BUCKET", "mlops")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )


def _read_text_object(s3, bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8").strip()


def _write_text_object(s3, bucket: str, key: str, text: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def _list_parquet_keys(s3, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.endswith(".parquet"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def _download_keys(s3, bucket: str, keys: List[str], local_dir: Path, strip_prefix: str) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for k in keys:
        rel = k[len(strip_prefix):].lstrip("/")
        out = local_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, k, str(out))


def _upload_directory_to_s3(s3, bucket: str, local_dir: Path, prefix: str) -> int:
    count = 0
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        # optional cleanup: skip spark metadata + crc
        if name.startswith("_") or name.endswith(".crc"):
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}"
        s3.upload_file(str(path), bucket, key)
        count += 1
    return count


def _verify_has_parquet(s3, bucket: str, prefix: str) -> None:
    keys = _list_parquet_keys(s3, bucket, prefix)
    if not keys:
        raise RuntimeError(f"No parquet objects found under s3://{bucket}/{prefix}")


def resolve_inputs(ds: str, ti=None, **_):
    s3 = _s3_client()
    tx_latest_key = f"bronze/transactions/dt={ds}/_LATEST"
    lbl_latest_key = f"bronze/fraud_labels/dt={ds}/_LATEST"

    tx_prefix = _read_text_object(s3, BUCKET, tx_latest_key)
    lbl_prefix = _read_text_object(s3, BUCKET, lbl_latest_key)

    ti.xcom_push(key="tx_prefix", value=tx_prefix)
    ti.xcom_push(key="lbl_prefix", value=lbl_prefix)


def build_silver(ds: str, ti=None, **_):
    s3 = _s3_client()
    tx_prefix = ti.xcom_pull(key="tx_prefix")
    lbl_prefix = ti.xcom_pull(key="lbl_prefix")

    # Derive run_id from prefix if it contains run_id=...
    run_id = "unknown_run"
    if "run_id=" in tx_prefix:
        run_id = tx_prefix.split("run_id=")[-1].split("/")[0]

    tmp_root = Path("/tmp/ccfraud_silver") / f"dt={ds}" / f"run_id={run_id}"
    bronze_tx_local = tmp_root / "bronze_tx"
    bronze_lbl_local = tmp_root / "bronze_lbl"

    # Download parquet parts for both datasets
    tx_keys = _list_parquet_keys(s3, BUCKET, tx_prefix.rstrip("/") + "/")
    lbl_keys = _list_parquet_keys(s3, BUCKET, lbl_prefix.rstrip("/") + "/")
    if not tx_keys:
        raise RuntimeError(f"No parquet parts found for tx_prefix={tx_prefix}")
    if not lbl_keys:
        raise RuntimeError(f"No parquet parts found for lbl_prefix={lbl_prefix}")

    _download_keys(s3, BUCKET, tx_keys, bronze_tx_local, strip_prefix=tx_prefix.rstrip("/") + "/")
    _download_keys(s3, BUCKET, lbl_keys, bronze_lbl_local, strip_prefix=lbl_prefix.rstrip("/") + "/")

    out_training = tmp_root / "silver_training_dataset"
    out_card = tmp_root / "silver_card_snapshot"
    out_merchant = tmp_root / "silver_merchant_snapshot"

    spark = build_spark()
    try:
        compute_silver(
            SilverPaths(
                bronze_tx_parquet_dir=bronze_tx_local,
                bronze_labels_parquet_dir=bronze_lbl_local,
                out_training_dataset_dir=out_training,
                out_card_snapshot_dir=out_card,
                out_merchant_snapshot_dir=out_merchant,
            ),
            spark=spark,
        )
    finally:
        stop_spark(spark)

    # Upload to MinIO (versioned prefixes)
    silver_training_prefix = f"silver/training_dataset/dt={ds}/run_id={run_id}"
    silver_card_prefix = f"silver/online/card_features/dt={ds}/run_id={run_id}"
    silver_merchant_prefix = f"silver/online/merchant_features/dt={ds}/run_id={run_id}"

    _upload_directory_to_s3(s3, BUCKET, out_training, silver_training_prefix)
    _upload_directory_to_s3(s3, BUCKET, out_card, silver_card_prefix)
    _upload_directory_to_s3(s3, BUCKET, out_merchant, silver_merchant_prefix)

    # Write _LATEST pointers (versioned)
    _write_text_object(s3, BUCKET, f"silver/training_dataset/dt={ds}/_LATEST", silver_training_prefix)
    _write_text_object(s3, BUCKET, f"silver/online/card_features/dt={ds}/_LATEST", silver_card_prefix)
    _write_text_object(s3, BUCKET, f"silver/online/merchant_features/dt={ds}/_LATEST", silver_merchant_prefix)

    # Verify parquet exists under these prefixes
    _verify_has_parquet(s3, BUCKET, silver_training_prefix)
    _verify_has_parquet(s3, BUCKET, silver_card_prefix)
    _verify_has_parquet(s3, BUCKET, silver_merchant_prefix)

    # Mirror stable "current" snapshots locally for Feast dev
    repo_root = Path("/opt/airflow/repo")
    feast_offline_root = repo_root / "feature_repo" / "data" / "offline"
    card_current = feast_offline_root / "card_features" / "current"
    merchant_current = feast_offline_root / "merchant_features" / "current"

    # overwrite local mirrors
    import shutil
    if card_current.exists():
        shutil.rmtree(card_current)
    if merchant_current.exists():
        shutil.rmtree(merchant_current)

    shutil.copytree(out_card, card_current)
    shutil.copytree(out_merchant, merchant_current)

    # leave breadcrumbs for humans
    (feast_offline_root / "card_features" / "_CURRENT_RUN.txt").write_text(f"dt={ds} run_id={run_id}\n", encoding="utf-8")
    (feast_offline_root / "merchant_features" / "_CURRENT_RUN.txt").write_text(f"dt={ds} run_id={run_id}\n", encoding="utf-8")


with DAG(
    dag_id="03_build_silver_mit_features_v1",
    default_args=DEFAULT_ARGS,
    description="Build Silver MIT features from Bronze parquet and prepare Feast snapshots",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "silver", "spark", "feast"],
) as dag:

    t1 = PythonOperator(
        task_id="resolve_bronze_latest",
        python_callable=resolve_inputs,
    )

    t2 = PythonOperator(
        task_id="build_silver_and_snapshots",
        python_callable=build_silver,
    )

    t1 >> t2
