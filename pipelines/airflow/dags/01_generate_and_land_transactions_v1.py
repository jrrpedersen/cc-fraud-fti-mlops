from __future__ import annotations

import json
import os
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


def read_current_world_id(**context) -> str:
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    obj = s3.get_object(Bucket=bucket, Key="landing/reference/_CURRENT")
    world_id = obj["Body"].read().decode("utf-8").strip()
    if not world_id:
        raise RuntimeError("landing/reference/_CURRENT is empty. Run 00_bootstrap_reference_world first.")
    print(f"Using world_id={world_id}")
    return world_id


def download_world_files(**context) -> dict:
    """
    Download the reference world JSONL files to /tmp for local generation.

    Returns: {"world_id": ..., "local_dir": ..., "datasets": {...}}
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    world_id = context["ti"].xcom_pull(task_ids="read_current_world_id")

    local_dir = Path(f"/tmp/{world_id}_ref")
    local_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for name in ["banks", "merchants", "accounts", "cards"]:
        key = f"landing/reference/world_id={world_id}/{name}.jsonl"
        local_path = local_dir / f"{name}.jsonl"
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        local_path.write_bytes(body)
        datasets[name] = str(local_path)
        print(f"Downloaded {key} -> {local_path}")

    return {"world_id": world_id, "local_dir": str(local_dir), "datasets": datasets}


def generate_transactions_and_labels(**context) -> dict:
    """
    Generate transactions + fraud labels using the active reference world.
    Writes local JSONL files under /tmp and returns paths.
    """
    from data_gen.ccfraud_gen.transactions import generate_transactions_and_labels as gen_tx

    payload = context["ti"].xcom_pull(task_ids="download_world_files")
    world_id = payload["world_id"]
    datasets: dict = payload["datasets"]

    def load_jsonl(path: str) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    world = {name: load_jsonl(path) for name, path in datasets.items()}

    run_id = context["run_id"].replace(":", "_").replace("/", "_")
    ds = context["ds"]
    seed = abs(hash(f"{world_id}:{run_id}")) % (2**31)

    out_dir = Path(f"/tmp/tx_{ds}_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n_transactions": int(os.getenv("N_TRANSACTIONS", "50000")),
        "duration_minutes": int(os.getenv("TX_DURATION_MINUTES", "60")),
        "fraud_rate": float(os.getenv("FRAUD_RATE", "0.0005")),
        "chain_attack_ratio": float(os.getenv("CHAIN_ATTACK_RATIO", "0.9")),
    }

    paths = gen_tx(out_dir=out_dir, world=world, seed=seed, cfg=cfg)

    result = {"world_id": world_id, "paths": {k: str(v) for k, v in paths.items()}}
    print(f"Generated tx/labels for world_id={world_id}: {list(paths.keys())}")
    return result


def upload_tx_and_labels(**context) -> dict:
    """
    Upload transactions + fraud_labels to landing partitions, returning keys.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    payload = context["ti"].xcom_pull(task_ids="generate_transactions_and_labels")
    paths: dict = payload["paths"]

    ds = context["ds"]
    run_id = context["run_id"].replace(":", "_").replace("/", "_")

    keys = {}
    for dataset, local_path in paths.items():
        key = f"landing/{dataset}/dt={ds}/run_id={run_id}/{dataset}.jsonl"
        s3.upload_file(local_path, bucket, key)
        keys[dataset] = key
        print(f"Uploaded {dataset}: {local_path} -> s3://{bucket}/{key}")

    return keys


def write_latest_pointers(**context) -> None:
    """
    Write/overwrite _LATEST pointers for transactions and fraud_labels under their dt partitions.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")

    keys: dict = context["ti"].xcom_pull(task_ids="upload_tx_and_labels")
    ds = context["ds"]

    for dataset, landed_key in keys.items():
        latest_key = f"landing/{dataset}/dt={ds}/_LATEST"
        s3.put_object(
            Bucket=bucket,
            Key=latest_key,
            Body=(landed_key + "\n").encode("utf-8"),
            ContentType="text/plain",
        )
        print(f"Wrote pointer s3://{bucket}/{latest_key} -> {landed_key}")


def verify_latest_pointers(**context) -> None:
    """
    Verify _LATEST exists for both datasets and points to an existing object.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    ds = context["ds"]

    for dataset in ["transactions", "fraud_labels"]:
        latest_key = f"landing/{dataset}/dt={ds}/_LATEST"
        landed_key = s3.get_object(Bucket=bucket, Key=latest_key)["Body"].read().decode("utf-8").strip()
        if not landed_key:
            raise RuntimeError(f"{latest_key} is empty")
        s3.head_object(Bucket=bucket, Key=landed_key)
        print(f"Verified {latest_key} -> {landed_key}")

def validate_referential_integrity(**context) -> None:
    """
    Validates that:
      - transactions reference existing cards/accounts/merchants
      - fraud_labels reference existing transaction IDs and cards
      - required fields exist

    Runs on the locally generated files (before upload).
    """
    payload = context["ti"].xcom_pull(task_ids="generate_transactions_and_labels")
    paths: dict = payload["paths"]

    ref_payload = context["ti"].xcom_pull(task_ids="download_world_files")
    datasets: dict = ref_payload["datasets"]

    required_tx_fields = {
        "t_id","cc_num","account_id","merchant_id","amount","currency","country",
        "ip_address","card_present","channel","category","lat","lon","ts"
    }
    required_label_fields = {"t_id","cc_num","explanation","ts"}

    def load_id_set(path: str, key: str) -> set[str]:
        ids = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                ids.add(obj[key])
        return ids

    card_ids = load_id_set(datasets["cards"], "cc_num")
    account_ids = load_id_set(datasets["accounts"], "account_id")
    merchant_ids = load_id_set(datasets["merchants"], "merchant_id")

    tx_ids: set[str] = set()
    n_tx = 0
    n_bad = 0

    tx_path = paths["transactions"]
    with open(tx_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tx = json.loads(line)
            n_tx += 1

            missing = required_tx_fields - set(tx.keys())
            if missing:
                raise RuntimeError(f"Transaction missing fields {missing}: {tx}")

            if tx["cc_num"] not in card_ids:
                n_bad += 1
                raise RuntimeError(f"Unknown cc_num in tx: {tx['cc_num']} (t_id={tx.get('t_id')})")
            if tx["account_id"] not in account_ids:
                n_bad += 1
                raise RuntimeError(f"Unknown account_id in tx: {tx['account_id']} (t_id={tx.get('t_id')})")
            if tx["merchant_id"] not in merchant_ids:
                n_bad += 1
                raise RuntimeError(f"Unknown merchant_id in tx: {tx['merchant_id']} (t_id={tx.get('t_id')})")

            tx_ids.add(tx["t_id"])

    label_path = paths["fraud_labels"]
    n_labels = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lab = json.loads(line)
            n_labels += 1

            missing = required_label_fields - set(lab.keys())
            if missing:
                raise RuntimeError(f"Fraud label missing fields {missing}: {lab}")

            if lab["t_id"] not in tx_ids:
                raise RuntimeError(f"Fraud label references unknown t_id: {lab['t_id']}")
            if lab["cc_num"] not in card_ids:
                raise RuntimeError(f"Fraud label references unknown cc_num: {lab['cc_num']}")

    print("[integrity] OK")
    print(f"[integrity] transactions={n_tx:,} unique_tx_ids={len(tx_ids):,}")
    print(f"[integrity] fraud_labels={n_labels:,}")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="01_generate_and_land_transactions_v1",
    description="Generate transactions + fraud labels from active reference world and land to MinIO with _LATEST pointers",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase-1", "feature", "transactions", "labels"],
) as dag:
    t_world = PythonOperator(task_id="read_current_world_id", python_callable=read_current_world_id)
    t_dl = PythonOperator(task_id="download_world_files", python_callable=download_world_files)
    t_gen = PythonOperator(task_id="generate_transactions_and_labels", python_callable=generate_transactions_and_labels)
    t_up = PythonOperator(task_id="upload_tx_and_labels", python_callable=upload_tx_and_labels)
    t_latest = PythonOperator(task_id="write_latest_pointers", python_callable=write_latest_pointers)
    t_verify = PythonOperator(task_id="verify_latest_pointers", python_callable=verify_latest_pointers)
    t_validate = PythonOperator(task_id="validate_referential_integrity",python_callable=validate_referential_integrity)   
    t_world >> t_dl >> t_gen >> t_validate >> t_up >> t_latest >> t_verify