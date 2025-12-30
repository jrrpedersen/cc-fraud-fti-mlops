from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator


def _s3_client():
    """
    Create and return a boto3 S3 client configured via environment variables.

    This helper supports both AWS S3 and S3-compatible object stores (e.g., MinIO).
    Connection settings are read from the environment, with MinIO-friendly defaults:

      - S3_ENDPOINT_URL (default: "http://minio:9000")
      - AWS_DEFAULT_REGION (default: "us-east-1")
      - AWS_ACCESS_KEY_ID (default: "minioadmin")
      - AWS_SECRET_ACCESS_KEY (default: "minioadmin")

    Returns:
        botocore.client.S3: A configured S3 client usable for operations such as
        get_object, put_object, and list_objects_v2.
    """
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
    """
    Read and return the current reference "world_id" pointer from the landing bucket.

    This function fetches the object at key: "landing/reference/_CURRENT" from the
    bucket specified by LANDING_BUCKET (default: "mlops"). The object is expected to
    contain a UTF-8 encoded string identifying the current reference dataset/version
    (the "world_id"), typically written by a bootstrap step.

    The object content is read from the StreamingBody returned by boto3's get_object,
    decoded as UTF-8, and stripped of whitespace/newlines. If the resulting world_id
    is empty, a RuntimeError is raised to indicate the reference world has not been
    initialized.

    Args:
        **context: Airflow task context (unused). Included for compatibility with
        PythonOperator/TaskFlow callables that pass context as keyword arguments.

    Returns:
        str: The current world_id string read from "landing/reference/_CURRENT".

    Raises:
        RuntimeError: If "landing/reference/_CURRENT" exists but is empty.
        botocore.exceptions.ClientError: If the bucket/key does not exist or the
            request is unauthorized.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    # get_object returns a Python dict with metadata plus the content stream
    obj = s3.get_object(Bucket=bucket, Key="landing/reference/_CURRENT")
    # obj["Body"] is a streaming file-like object (botocore calls it a StreamingBody)
    # .read() gives the raw bytes; .decode("utf-8") turns bytes into a string; .strip() removes whitespace/newlines (if the file ends with \n)
    world_id = obj["Body"].read().decode("utf-8").strip()
    if not world_id:
        raise RuntimeError("landing/reference/_CURRENT is empty. Run 00_bootstrap_reference_world first.")
    print(f"Using world_id={world_id}")
    return world_id


def download_world_files(**context) -> dict:
    """
    Download the current reference "world" datasets from S3/MinIO to the local filesystem.

    This task:
      1) Creates an S3-compatible client (via `_s3_client()`).
      2) Pulls the current `world_id` from XCom (produced by the upstream task
         `read_current_world_id`).
      3) Downloads a fixed set of JSONL reference datasets for that `world_id`
         from the landing bucket into a local directory under `/tmp`.
      4) Returns a mapping describing the downloaded files for downstream tasks.

    Expected object layout in the bucket:
        landing/reference/world_id=<world_id>/banks.jsonl
        landing/reference/world_id=<world_id>/merchants.jsonl
        landing/reference/world_id=<world_id>/accounts.jsonl
        landing/reference/world_id=<world_id>/cards.jsonl

    Local output layout:
        /tmp/<world_id>_ref/banks.jsonl
        /tmp/<world_id>_ref/merchants.jsonl
        /tmp/<world_id>_ref/accounts.jsonl
        /tmp/<world_id>_ref/cards.jsonl

    Notes / assumptions:
      - The downloaded files are written to `/tmp` *inside the running Airflow worker/container*.
        This pattern works well when downstream tasks run on the same worker/container (e.g.,
        local Docker / LocalExecutor). In distributed executors, `/tmp` is usually not shared.
      - Files are currently downloaded by calling `get_object(...)[\"Body\"].read()`, which
        reads the entire object into memory before writing to disk. This is fine for small
        reference files but can be memory-heavy for large datasets; consider streaming to disk
        for large objects.

    Args:
        **context: Airflow task context. Uses `context["ti"]` to pull `world_id` from XCom.

    Returns:
        dict: A dictionary of the form:
            {
              "world_id": <str>,
              "local_dir": <str>,
              "datasets": {
                  "banks": <local path>,
                  "merchants": <local path>,
                  "accounts": <local path>,
                  "cards": <local path>
              }
            }

    Raises:
        KeyError: If expected Airflow context keys are missing (e.g., "ti").
        botocore.exceptions.ClientError: If a bucket/key is missing or access is denied.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    # XCom values are stored scoped to a specific DAG run and task
    # context["ti"] is the TaskInstance object for the currently running task
    world_id = context["ti"].xcom_pull(task_ids="read_current_world_id")

    # Create a local directory under /tmp
    local_dir = Path(f"/tmp/{world_id}_ref")
    # parents=True: create any missing parent dirs; exist_ok=True: don’t error if it already exists
    local_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for name in ["banks", "merchants", "accounts", "cards"]:
        # Construct the S3 key and local path
        key = f"landing/reference/world_id={world_id}/{name}.jsonl"
        local_path = local_dir / f"{name}.jsonl"
        # Downloads the object using get_object; ["Body"].read() reads the entire file into memory as bytes
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        # Write the bytes to the local file
        local_path.write_bytes(body)
        # Store the local path as a string in the datasets dict
        datasets[name] = str(local_path)
        print(f"Downloaded {key} -> {local_path}")
    # Return metadata for downstream tasks
    return {"world_id": world_id, "local_dir": str(local_dir), "datasets": datasets}


def generate_transactions_and_labels(**context) -> dict:
    """
    Generate synthetic transactions and fraud labels using the active reference world.

    This task pulls the output of `download_world_files` from XCom (which contains the
    active `world_id` and local paths to reference JSONL files), loads those JSONL files
    into an in-memory `world` dictionary, and then calls the project generator
    (`data_gen.ccfraud_gen.transactions.generate_transactions_and_labels`) to produce
    transactions and corresponding label outputs.

    Generated datasets are written as JSONL files under a run-scoped directory in `/tmp`
    (e.g., `/tmp/tx_<ds>_<run_id>/`). The task returns metadata only (world_id + output
    file paths) for downstream tasks, rather than the full data payload.

    Args:
        **context: Airflow task context. Uses:
            - context["ti"] to pull the upstream payload from XCom.
            - context["ds"] and context["run_id"] for output naming and seed derivation.

    Returns:
        dict: A dictionary of the form:
            {
              "world_id": <str>,
              "paths": {<dataset_name>: <local_path_str>, ...}
            }

    Raises:
        KeyError: If expected context keys or payload fields are missing.
        FileNotFoundError / OSError: If expected local reference files are missing or unreadable.
        json.JSONDecodeError: If a reference JSONL line cannot be parsed.
        botocore.exceptions.ClientError or RuntimeError: Propagated from upstream steps or the
            generator if generation fails.
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

    # Build the “world” object (in-memory reference data)
    # Scalability depends on size of reference files, container memory
    world = {name: load_jsonl(path) for name, path in datasets.items()}
    """ world = {
    "banks": [...list of bank dicts...],
    "merchants": [...],
    "accounts": [...],
    "cards": [...],
    } """

    run_id = context["run_id"].replace(":", "_").replace("/", "_")
    ds = context["ds"]
    seed = abs(hash(f"{world_id}:{run_id}")) % (2**31)
    # TODO: hash() may differ across container restarts unless PYTHONHASHSEED is fixed. 
    # Consider using hashlib for stable hashing:
    """ import hashlib

    def stable_seed(s: str) -> int:
        
        Derive a stable 32-bit integer seed from an input string.

        Uses a cryptographic hash (SHA-256) to produce a deterministic seed that is
        consistent across Python processes, container restarts, machines, and Python
        versions (unlike Python's built-in `hash()` which is randomized by default).

        Args:
            s: Input string to hash (e.g., f"{world_id}:{run_id}").

        Returns:
            An integer in the range [0, 2**31 - 1] suitable for seeding RNGs.

        digest = hashlib.sha256(s.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**31)

    seed = stable_seed(f"{world_id}:{run_id}") """

    out_dir = Path(f"/tmp/tx_{ds}_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n_transactions": int(os.getenv("N_TRANSACTIONS", "50000")),
        "duration_minutes": int(os.getenv("TX_DURATION_MINUTES", "60")),
        "fraud_rate": float(os.getenv("FRAUD_RATE", "0.0005")),
        "chain_attack_ratio": float(os.getenv("CHAIN_ATTACK_RATIO", "0.9")),

        "start_utc": f"{ds}T00:00:00Z",
        
    }

    paths = gen_tx(out_dir=out_dir, world=world, seed=seed, cfg=cfg)
    """ paths = {
    "transactions": Path("/tmp/tx_.../transactions.jsonl"),
    "labels": Path("/tmp/tx_.../labels.jsonl"),
    } """

    # Return paths (as strings) via XCom
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

    # Use Airflow’s run metadata to build partitions
    ds = context["ds"]
    run_id = context["run_id"].replace(":", "_").replace("/", "_")

    # Upload each dataset to its landing partition
    keys = {}
    for dataset, local_path in paths.items():
        key = f"landing/{dataset}/dt={ds}/run_id={run_id}/{dataset}.jsonl"
        s3.upload_file(local_path, bucket, key)
        keys[dataset] = key
        print(f"Uploaded {dataset}: {local_path} -> s3://{bucket}/{key}")

    """ This assumes the local files referenced by local_path are accessible in the environment where this 
    task runs (same container/worker or shared volume). In a local Docker setup that’s usually fine; 
    in distributed executors it can break unless you share storage or generate+upload in the same task. """
    return keys


def write_latest_pointers(**context) -> None:
    """
    Write/overwrite per-date `_LATEST` pointer objects for each landed dataset.

    This task pulls the dataset-to-key mapping produced by the upstream task
    `upload_tx_and_labels` (via XCom). For each dataset (e.g., "transactions",
    "fraud_labels"), it writes a small text object named `_LATEST` under that
    dataset's date partition:

        landing/<dataset>/dt=<ds>/_LATEST

    The contents of `_LATEST` is the fully qualified landed object key for the
    most recent run for that date (followed by a newline), e.g.:

        landing/<dataset>/dt=<ds>/run_id=<run_id>/<dataset>.jsonl

    This pattern allows downstream consumers to find "the latest" file for a
    given date partition without listing all `run_id=...` prefixes.

    Args:
        **context: Airflow task context. Uses:
            - context["ti"] to pull the upstream mapping from XCom.
            - context["ds"] to determine the date partition.

    Returns:
        None. (The pointer objects are written to S3/MinIO; no data is returned.)

    Raises:
        KeyError: If expected context keys or XCom payload are missing.
        botocore.exceptions.ClientError: If the put_object request fails (e.g.,
            permission issues, missing bucket, connectivity problems).
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
    Validate that per-date `_LATEST` pointers exist and reference real landed objects.

    For each expected dataset ("transactions" and "fraud_labels"), this task:
      1) Reads the pointer object at:
            landing/<dataset>/dt=<ds>/_LATEST
         and interprets its UTF-8 contents as the landed object key.
      2) Fails if the pointer file is empty/whitespace.
      3) Calls `head_object` on the landed key to confirm the referenced object exists
         (without downloading the full dataset).

    This provides an end-of-pipeline sanity check that the "latest" pointers were
    written and that they correctly point to uploaded outputs for the current date
    partition.

    Args:
        **context: Airflow task context. Uses `context["ds"]` to select the date
        partition to verify.

    Returns:
        None.

    Raises:
        RuntimeError: If a `_LATEST` pointer exists but is empty.
        botocore.exceptions.ClientError: If `_LATEST` does not exist, or if the landed
            object referenced by `_LATEST` does not exist (e.g., NoSuchKey / 404), or
            if access is denied.
        KeyError: If expected context keys are missing.
    """
    s3 = _s3_client()
    bucket = os.getenv("LANDING_BUCKET", "mlops")
    ds = context["ds"]

    for dataset in ["transactions", "fraud_labels"]:
        latest_key = f"landing/{dataset}/dt={ds}/_LATEST"
        landed_key = s3.get_object(Bucket=bucket, Key=latest_key)["Body"].read().decode("utf-8").strip()
        if not landed_key:
            raise RuntimeError(f"{latest_key} is empty")
        # head_object asks S3 for the object’s metadata without downloading the whole file
        s3.head_object(Bucket=bucket, Key=landed_key)
        print(f"Verified {latest_key} -> {landed_key}")

def validate_referential_integrity(**context) -> None:
    """
    Validate schema, referential integrity, and basic temporal consistency for locally generated outputs.

    This task runs on local JSONL files produced by `generate_transactions_and_labels` (before upload)
    and checks that the generated transactions and fraud labels are consistent with the active
    reference world downloaded by `download_world_files`.

    Specifically, it enforces:
      - Required fields exist in each transaction and label record.
      - Transactions reference valid entities in the reference world:
          * tx["cc_num"] exists in cards
          * tx["account_id"] exists in accounts
          * tx["merchant_id"] exists in merchants
      - Transaction IDs (t_id) are unique.
      - Fraud labels reference existing transactions and valid cards:
          * lab["t_id"] exists in the transactions file
          * lab["cc_num"] exists in cards
          * lab["cc_num"] matches the cc_num of the referenced transaction t_id
          * no duplicate labels for the same t_id
      - Timestamps are parseable ISO-8601 strings with timezone info, and label timestamps
        are not earlier than their referenced transaction timestamps.

    The task fails fast by raising RuntimeError on the first detected violation.

    Args:
        **context: Airflow task context. Uses `context["ti"]` to pull upstream payloads from XCom.

    Returns:
        None.

    Raises:
        RuntimeError: If any validation check fails (missing required fields, invalid references,
            duplicate IDs, unparseable/naive timestamps, or label timestamps earlier than tx timestamps).
        KeyError: If expected context keys or XCom payload fields are missing.
        OSError / FileNotFoundError: If expected local JSONL files cannot be read.
        json.JSONDecodeError: If an input JSONL line cannot be parsed.
    """
    # Pull the generated output file paths from XCom
    payload = context["ti"].xcom_pull(task_ids="generate_transactions_and_labels")
    paths: dict = payload["paths"]

    # Pull the reference world file paths from XCom
    ref_payload = context["ti"].xcom_pull(task_ids="download_world_files")
    datasets: dict = ref_payload["datasets"]

    # These are sets of keys that must exist in each JSON object
    required_tx_fields = {
        "t_id","cc_num","account_id","merchant_id","amount","currency","country",
        "ip_address","card_present","channel","category","lat","lon","ts"
    }
    required_label_fields = {"t_id","cc_num","explanation","ts"}

    # Build “lookup sets” from reference files
    # Using a set makes those checks very fast (O(1) average)
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
    tx_ts_by_id: dict[str, datetime] = {}
    tx_cc_by_id: dict[str, str] = {}
    n_tx = 0

    # Validate transactions file
    tx_path = paths["transactions"]
    with open(tx_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tx = json.loads(line)
            n_tx += 1

            # Required fields exist
            missing = required_tx_fields - set(tx.keys())
            if missing:
                raise RuntimeError(f"Transaction missing fields {missing}: {tx}")
            
            # Populate transaction ID set
            t_id = tx["t_id"]
            tx_ids.add(t_id)
            # Store per-transaction reference values for later label checks
            tx_cc_by_id[t_id] = tx["cc_num"]
            
            # amount should be numeric and non-negative
            amt = tx["amount"]
            if not isinstance(amt, (int, float)) or amt < 0:
                raise RuntimeError(f"Invalid amount={amt} (t_id={tx.get('t_id')})")

            # lat/lon bounds
            lat, lon = tx["lat"], tx["lon"]
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise RuntimeError(f"Invalid lat/lon=({lat},{lon}) (t_id={tx.get('t_id')})")

            # Referential integrity vs reference world
            if tx["cc_num"] not in card_ids:
                n_bad += 1
                raise RuntimeError(f"Unknown cc_num in tx: {tx['cc_num']} (t_id={tx.get('t_id')})")
            if tx["account_id"] not in account_ids:
                n_bad += 1
                raise RuntimeError(f"Unknown account_id in tx: {tx['account_id']} (t_id={tx.get('t_id')})")
            if tx["merchant_id"] not in merchant_ids:
                n_bad += 1
                raise RuntimeError(f"Unknown merchant_id in tx: {tx['merchant_id']} (t_id={tx.get('t_id')})")
            
            # Timestamp parseability + rough time-window sanity
            ts = tx["ts"]
            try:
                # handles "...Z" by converting to "+00:00"
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))

            except Exception as e:
                raise RuntimeError(f"Unparseable tx ts={ts} (t_id={tx.get('t_id')}): {e}")

            # optional: ensure UTC-aware
            if dt.tzinfo is None:
                raise RuntimeError(f"Naive (timezone-less) ts={ts} (t_id={tx.get('t_id')})")
            
            tx_ts_by_id[tx["t_id"]] = dt


    if len(tx_ids) != n_tx:
        raise RuntimeError(
            f"Duplicate t_id detected: transactions={n_tx:,} unique_tx_ids={len(tx_ids):,}"
        )

    # Validate fraud labels file
    labeled_tx_ids: set[str] = set()
    label_path = paths["fraud_labels"]
    n_labels = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lab = json.loads(line)
            n_labels += 1

            # Required fields exist
            missing = required_label_fields - set(lab.keys())
            if missing:
                raise RuntimeError(f"Fraud label missing fields {missing}: {lab}")
            # Labels reference real transactions and cards
            if lab["t_id"] not in tx_ids:
                raise RuntimeError(f"Fraud label references unknown t_id: {lab['t_id']}")
            
            expected_cc = tx_cc_by_id.get(lab["t_id"])
            if expected_cc is None:
                raise RuntimeError(f"Missing tx cc_num for t_id={lab['t_id']}")
            if lab["cc_num"] != expected_cc:
                raise RuntimeError(
                    f"Fraud label cc_num mismatch for t_id={lab['t_id']}: "
                    f"label_cc_num={lab['cc_num']} tx_cc_num={expected_cc}"
                )

            if lab["cc_num"] not in card_ids:
                raise RuntimeError(f"Fraud label references unknown cc_num: {lab['cc_num']}")
            # Prevent duplicate labels for the same transaction
            if lab["t_id"] in labeled_tx_ids:
                raise RuntimeError(f"Duplicate fraud label for t_id={lab['t_id']}")
            labeled_tx_ids.add(lab["t_id"])
            # Parse label timestamp
            lab_ts = lab["ts"]
            try:
                lab_dt = datetime.fromisoformat(lab_ts.replace("Z", "+00:00"))
            except Exception as e:
                raise RuntimeError(f"Unparseable label ts={lab_ts} (t_id={lab.get('t_id')}): {e}")

            if lab_dt.tzinfo is None:
                raise RuntimeError(f"Naive (timezone-less) label ts={lab_ts} (t_id={lab.get('t_id')})")

            # Compare label timestamp vs transaction timestamp
            tx_dt = tx_ts_by_id.get(lab["t_id"])
            if tx_dt is None:
                # Shouldn't happen if tx_ids check passed, but gives clearer error if it does.
                raise RuntimeError(f"Missing tx timestamp for t_id={lab['t_id']}")

            if lab_dt < tx_dt:
                raise RuntimeError(
                    f"Label ts before tx ts for t_id={lab['t_id']}: label={lab_dt.isoformat()} tx={tx_dt.isoformat()}"
                )

    # Print summary if everything passes
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