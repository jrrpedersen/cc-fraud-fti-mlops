# Phase 1 — Step 2: Generate & Land Synthetic Transactions in MinIO

This step implements the **first “real” slice** of the *Feature* pipeline in an FTI (Feature → Train → Inference) architecture:

1. **Generate** synthetic transaction events (JSON Lines / JSONL)
2. **Land** the raw events into S3-compatible object storage (MinIO)
3. **Verify** the objects exist (and optionally preview samples in logs)

The goal is to establish a **reproducible ingestion contract** and a **lakehouse-style layout** that later Spark and Feast components can rely on.

---

## Why we did this next

After validating the local stack with a smoke test (Airflow ↔ MinIO/Redis connectivity), the next milestone is to prove:

- Airflow can run project code (generator)
- Airflow can write data to object storage (MinIO)
- We can consistently locate data by **partitioned paths** (`dt` / `run_id`)

This becomes the backbone for the next steps:
- **Bronze ingestion (Spark):** landing → bronze Parquet  
- **Feature computation (Spark):** bronze → silver/features Parquet  
- **Online materialization (Feast → Redis):** silver/features → online store

---

## Technologies involved in Step 2

- **Apache Airflow**: orchestrates generation, upload, and verification tasks
- **MinIO**: S3-compatible object storage for raw landing data (cloud-shaped local storage)
- **boto3**: S3 API client used by Airflow tasks to upload/list objects
- **Python**: synthetic data generator + Airflow operators

---

## What was added / changed

### 1) Synthetic transaction generator (Python)

**Files**
- `data_gen/generator/__init__.py`
- `data_gen/generator/generate.py`

**Outputs**
- Generates **JSONL** where each line is a JSON object:
  - `event_timestamp` (UTC ISO8601)
  - `transaction_id`
  - `card_id`
  - `merchant_id`
  - `amount`, `currency`
  - `category`, `channel`
  - `lat`, `lon`

**Why JSONL?**
- Great for event-log style data
- Easy to stream/process incrementally
- Human-inspectable with simple tools (`head`, `tail`, download & view)

---

### 2) Airflow DAG: generate → upload → verify

**File**
- `pipelines/airflow/dags/01_generate_and_land.py`

**DAG ID**
- `01_generate_and_land_transactions`

**Task graph**
- `generate_transactions`  
  Creates a file under `/tmp/transactions_<ds>_<run_id>.jsonl` inside the Airflow container.

- *(optional, recommended)* `preview_sample`  
  Prints record count + first 3 lines to logs for fast debugging.

- `upload_to_minio`  
  Uploads the JSONL file to MinIO using the S3 API (`boto3`).

- `verify_objects_exist`  
  Lists objects under the expected prefix and fails loudly if nothing is found.

---

## Data layout (lakehouse-style)

The raw landing data is stored in MinIO bucket `mlops` using a partitioned prefix:

```
s3://mlops/
  landing/
    transactions/
      dt=YYYY-MM-DD/
        run_id=<airflow_run_id>/
          transactions.jsonl
```

**Why partition by `dt` + `run_id`?**
- Makes backfills and replays easy
- Avoids overwriting data
- Allows Spark to filter by date partitions later

---

## How to run & verify

### 1) Make sure the stack is up
From repo root:

```powershell
cd docker
docker compose up -d
```

### 2) Trigger the DAG
1. Open Airflow UI: http://localhost:8080  
2. Find DAG: `01_generate_and_land_transactions`
3. Unpause and **Trigger** a run

### 3) Verify data exists in MinIO
1. Open MinIO Console: http://localhost:9001  
2. Bucket: `mlops`
3. Navigate to:
   - `landing/transactions/dt=<today>/run_id=<...>/transactions.jsonl`

### 4) Sanity-check record contents
MinIO Console may not render JSONL previews reliably. The simplest verification is to **download** the JSONL object and open it locally (VS Code / Notepad++).

You should see **one JSON object per line**.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'data_gen'`
Airflow couldn’t import the generator module. Fix by ensuring project code is on the Python path within containers:

- Mount repo into the Airflow container (recommended): `..:/opt/airflow/repo`
- Set: `PYTHONPATH=/opt/airflow/repo`
- Recreate containers:
  ```powershell
  cd docker
  docker compose down
  docker compose up -d
  ```

### Upload fails with `NoSuchBucket`
Ensure the `mlops` bucket exists. If using the provided `minio-mc` one-shot job, it creates buckets at startup.

---

## What’s next (Step 3)

Next we will implement **Bronze ingestion using Spark**:

- Read: `s3://mlops/landing/transactions/.../*.jsonl`
- Write: `s3://mlops/bronze/transactions/.../*.parquet`

Then we’ll compute **silver feature datasets** from bronze, and later integrate **Feast** for online materialization into Redis.