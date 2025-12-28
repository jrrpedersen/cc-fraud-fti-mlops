# Phase 1 — Step 2 (v1): Bronze Ingestion with Spark (JSONL → Parquet)

This step continues Phase 1 (Feature side) from the point where we already have:

- A reproducible local platform (Airflow + MinIO + Redis)
- A realistic synthetic data world (reference entities)
- A transaction stream generator with fraud mechanisms
- Versioned landing layout with `_CURRENT` + `_LATEST` pointers
- Referential integrity checks

Now we add a **Bronze layer** and a **Spark ingestion pipeline**.

---

## Motivation: Why a Bronze layer?

In production data platforms, you typically separate storage into layers:

- **Landing / Raw**: append-only, unmodified payloads (here: JSONL in MinIO/S3)
- **Bronze**: standardized storage format + basic schema normalization (here: Parquet)
- **Silver/Gold** (later): feature-ready tables, aggregates, and ML training datasets

Bronze is valuable because it:

- dramatically improves read performance (Parquet is columnar and compressed)
- gives you a consistent schema boundary for downstream pipelines
- provides a clean foundation for Spark-based feature engineering

In this project, **Landing remains immutable**. Bronze is derived and can be rebuilt at any time.

---

## What we built

### 1) Custom Airflow image (Java + PySpark)

Spark (PySpark) requires Java. To keep the project reproducible on a laptop, we run Spark in **local mode inside the Airflow container**.

We created:

- `docker/airflow/Dockerfile`

Key behavior:
- installs Java 17 (OpenJDK)
- installs `pyspark==3.5.3`
- installs a few common runtime libs (boto3/requests/redis)

### 2) Spark job: JSONL → Parquet

We added:

- `pipelines/spark_jobs/ingest_to_bronze.py`

It provides utilities to:
- build a local SparkSession (`local[*]`)
- read JSONL
- convert timestamps
- write Parquet outputs

### 3) Airflow DAG: `02_ingest_to_bronze_v1`

We added:

- `pipelines/airflow/dags/02_ingest_to_bronze_v1.py`

This DAG:

1. Reads landing pointers for the run date:
   - `landing/transactions/dt=<ds>/_LATEST`
   - `landing/fraud_labels/dt=<ds>/_LATEST`

2. Downloads the referenced JSONL objects locally under `/tmp`

3. Runs Spark local mode to write Parquet files

4. Uploads Parquet files into Bronze prefixes:
   - `bronze/transactions/dt=<ds>/run_id=<run_id>/...`
   - `bronze/fraud_labels/dt=<ds>/run_id=<run_id>/...`

5. Writes Bronze pointers:
   - `bronze/transactions/dt=<ds>/_LATEST`
   - `bronze/fraud_labels/dt=<ds>/_LATEST`

6. Verifies that `.parquet` objects exist under the Bronze prefix

---

## Expected MinIO layout

After running the DAG successfully, you should see something like:

### Transactions
```
bronze/transactions/dt=2025-12-28/run_id=manual__2025-12-28T16_02_58.315800+00_00/part-*.snappy.parquet
bronze/transactions/dt=2025-12-28/_LATEST
```

### Fraud labels
```
bronze/fraud_labels/dt=2025-12-28/run_id=manual__2025-12-28T16_02_58.315800+00_00/part-*.snappy.parquet
bronze/fraud_labels/dt=2025-12-28/_LATEST
```

**Note:** Spark often writes `_SUCCESS` and `*.crc` files. They are harmless.  
(We can optionally filter these from uploads later.)

---

## How to run (local)

1) Start the stack:
```powershell
cd docker
docker compose up -d
```

2) In Airflow UI, trigger these DAGs in order:
- `00_bootstrap_reference_world` (one-time per “world”)
- `01_generate_and_land_transactions_v1` (creates landing `_LATEST`)
- `02_ingest_to_bronze_v1` (creates bronze parquet + bronze `_LATEST`)

---

## Quick validation commands (debugging helpers)

### Check running containers and images
```powershell
docker ps --format "table {{.Names}}\t{{.Image}}"
docker compose ps
```

You should see Airflow containers running **your custom image** (e.g. `cc-fraud-airflow:local`).

### Inspect Airflow import errors
```powershell
docker exec -it docker-airflow-scheduler-1 bash -lc "airflow dags list-import-errors"
```

### Verify Java and PySpark are present inside the Airflow scheduler
```powershell
docker exec -it docker-airflow-scheduler-1 bash -lc "java -version"
docker exec -it docker-airflow-scheduler-1 bash -lc "python -c 'import pyspark; print(pyspark.__version__)'"
```

### Airflow logs (last 200 lines)
```powershell
docker compose logs --tail 200 airflow-scheduler
docker compose logs --tail 200 airflow-webserver
```

### Force a clean rebuild (avoid Docker cache issues)
This is useful when you change dependencies in the Dockerfile and want to ensure
you are not using a previously cached layer.

```powershell
cd docker
docker compose down
docker image rm cc-fraud-airflow:local -f
docker compose build --no-cache airflow-image
docker compose up -d
```

### Confirm Bronze pointers
If you want to see what Bronze `_LATEST` points to, open the `_LATEST` object in MinIO.
It should contain a prefix like:

```
bronze/fraud_labels/dt=2025-12-28/run_id=manual__2025-12-28T16_02_58.315800+00_00
```

---

## What this enables next

With Bronze in place, the next natural work is:

- **Silver features**: Spark jobs that join and aggregate transactions and reference entities
- **Feast**: materialize feature views (offline Parquet + online Redis)
- **Training pipeline**: train a model with MLflow tracking + model registry
- **Inference pipeline**: FastAPI service reading online features, deployed to Kubernetes

This step completes the “raw → bronze” backbone of the Feature pipeline.
