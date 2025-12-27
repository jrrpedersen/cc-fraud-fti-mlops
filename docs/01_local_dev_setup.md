# Phase 1 — Local Dev Stack & Smoke Test (Feature Pipeline Foundation)

This project follows an **FTI-style architecture** (Feature → Train → Inference) inspired by *Building Machine Learning Systems with a Feature Store*.
Phase 1 focuses on the **Feature (F)** foundation: getting a reproducible local platform running so we can iterate quickly and later “swap in” cloud services (AWS) with minimal changes.

---

## Why this approach?

### Local-first, cloud-shaped
In Phase 1 we run everything **locally** (fast feedback, easy onboarding), but we choose components that mirror a realistic cloud deployment:

- **MinIO (S3-compatible)** → mirrors AWS S3 for offline data/feature storage  
- **Redis** → mirrors a low-latency online feature store backend (and later can be replaced by AWS DynamoDB/ElastiCache depending on the design)
- **Airflow + Postgres** → mirrors production-style orchestration with durable metadata storage

This gives you a setup that is:
- **Reproducible** (one command to boot the stack)
- **Portable** (no cloud account required to run Phase 1)
- **Production-aligned** (the same “shape” as the eventual AWS/EKS setup)

### Why start with a smoke test?
Before adding Spark jobs, feature definitions, and materialization, we validate the most failure-prone part early:
**service-to-service connectivity**.

If Airflow can reliably reach MinIO and Redis, we have a solid base to:
1. land synthetic transaction data into object storage, then  
2. compute feature datasets, then  
3. materialize online features for serving.

---

## Technologies used in Phase 1

### Orchestration & metadata
- **Apache Airflow**: orchestrates pipelines via DAGs (Phase 1 starts with a smoke-test DAG)
- **PostgreSQL**: Airflow metadata database

### Storage (cloud-shaped local)
- **MinIO**: S3-compatible object storage (offline datasets / lakehouse-style layout)
- **Redis**: online key-value store for low-latency access (used later by Feast as the online store)

### Container runtime
- **Docker Desktop + Docker Compose**: local dev environment bootstrapping

> Coming next (still Phase 1): Spark jobs and Feast feature store repo definitions.

---

## Architecture (Phase 1)

```
+------------------+         +------------------+
|   Airflow UI     |         |  Airflow Scheduler|
|  localhost:8080  |         |  runs DAG tasks   |
+--------+---------+         +---------+---------+
         |                             |
         | runs tasks                  | triggers
         v                             v
    +----+-------------------------------+
    |         Airflow Worker Runtime     |
    | (PythonOperator tasks, future jobs)|
    +----+---------------+---------------+
         |               |
         | HTTP (S3)     | TCP
         v               v
  +------+-----+     +---+------+
  |  MinIO     |     |  Redis   |
  | :9000/:9001|     | :6379    |
  +------------+     +----------+
```

---

## Folder locations (relevant to Phase 1)

- `docker/docker-compose.yml` — local stack definition
- `pipelines/airflow/dags/00_smoke_test.py` — smoke-test DAG
- `docs/` — project documentation (this file lives here)

---

## Getting started

### Prerequisites
- Docker Desktop (running)
- Git
- (Optional) Python 3.10+ for local tooling (not required to run the stack)

### Start the stack
From the repo root:

```powershell
cd docker
docker compose up -d postgres minio redis
docker compose up -d minio-mc
docker compose up -d airflow-init
docker compose up -d airflow-webserver airflow-scheduler
```

### Access the UIs
- **Airflow**: http://localhost:8080  
  - Username: `airflow`  
  - Password: `airflow`
- **MinIO Console**: http://localhost:9001  
  - Username: `minioadmin`  
  - Password: `minioadmin`
- **Redis**: available on `localhost:6379`

---

## Smoke test: validate MinIO + Redis connectivity

### What it does
The smoke-test DAG confirms:
1. Airflow can reach MinIO readiness endpoint  
2. Airflow can connect to Redis and successfully `PING`

### Where it is
- DAG ID: `00_smoke_test_stack`
- File: `pipelines/airflow/dags/00_smoke_test.py`

### How to run
1. Open Airflow UI (http://localhost:8080)
2. Find `00_smoke_test_stack`
3. **Unpause** the DAG
4. Trigger a run (▶)

### Expected output
In task logs you should see:
- `MinIO ready: ... -> 200`
- `Redis ping ok: redis:6379`

If you see this, the stack is correctly wired and ready for Phase 1 feature ingestion.

---

## Notes & common warnings

### Docker Compose warning about `version:`
You may see:

> `the attribute 'version' is obsolete, it will be ignored`

This is safe. Newer Compose no longer requires the `version:` key.  
**Fix**: remove the `version: "3.8"` line from `docker-compose.yml` to silence the warning.

---

## Troubleshooting

### DAG doesn’t show up in Airflow
- Confirm the DAG volume mount exists:
  - `../pipelines/airflow/dags:/opt/airflow/dags`
- Restart Airflow containers:
  ```powershell
  docker compose restart airflow-webserver airflow-scheduler
  ```

### `check_minio` fails
- Confirm MinIO is healthy:
  ```powershell
  docker compose ps
  ```
- Confirm MinIO readiness:
  - http://localhost:9000/minio/health/ready

### `check_redis` fails
- Confirm Redis is healthy:
  ```powershell
  docker compose ps
  ```
- Validate Redis locally:
  ```powershell
  docker exec -it docker-redis-1 redis-cli ping
  ```

---

## What’s next (Phase 1 roadmap)

After the smoke test is green, the Phase 1 build continues:

1. **Synthetic transaction generator** (Python)  
   - produce realistic transaction events (Dowling-inspired)
   - write to MinIO `landing/transactions/` as JSONL/CSV

2. **Bronze ingestion job** (PySpark)  
   - landing → `bronze/transactions/` (Parquet)

3. **Feature computation job** (PySpark)  
   - bronze → `silver/features/` (Parquet feature dataset)

4. **Feast feature repo + materialization**  
   - define entities/feature views
   - materialize recent features into Redis
   - validate online feature retrieval by `card_id`

Once these are in place, Phase 2 will introduce Kubernetes (kind → EKS) and Terraform for infrastructure-as-code.