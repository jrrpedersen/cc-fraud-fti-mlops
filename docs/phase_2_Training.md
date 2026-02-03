# Phase 2 — Training (T in Feature → Train → Inference)

This phase turns our **MIT feature pipeline** into a repeatable **training pipeline** that produces a **versioned model artifact**, logs it to **MLflow Tracking Server**, and registers the model in the **MLflow Model Registry** (with an alias for easy retrieval in inference).

---

## What we have (end of Phase 1)

- **Bronze** data in **MinIO** (transactions + labels)
- **Silver** MIT feature snapshots (card + merchant aggregates)
- **Feast** feature definitions + **Redis** online store materialization (online retrieval working)

---

## What we build in Phase 2

1. **Training dataset builder (DAG 04)**  
   Point‑in‑time correct joins (as‑of) to prevent time travel leakage.

2. **Model training job (DAG 05)**  
   MDTs inside an sklearn Pipeline + **MLflow logging** + **Registry** + **alias**.

3. **Testing + sanity checks**  
   A repeatable way to inspect datasets, validate class balance, and verify MLflow artifacts/registry.

---

# 1) Prevent label leakage (critical)

## Why this matters
If a feature uses information that would not be available at prediction time, we get **label leakage**.  
For merchant features, leakage often happens when you compute rolling fraud counts using the label of the **current** transaction.

## Leakage pattern
If you compute merchant fraud counts like this, you leak the current label into the feature:

```python
# BAD: includes the current row (end=0) -> leakage
w_m = Window.partitionBy("merchant_id").orderBy(F.col("ts_epoch")).rangeBetween
tx = (
    tx
    .withColumn("m_fraud_cnt_1d",  F.sum("label_is_fraud").over(w_m(-86400, 0)))
    .withColumn("m_fraud_cnt_7d",  F.sum("label_is_fraud").over(w_m(-(7*86400), 0)))
    .withColumn("m_fraud_cnt_30d", F.sum("label_is_fraud").over(w_m(-(30*86400), 0)))
)
```

## Minimum viable fix
Exclude the current row by ending the range at **-1** second:

```python
# GOOD: excludes current row (end=-1) -> no leakage
w_m = Window.partitionBy("merchant_id").orderBy(F.col("ts_epoch")).rangeBetween
tx = (
    tx
    .withColumn("m_fraud_cnt_1d",  F.sum("label_is_fraud").over(w_m(-86400, -1)))
    .withColumn("m_fraud_cnt_7d",  F.sum("label_is_fraud").over(w_m(-(7*86400), -1)))
    .withColumn("m_fraud_cnt_30d", F.sum("label_is_fraud").over(w_m(-(30*86400), -1)))
)
```

## Practical note
This fix makes merchant risk features represent “fraud history **before** the current transaction.”

Later (more realistic): treat labels as arriving with delay (e.g., chargebacks) and compute merchant risk from a separate **label‑event stream** keyed by `label_available_ts`. For now, the `-1` fix is correct and keeps the project focused.

---

# 2) Build the training dataset (Airflow DAG 04)

## Goal
Create a labeled dataset for training by joining:

- **Bronze transactions**
- **Bronze labels**
- **Silver MIT features** (card + merchant snapshots)

## Key requirement: point‑in‑time correctness
We must avoid “time travel” joins (using features computed *after* the transaction).  
We use an **as‑of join**:

> For each tx, pick the latest feature row where `feature.event_ts <= tx.event_ts`.

## Output (versioned)
The builder writes Parquet to MinIO:

- `s3a://mlops/silver/training_dataset/dt=<ds>/run_id=<run_id>/`

…and writes pointer files for convenience:

- `mlops/silver/training_dataset/dt=<ds>/_LATEST`  
- `mlops/silver/training_dataset/dt=<ds>/_CURRENT`

These pointers contain the *prefix* (without `s3a://mlops/`) so downstream jobs can safely construct a full URI.

✅ This lets training read the correct dataset even when Airflow run_id changes.

## Expected repository files
- `pipelines/spark_jobs/build_training_dataset.py`
- `pipelines/airflow/dags/04_build_training_dataset_v1.py`

---

# 3) Train a model with MDTs inside the pipeline (Airflow DAG 05)

## MIT vs MDT (Dowling alignment)
- **MITs**: reusable feature engineering outputs (aggregates/windows) — belong in feature pipeline / feature store.
- **MDTs**: model‑dependent transformations (scaling, encoding, learned transforms) — belong **only in training and inference**.

## Why MDTs should NOT go into the feature store
- They are training‑data/model dependent → risk of skew
- They make EDA harder (values no longer “raw”)
- They cause unnecessary write amplification

## Our approach
A baseline classifier using **one sklearn Pipeline**:
- numeric imputation + scaling
- categorical imputation + one‑hot encoding
- baseline model (Logistic Regression)

This guarantees:
- training & inference use identical MDT logic
- minimal skew risk

---

# 4) MLflow as a real service (recommended)

We run **MLflow Tracking Server** as a Docker Compose service.

## Storage layout (local now → cloud later)
- **Metadata store (runs/params/metrics):** Postgres database `mlflow`
- **Artifacts (models, files):** S3‑compatible bucket in MinIO (e.g., `s3://mlflow/artifacts`)

This matches a production‑style deployment:
- Later you can replace MinIO with AWS S3 with minimal code changes.
- You can keep Postgres (or migrate to managed RDS).

## Environment variables (Airflow containers)
Make sure Airflow sees these (e.g., via `x-airflow-common.environment`):

- `MLFLOW_TRACKING_URI=http://mlflow:5000`
- `MLFLOW_EXPERIMENT_NAME=ccfraud-train`
- `MLFLOW_S3_ENDPOINT_URL=http://minio:9000`
- `MLFLOW_REGISTERED_MODEL=ccfraud_baseline`
- `MLFLOW_SET_ALIAS=candidate` (optional but useful)
- `TRAIN_DATA_POINTER=latest` (`latest` or `current`)

---

# 5) How to run Phase 2 end‑to‑end

## Step A — Fix leakage + rebuild Silver MIT features
1) Apply merchant window fix (`rangeBetween(..., -1)`).
2) Re‑run the Silver MIT DAG so snapshots are rebuilt.

## Step B — Build training dataset (DAG 04)
In Airflow UI:
- Trigger `04_build_training_dataset_v1`
- Confirm output written to:
  `silver/training_dataset/dt=<ds>/run_id=<run_id>/`
- Confirm pointer files:
  - `silver/training_dataset/dt=<ds>/_LATEST`
  - `silver/training_dataset/dt=<ds>/_CURRENT`

## Step C — Train + log to MLflow (DAG 05)
In Airflow UI:
- Trigger `05_train_model_mlflow_v1`
- Confirm:
  - new MLflow run appears
  - model artifact is logged
  - model is registered under `ccfraud_baseline`
  - alias (e.g., `candidate`) resolves

---

# 6) Testing & sanity checks (recommended runbook)

This section gives a “confidence loop” so you can verify every stage.

---

## 6.1 Inspect the training dataset (Spark sanity checks)

We have a helper script:
- `tools/inspect_training_dataset.py`

Example (inside the scheduler container):

```bash
cd /opt/airflow/repo
python tools/inspect_training_dataset.py --ds 2026-01-08 --pointer latest
```

The script prints:
- row count / column count
- schema
- event_ts time range
- label distribution
- distinct ids + duplicate checks
- null rates (top columns)
- basic amount stats
- sample fraud and non‑fraud rows
- a “leakage smell test” (e.g., mean merchant fraud count by label)

### What’s “expected”?
- **Fraud rate is small by design** (unless you increase `FRAUD_RATE`)
- Many `card_*` features will be null early in the stream (first tx per card)
- Many `m_*` features will be null if merchant has limited history in the chosen time window

If you want a dev‑friendly dataset:
- increase `TX_DURATION_MINUTES`
- increase `FRAUD_RATE`
- consider generating multiple hours/days of transactions

---

## 6.2 Verify pointer files in MinIO

From Airflow scheduler container:

```bash
python - <<'PY'
import os, boto3
endpoint=os.getenv("S3_ENDPOINT_URL","http://minio:9000")
key=os.getenv("AWS_ACCESS_KEY_ID","minioadmin")
secret=os.getenv("AWS_SECRET_ACCESS_KEY","minioadmin")
s3=boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret)

bucket="mlops"
ds="2026-01-08"
for p in ["_LATEST","_CURRENT"]:
    k=f"silver/training_dataset/dt={ds}/{p}"
    try:
        v=s3.get_object(Bucket=bucket, Key=k)["Body"].read().decode().strip()
        print(k, "->", v)
    except Exception as e:
        print(k, "missing:", e)
PY
```

---

## 6.3 Check MLflow server health from Airflow

```bash
curl -sS -I http://mlflow:5000 | head -n 3
```

Expect `HTTP/1.1 200 OK`.

---

## 6.4 Confirm model registry + alias resolution (inside container)

```bash
python - <<'PY'
import os, mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000"))
c = MlflowClient()
name = os.getenv("MLFLOW_REGISTERED_MODEL","ccfraud_baseline")
alias = os.getenv("MLFLOW_SET_ALIAS","candidate")

print("tracking:", mlflow.get_tracking_uri())
print("registered models:", [m.name for m in c.search_registered_models()])
print("versions:", [(v.version, v.status, v.run_id) for v in c.search_model_versions(f"name='{name}'")][-10:])

mv = c.get_model_version_by_alias(name, alias)
print("alias resolved:", alias, "->", mv.version, mv.run_id)
PY
```

---

## 6.5 Verify artifacts exist in MinIO (models, feature_columns.txt, etc.)

On Windows/PowerShell, use `--entrypoint /bin/sh` to avoid shell issues:

```powershell
cd docker
docker compose run --rm --entrypoint /bin/sh minio-mc -lc "
mc alias set local http://minio:9000 minioadmin minioadmin;
mc ls local/mlflow;
mc ls --recursive local/mlflow/artifacts | head -n 30;
"
```

You should see:
- `.../artifacts/model/model.pkl`
- `.../artifacts/model/MLmodel`
- `.../feature_columns.txt`
- `.../input_example.json` (if logged)

---

## 6.6 Load the model back (smoke test)

Inside a container that can reach MLflow:

```bash
python - <<'PY'
import os, mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000"))
name=os.getenv("MLFLOW_REGISTERED_MODEL","ccfraud_baseline")
alias=os.getenv("MLFLOW_SET_ALIAS","candidate")

model_uri = f"models:/{name}@{alias}"
print("loading:", model_uri)
model = mlflow.pyfunc.load_model(model_uri)
print("loaded model type:", type(model))
PY
```

---

# 7) Troubleshooting

## 7.1 “s3fs missing” or “403 Forbidden” while reading parquet with pandas
Fix: don’t use pandas to read `s3a://...` directly. Instead:
- read with Spark (S3A) → convert to pandas (as we do now)

Also ensure MLflow has S3 endpoint env:
- `MLFLOW_S3_ENDPOINT_URL=http://minio:9000`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

## 7.2 MLflow UI not visible on host
Check docker port mapping:

- container listens on 5000
- host maps to 5001:
  - `127.0.0.1:5001 -> 5000`

Confirm:
```powershell
cd docker
docker compose ps mlflow
```

Then visit:
- `http://127.0.0.1:5001/`

## 7.3 “mc: connect refused localhost:9000”
Inside containers, **localhost** means “that container”, not MinIO. Use:
- `http://minio:9000` from docker network
- `http://127.0.0.1:9000` from host

For `docker compose run minio-mc`, use `http://minio:9000`.

## 7.4 If code changes aren’t picked up by Airflow
If the code is volume‑mounted, usually a re-run is enough. If not:
```powershell
cd docker
docker compose restart airflow-scheduler airflow-webserver
```

If you rebuilt the image:
```powershell
cd docker
docker compose down
docker compose build --no-cache airflow-image
docker compose up -d
```

---

# 8) Notes on naming, versions, and traceability

## “gaudy-stork-892” run names in MLflow
MLflow auto-generates human-friendly run names. This is fine.

For traceability, we rely on:
- tags: `train_data_uri`, `airflow_run_id`, `ds`
- params: dataset pointer, DS, run_id, model hyperparams
- registry: model name + version + alias (e.g., `ccfraud_baseline@candidate`)

## Recommended convention
- **Model name:** `ccfraud_baseline` (or `ccfraud_logreg_v1`)
- **Alias:** `candidate` (latest trained), later `prod` or `champion`
- **Tag in run:** `ds=<ds>`, `pointer=<latest|current>`, `train_data_uri=<s3a://...>`

This gives you the “date-like reference” without encoding dates into filenames.

---

# 9) Next: Phase 3 (Inference)

Once training is stable and logged + registered, we implement:
- inference service that retrieves online features from Feast/Redis
- request logging (ODT inputs)
- durable ingestion so requests are never lost (queue/stream or append-only bronze)
- model selection by registry alias (`models:/ccfraud_baseline@prod`)
- CI/CD and rollout strategy

---

## Quick checklist (Phase 2 done when…)

- [ ] DAG 04 writes `silver/training_dataset/...` plus `_LATEST/_CURRENT`
- [ ] `inspect_training_dataset.py` shows sane schema + no duplicates + plausible label distribution
- [ ] DAG 05 creates MLflow run and logs model artifact
- [ ] MLflow Registry has `ccfraud_baseline` with at least one version
- [ ] Alias `candidate` resolves to the latest model version
- [ ] Model artifacts exist in `s3://mlflow/artifacts/...`
