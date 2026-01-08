# Phase 2 — Training (T in Feature → Train → Inference)

This phase turns our **MIT feature pipeline** into a repeatable **training pipeline** that produces a versioned model artifact, logs it to MLflow, and sets us up for deployment in the Inference phase.

**What we already have (end of Phase 1):**
- Bronze data in MinIO (transactions + labels)
- Silver **MIT** feature snapshots (card + merchant aggregates)
- Feast feature definitions + Redis online store materialization (working online retrieval)

**What we build now (Phase 2):**
1. **Training dataset builder** (point-in-time correct joins)
2. **Model training job** (MDTs inside a pipeline) + **MLflow logging**
3. Airflow DAGs to orchestrate both steps


---

## 1) Prevent label leakage (critical)

### Why this matters
If a feature uses information that would not be available at prediction time, we get **label leakage**.
For merchant features, this often happens when you compute rolling fraud counts using the label of the *current* transaction.

### The leakage pattern
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

### Minimum viable fix
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

### Practical note
This fix makes the merchant risk features represent “fraud history **before** the current transaction”.

**Later (more realistic):** treat labels as arriving with delay (e.g., chargeback) and compute merchant risk from a separate **label-event stream** keyed by a `label_available_ts`. For now, the `-1` window-end fix is correct and keeps the project focused.


---

## 2) Build a training dataset (Airflow DAG 04)

### Goal
Create a labeled dataset for model training by joining:
- **Bronze transactions**
- **Bronze labels**
- **Silver MIT features** (card + merchant)

### Key requirement: point-in-time correctness
We must avoid “time travel” joins (using features computed *after* the transaction).

We use an **as-of join**:
> for each transaction, pick the latest feature row where `feature.event_ts <= tx.event_ts`.

### Output location (versioned)
The builder writes a parquet dataset to MinIO:
- `s3a://mlops/silver/training_dataset/dt=<ds>/run_id=<run_id>/`

### Expected repository files
- `pipelines/spark_jobs/build_training_dataset.py`
- `pipelines/airflow/dags/04_build_training_dataset_v1.py`


---

## 3) Train a model with MDTs inside the pipeline (Airflow DAG 05)

### MIT vs MDT (Dowling alignment)
- **MITs**: reusable feature engineering outputs (aggregates/windows) — belong in feature pipeline / feature store.
- **MDTs**: model-dependent transformations (scaling, encoding, learned transforms) — belong **only in training and inference**.

### Why MDTs must NOT go into the feature store
- They are model/training-data dependent (risk of skew)
- They make EDA harder
- They can cause expensive write amplification if stored

### Our approach
We train a baseline classifier with a **single sklearn Pipeline** that includes:
- numeric imputation + scaling
- categorical imputation + one-hot encoding
- baseline model (logistic regression to start)

This guarantees:
- **training and inference use identical MDT logic**
- minimal skew risk

### MLflow logging (local, for now)
We log:
- metrics: ROC-AUC, PR-AUC
- model artifact
- feature column list

Default: MLflow local file store (inside the repo):
- `MLFLOW_TRACKING_URI=file:/opt/airflow/repo/mlruns` (when running inside Airflow container)


---

## 4) Dependencies (Airflow image)

If training runs inside Airflow, ensure the Airflow image includes:

- `pandas`
- `pyarrow`
- `scikit-learn`
- `mlflow`

(Plus what you already have: Java + pyspark, boto3, etc.)

**Suggested pins (example):**
- pandas==2.2.3
- pyarrow==18.1.0
- scikit-learn==1.5.2
- mlflow==2.17.2


---

## 5) How to run Phase 2 end-to-end

### Step A — Fix leakage + rebuild Silver MIT features
1) Apply the merchant window change (`rangeBetween(..., -1)`).
2) Re-run your Silver MIT DAG so snapshots are rebuilt.

### Step B — Build training dataset (DAG 04)
In Airflow UI:
- Trigger `04_build_training_dataset_v1`
- Confirm output written to:
  `silver/training_dataset/dt=<ds>/run_id=<run_id>/`

### Step C — Train + log to MLflow (DAG 05)
In Airflow UI:
- Trigger `05_train_model_mlflow_v1`
- Confirm a new MLflow run appears in `mlruns/`.

If you want to look at the artifacts quickly, you can open the `mlruns` folder locally in your repo.


---

## 6) Sanity checks and troubleshooting

### Verify training dataset exists in MinIO
Use MinIO console (`http://localhost:9001`) and browse:
- `mlops/silver/training_dataset/...`

Or via MinIO client:
```powershell
cd docker
docker compose run --rm minio-mc sh -lc "mc ls --recursive local/mlops/silver/training_dataset | tail -n 50"
```

### Verify class balance
Fraud is rare. Expect a strong imbalance.
If your realized fraud rate is too low for development, temporarily increase `FRAUD_RATE` to validate the pipeline.

### If MLflow runs aren’t visible
- Confirm `MLFLOW_TRACKING_URI` points to a writable path.
- For Airflow container runs, a safe default is:
  `file:/opt/airflow/repo/mlruns`

### If you see import errors in Airflow
Check inside scheduler container:
```powershell
cd docker
docker exec -it docker-airflow-scheduler-1 bash -lc "airflow dags list-import-errors"
```

### If you changed Dockerfile dependencies
Rebuild without cache:
```powershell
cd docker
docker compose down
docker image rm cc-fraud-airflow:local -f
docker compose build --no-cache airflow-image
docker compose up -d
```


---

## Next: Phase 3 (Inference)
Once training is stable and logged to MLflow, we’ll implement:
- an inference service that fetches online features from Feast/Redis
- request logging (ODT inputs)
- durable event ingestion so requests are never lost (bronze write via queue/stream)
- CI/CD to deploy model versions