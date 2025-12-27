# cc-fraud-fti-mlops

FTI-style (Feature → Transform → Inference) MLOps portfolio project for credit card fraud detection.

## Repo structure
- docs/                Project documentation
- docker/              Local dev stack (Compose)
- data_gen/            Synthetic data generator
- pipelines/           Airflow DAGs + Spark jobs
- feature_repo/        Feast feature store definitions
- services/            Online services (later)
