"""
Baseline training + MLflow logging.

Reads the training dataset from MinIO (S3A) using Spark, converts to pandas,
then trains a sklearn pipeline and logs to MLflow.

MDTs live here (inside sklearn Pipeline) to avoid train/serve skew.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
@dataclass
class TrainConfig:
    parquet_path: str
    label_col: str = "label_is_fraud"
    random_state: int = 42
    test_size: float = 0.2

    # optional safety knobs
    max_rows: int | None = None
    spark_partitions: int = 64

    # provenance (new)
    ds: Optional[str] = None
    pointer_name: Optional[str] = None          # "latest" or "current"
    pointer_key: Optional[str] = None           # e.g. silver/.../_LATEST
    resolved_prefix: Optional[str] = None       # the actual prefix read from pointer

def _build_spark(app_name: str):
    """Spark session configured for MinIO via S3A."""
    from pyspark.sql import SparkSession

    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    # IMPORTANT: compatible with Spark 3.5.x + Hadoop 3.3.x
    spark_packages = os.getenv(
        "SPARK_JARS_PACKAGES",
        "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")

        # jars for s3a
        .config("spark.jars.packages", spark_packages)
        .config("spark.jars.ivy", "/tmp/.ivy2")  # avoid permission issues

        # s3a configs for MinIO
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _read_training_parquet_to_pandas(cfg: TrainConfig) -> pd.DataFrame:
    """Read parquet directory (s3a://...) with Spark, convert to pandas."""
    from pyspark.sql import functions as F

    spark = _build_spark(app_name="ccfraud_train_read")

    try:
        path = cfg.parquet_path
        # normalize to s3a if someone passes s3://
        if path.startswith("s3://"):
            path = "s3a://" + path[len("s3://") :]

        df = spark.read.parquet(path)

        # Optional: enforce label col presence early
        if cfg.label_col not in df.columns:
            raise ValueError(
                f"Label column '{cfg.label_col}' not found. Columns={df.columns}"
            )

        # Optional row cap (for safety in small containers)
        if cfg.max_rows is not None:
            df = df.limit(int(cfg.max_rows))

        # Coalesce partitions before toPandas (less overhead)
        if cfg.spark_partitions and cfg.spark_partitions > 0:
            df = df.coalesce(int(cfg.spark_partitions))

        # Convert timestamp columns to something pandas handles well
        # (Spark timestamps become pandas datetime64[ns] via Arrow)
        pdf = df.toPandas()

        return pdf
    finally:
        spark.stop()


def train_and_log(cfg: TrainConfig) -> str:
    import os

    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # -------------------------
    # MLflow config
    # -------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ccfraud-train"))

    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL", "ccfraud_baseline")
    promote_to_stage = os.getenv("MLFLOW_PROMOTE_TO_STAGE", "")  # e.g. "Staging" / "Production"
    set_alias = os.getenv("MLFLOW_SET_ALIAS", "")  # e.g. "candidate"

    # Nice run name instead of "gaudy-stork-892"
    run_name = None
    if getattr(cfg, "ds", None) and getattr(cfg, "pointer_name", None):
        run_name = f"train_ds={cfg.ds}_ptr={cfg.pointer_name}"
    elif getattr(cfg, "ds", None):
        run_name = f"train_ds={cfg.ds}"

    # -------------------------
    # Load training data
    # -------------------------
    df = _read_training_parquet_to_pandas(cfg)
    df = df.dropna(subset=[cfg.label_col])
    y = df[cfg.label_col].astype(int)

    drop_cols = {"t_id", "event_ts", cfg.label_col}
    X = df[[c for c in df.columns if c not in drop_cols]]

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if len(set(y)) > 1 else None,  # guard if all-one-class in tiny samples
    )

    # -------------------------
    # Train + log
    # -------------------------
    with mlflow.start_run(run_name=run_name) as run:
        # ---- tags (discoverability) ----
        # Airflow context tags (present in Airflow task env)
        mlflow.set_tag("airflow_dag_id", os.getenv("AIRFLOW_CTX_DAG_ID", ""))
        mlflow.set_tag("airflow_task_id", os.getenv("AIRFLOW_CTX_TASK_ID", ""))
        mlflow.set_tag("airflow_run_id", os.getenv("AIRFLOW_CTX_DAG_RUN_ID", ""))
        mlflow.set_tag("airflow_execution_date", os.getenv("AIRFLOW_CTX_EXECUTION_DATE", ""))

        # Training data provenance tags (from cfg)
        if getattr(cfg, "ds", None):
            mlflow.set_tag("ds", cfg.ds)
        if getattr(cfg, "pointer_name", None):
            mlflow.set_tag("train_pointer", cfg.pointer_name)
        if getattr(cfg, "pointer_key", None):
            mlflow.set_tag("train_pointer_key", cfg.pointer_key)
        if getattr(cfg, "resolved_prefix", None):
            mlflow.set_tag("train_resolved_prefix", cfg.resolved_prefix)

        mlflow.set_tag("train_data_uri", cfg.parquet_path)

        # ---- train ----
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        # ---- metrics ----
        mlflow.log_metrics({
            "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y_test, proba)) if len(set(y_test)) > 1 else float("nan"),
            "n_rows": int(len(df)),
            "n_pos": int(y.sum()),
            "pos_rate": float(y.mean()),
        })

        # ---- params ----
        mlflow.log_params({
            "parquet_path": cfg.parquet_path,
            "label_col": cfg.label_col,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "max_rows": cfg.max_rows if getattr(cfg, "max_rows", None) is not None else "",
            "spark_partitions": getattr(cfg, "spark_partitions", ""),
            "model_type": "logreg",
            "class_weight": "balanced",
            "max_iter": 500,
            "n_features_input": int(X.shape[1]),
            "n_cat_cols": int(len(cat_cols)),
            "n_num_cols": int(len(num_cols)),
        })

        # ---- signature + example ----
        input_example = X_train.head(5)
        signature = infer_signature(input_example, pipe.predict_proba(input_example)[:, 1])

        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        mlflow.log_text("\n".join(list(X.columns)), "feature_columns.txt")

        # ---- register model ----
        mv = mlflow.register_model(model_uri=model_info.model_uri, name=registered_model_name)
        mlflow.set_tag("registered_model_name", registered_model_name)
        mlflow.set_tag("registered_model_version", str(mv.version))

        client = mlflow.tracking.MlflowClient()

        # Optional: promote to stage (older registry workflow)
        if promote_to_stage:
            client.transition_model_version_stage(
                name=registered_model_name,
                version=mv.version,
                stage=promote_to_stage,
                archive_existing_versions=True,
            )
            mlflow.set_tag("promoted_to_stage", promote_to_stage)

        # Optional: set alias (newer workflow; nicer than stages)
        if set_alias:
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=set_alias,
                version=mv.version,
            )
            mlflow.set_tag("set_alias", set_alias)

        return run.info.run_id

