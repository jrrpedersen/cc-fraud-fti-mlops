"""
Baseline training + MLflow logging.

MDTs live here (inside sklearn Pipeline) to avoid train/serve skew.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
import pandas as pd


@dataclass
class TrainConfig:
    parquet_path: str
    label_col: str = "label_is_fraud"
    random_state: int = 42
    test_size: float = 0.2


def train_and_log(cfg: TrainConfig) -> str:
    import mlflow
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/opt/airflow/repo/mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ccfraud-train"))

    df = pd.read_parquet(cfg.parquet_path).dropna(subset=[cfg.label_col])
    y = df[cfg.label_col].astype(int)

    drop_cols = {"t_id", "event_ts", cfg.label_col}
    X = df[[c for c in df.columns if c not in drop_cols]]

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        mlflow.log_metrics({
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        })
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        mlflow.log_text("\n".join(list(X.columns)), "feature_columns.txt")
        return run.info.run_id