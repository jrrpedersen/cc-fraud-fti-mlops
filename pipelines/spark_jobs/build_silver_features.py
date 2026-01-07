"""
Build Silver MIT features from Bronze parquet.

Design goals:
- Point-in-time safe for training (per-transaction rolling windows).
- Also produce "latest snapshot" per entity for online serving materialization.
- Keep transformations model-independent (MIT): windowed sums/counts, lags, recency.
  (MDTs like scaling/encoding should happen in training/inference pipelines.)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

def _lazy_import_spark():
    from pyspark.sql import SparkSession  # noqa
    from pyspark.sql import functions as F  # noqa
    from pyspark.sql.window import Window  # noqa
    return SparkSession, F, Window


@dataclass(frozen=True)
class SilverPaths:
    # Inputs (local)
    bronze_tx_parquet_dir: Path
    bronze_labels_parquet_dir: Path

    # Outputs (local)
    out_training_dataset_dir: Path
    out_card_snapshot_dir: Path
    out_merchant_snapshot_dir: Path


def build_spark(app_name: str = "ccfraud_silver_mit") -> "SparkSession":
    SparkSession, _, _ = _lazy_import_spark()
    return (
        SparkSession.builder
        .master("local[*]") # run Spark locally using all CPU cores available in the container/host
        .appName(app_name)
        # keep local runs stable on laptops
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )


def compute_silver(
    paths: SilverPaths,
    spark: Optional["SparkSession"] = None,
) -> None:
    """
    Reads bronze parquet datasets (transactions + fraud_labels),
    computes MIT features, and writes:
      - training_dataset (1 row per transaction, includes label)
      - card_features_snapshot (latest row per cc_num)
      - merchant_features_snapshot (latest row per merchant_id)
    """
    SparkSession, F, Window = _lazy_import_spark() # delays importing pyspark until runtime
    spark = spark or build_spark()

    tx = spark.read.parquet(str(paths.bronze_tx_parquet_dir))
    labels = spark.read.parquet(str(paths.bronze_labels_parquet_dir))

    # ---- Normalize + prepare ----
    # Expecting columns from our generator (v1). If schema differs,
    # adjust these field names in one place here.
    required_tx_cols = {"t_id", "ts", "cc_num", "merchant_id", "amount", "ip_address", "card_present"}
    missing = sorted(list(required_tx_cols - set(tx.columns)))
    if missing:
        raise ValueError(f"Transactions missing required columns: {missing}. Available={sorted(tx.columns)}")

    if "t_id" not in labels.columns:
        raise ValueError(f"Fraud labels parquet must contain t_id. Available={sorted(labels.columns)}")

    # Timestamp normalization
    tx = tx.withColumn("ts", F.to_timestamp("ts"))
    # Epoch seconds for window calculations
    tx = tx.withColumn("ts_epoch", F.unix_timestamp("ts").cast("long"))

    # Label: 1 if transaction id present in fraud_labels else 0
    labels_small = labels.select(F.col("t_id").alias("lbl_t_id")).withColumn("label_is_fraud", F.lit(1))
    tx = tx.join(labels_small, tx.t_id == labels_small.lbl_t_id, how="left").drop("lbl_t_id")
    tx = tx.withColumn("label_is_fraud", F.coalesce(F.col("label_is_fraud"), F.lit(0)).cast("int"))

    # ---- Card-level rolling windows (MIT) ----
    w_cc = Window.partitionBy("cc_num").orderBy(F.col("ts_epoch")).rangeBetween

    def _sum_amt(seconds: int):
        return F.sum("amount").over(w_cc(-seconds, 0))

    def _cnt(seconds: int):
        return F.count(F.lit(1)).over(w_cc(-seconds, 0))

    tx = (
        tx
        .withColumn("cc_sum_amt_10m", _sum_amt(600))
        .withColumn("cc_cnt_10m", _cnt(600))
        .withColumn("cc_sum_amt_1h", _sum_amt(3600))
        .withColumn("cc_cnt_1h", _cnt(3600))
        .withColumn("cc_sum_amt_1d", _sum_amt(86400))
        .withColumn("cc_cnt_1d", _cnt(86400))
        .withColumn("cc_sum_amt_7d", _sum_amt(7 * 86400))
        .withColumn("cc_cnt_7d", _cnt(7 * 86400))
    )

    # ---- Lags / recency (MIT) ----
    w_cc_rows = Window.partitionBy("cc_num").orderBy(F.col("ts_epoch"))
    tx = (
        tx
        .withColumn("cc_prev_ts_epoch", F.lag("ts_epoch", 1).over(w_cc_rows))
        .withColumn("cc_prev_ip", F.lag("ip_address", 1).over(w_cc_rows))
        .withColumn("cc_prev_card_present", F.lag("card_present", 1).over(w_cc_rows))
        .withColumn("cc_time_since_last_s", (F.col("ts_epoch") - F.col("cc_prev_ts_epoch")).cast("long"))
    )

    # ---- Merchant "risk" rolling fraud counts (MIT) ----
    # Count of prior fraudulent transactions for this merchant in various lookback windows
    # Note: we include the current transaction in the count (rangeBetween is inclusive).
    # Risk: Label leakage
    w_m = Window.partitionBy("merchant_id").orderBy(F.col("ts_epoch")).rangeBetween
    tx = (
        tx
        .withColumn("m_fraud_cnt_1d", F.sum("label_is_fraud").over(w_m(-86400, 0)))
        .withColumn("m_fraud_cnt_7d", F.sum("label_is_fraud").over(w_m(-(7 * 86400), 0)))
        .withColumn("m_fraud_cnt_30d", F.sum("label_is_fraud").over(w_m(-(30 * 86400), 0)))
    )

    # ---- Output: training dataset (per-transaction) ----
    # Keep it explicit and stable (helps downstream training + docs).
    keep_cols = [
        "t_id", "ts", "cc_num", "merchant_id", "amount", "ip_address", "card_present",
        "cc_sum_amt_10m", "cc_cnt_10m",
        "cc_sum_amt_1h", "cc_cnt_1h",
        "cc_sum_amt_1d", "cc_cnt_1d",
        "cc_sum_amt_7d", "cc_cnt_7d",
        "cc_prev_ts_epoch", "cc_prev_ip", "cc_prev_card_present", "cc_time_since_last_s",
        "m_fraud_cnt_1d", "m_fraud_cnt_7d", "m_fraud_cnt_30d",
        "label_is_fraud",
    ]
    for c in keep_cols:
        if c not in tx.columns:
            raise ValueError(f"Expected output column missing: {c}")

    training = tx.select(*keep_cols)

    # overwrite local dirs (safe: they are derived)
    paths.out_training_dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.out_card_snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths.out_merchant_snapshot_dir.mkdir(parents=True, exist_ok=True)

    training.write.mode("overwrite").parquet(str(paths.out_training_dataset_dir))

    # ---- Output: latest snapshot per entity for online store materialization ----
    # Latest card features
    w_cc_latest = Window.partitionBy("cc_num").orderBy(F.col("ts_epoch").desc())
    card_snapshot = (
        tx
        .withColumn("_rn", F.row_number().over(w_cc_latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .select(
            "cc_num",
            F.col("ts").alias("event_ts"),
            "cc_sum_amt_10m", "cc_cnt_10m",
            "cc_sum_amt_1h", "cc_cnt_1h",
            "cc_sum_amt_1d", "cc_cnt_1d",
            "cc_sum_amt_7d", "cc_cnt_7d",
            "cc_time_since_last_s",
        )
    )
    card_snapshot.write.mode("overwrite").parquet(str(paths.out_card_snapshot_dir))

    # Latest merchant features
    w_m_latest = Window.partitionBy("merchant_id").orderBy(F.col("ts_epoch").desc())
    merchant_snapshot = (
        tx
        .withColumn("_rn", F.row_number().over(w_m_latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .select(
            "merchant_id",
            F.col("ts").alias("event_ts"),
            "m_fraud_cnt_1d", "m_fraud_cnt_7d", "m_fraud_cnt_30d",
        )
    )
    merchant_snapshot.write.mode("overwrite").parquet(str(paths.out_merchant_snapshot_dir))


def stop_spark(spark: "SparkSession") -> None:
    try:
        spark.stop()
    except Exception:
        pass
