"""
Build a labeled training dataset using point-in-time (as-of) joins.

Inputs
- Bronze transactions parquet (MinIO): contains transaction fields including ts, cc_num, merchant_id, etc.
- Bronze labels parquet (MinIO): t_id -> label_is_fraud
- Silver MIT feature snapshots (local parquet under feature_repo/data/offline/.../current)
    - card_features: keyed by cc_num with event_ts
    - merchant_features: keyed by merchant_id with event_ts

Output
- s3a://mlops/silver/training_dataset/dt=<ds>/run_id=<run_id>/

We do "as-of joins" to prevent time-travel leakage:
for each tx, pick the latest feature row with feature.event_ts <= tx.event_ts.
"""

from __future__ import annotations

import os
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Optional

import boto3

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

def _s3_client():
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret, region_name=region)

def _read_pointer(bucket: str, key: str) -> str:
    s3 = _s3_client()
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8").strip()


@dataclass
class Paths:
    bucket: str = "mlops"
    bronze_tx_prefix: str = "bronze/transactions/_CURRENT"
    bronze_label_prefix: str = "bronze/fraud_labels/_CURRENT"
    out_prefix: str = "silver/training_dataset"

    # Local snapshots written by Silver MIT step (mirrored under repo)
    card_features_glob: str = "/opt/airflow/repo/feature_repo/data/offline/card_features/current/*.parquet"
    merchant_features_glob: str = "/opt/airflow/repo/feature_repo/data/offline/merchant_features/current/*.parquet"


def build_spark(app_name: str) -> SparkSession:
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    # S3A jars + writable ivy cache
    spark_packages = os.getenv(
        "SPARK_JARS_PACKAGES",
        "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")

        # jar deps for s3a
        .config("spark.jars.packages", spark_packages)
        .config("spark.jars.ivy", "/tmp/.ivy2")  # avoid permission issues in containers

        # S3A configs
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


def _s3a(bucket: str, prefix: str) -> str:
    return f"s3a://{bucket}/{prefix.lstrip('/')}"


def _read_bronze(spark: SparkSession, bucket: str, prefix: str) -> DataFrame:
    return spark.read.parquet(_s3a(bucket, prefix))


def _read_local_features(spark: SparkSession, glob_path: str) -> DataFrame:
    return spark.read.parquet(glob_path)


def _asof_join(
    left: DataFrame,
    right: DataFrame,
    left_key: str,
    right_key: str,
    left_ts: str,
    right_ts: str,
    prefix: str,
) -> DataFrame:
    """
    For each left row (tx), pick the right row with max(right_ts) where right_ts <= left_ts.
    Avoid duplicate column names by renaming right key + right ts inside the join.
    """

    r_key = f"__{prefix}key"   # temp name for right entity key
    r_ts  = f"__{prefix}ts"    # temp name for right timestamp

    r = (
        right
        .withColumnRenamed(right_key, r_key)
        .withColumnRenamed(right_ts, r_ts)
    )

    cond = (F.col(f"l.{left_key}") == F.col(f"r.{r_key}")) & (F.col(f"r.{r_ts}") <= F.col(f"l.{left_ts}"))
    joined = left.alias("l").join(r.alias("r"), cond, how="left")

    w = Window.partitionBy(F.col("l.t_id")).orderBy(F.col(f"r.{r_ts}").desc_nulls_last())
    ranked = joined.withColumn("__rn", F.row_number().over(w)).where(F.col("__rn") == 1).drop("__rn")

    # Rename right-side feature columns (avoid collisions)
    # Exclude temp key+ts columns
    right_cols = [c for c in right.columns if c not in (right_key, right_ts)]
    for c in right_cols:
        new_name = c if c.startswith(prefix) else f"{prefix}{c}"
        if new_name != c:
            ranked = ranked.withColumnRenamed(c, new_name)

    # Drop temp columns so they never collide downstream
    ranked = ranked.drop(r_key).drop(r_ts)

    return ranked

def _dedupe_columns(df: DataFrame) -> DataFrame:
    # keep first occurrence of each name
    cols = list(OrderedDict((c, None) for c in df.columns).keys())
    return df.select(*[F.col(c) for c in cols])


def build_training_dataset(ds: str, run_id: str, paths: Optional[Paths] = None) -> str:
    paths = paths or Paths()
    spark = build_spark(app_name="build_training_dataset_v1")

    try:
        # Bronze inputs
        tx_prefix = _read_pointer(paths.bucket, paths.bronze_tx_prefix)
        lbl_prefix = _read_pointer(paths.bucket, paths.bronze_label_prefix)

        print("Resolved tx_prefix:", tx_prefix)
        print("Resolved lbl_prefix:", lbl_prefix)

        tx = _read_bronze(spark, paths.bucket, tx_prefix).withColumn("event_ts", F.to_timestamp("ts"))

        y = _read_bronze(spark, paths.bucket, lbl_prefix).select(
            F.col("t_id").alias("t_id_y"),
            F.lit(1).cast("int").alias("label_is_fraud"),
            F.col("explanation").alias("fraud_explanation"),
        )

        # Join labels
        base = (
            tx.join(y, tx.t_id == y.t_id_y, how="left")
              .drop("t_id_y")
              .withColumn("label_is_fraud", F.coalesce(F.col("label_is_fraud"), F.lit(0)))
        )

        # Local MIT snapshots
        card = _read_local_features(spark, paths.card_features_glob).withColumn("event_ts", F.to_timestamp("event_ts"))
        merch = _read_local_features(spark, paths.merchant_features_glob).withColumn("event_ts", F.to_timestamp("event_ts"))

        # As-of joins
        base = _asof_join(base, card, "cc_num", "cc_num", "event_ts", "event_ts", "card_")
        base = _asof_join(base, merch, "merchant_id", "merchant_id", "event_ts", "event_ts", "m_")

        # Training columns
        keep = [
            "t_id", "event_ts",
            "cc_num", "merchant_id",
            "amount", "country", "category", "channel", "card_present",
            "label_is_fraud",
        ]
        for c in base.columns:
            if c.startswith("card_") or c.startswith("m_"):
                keep.append(c)

        dataset = base.select(*[c for c in keep if c in base.columns])

        dups = [c for c, n in Counter(dataset.columns).items() if n > 1]
        print("Duplicate columns:", dups)

        dataset = _dedupe_columns(dataset)

        out_prefix = f"{paths.out_prefix}/dt={ds}/run_id={run_id}"
        out_uri = _s3a(paths.bucket, out_prefix)

        # 1) Write dataset
        dataset.write.mode("overwrite").parquet(out_uri)

        # 2) Update pointers only after successful write
        s3 = _s3_client()
        latest_key = f"{paths.out_prefix}/dt={ds}/_LATEST"
        current_key = f"{paths.out_prefix}/dt={ds}/_CURRENT"
        body = (out_prefix + "\n").encode("utf-8")

        s3.put_object(Bucket=paths.bucket, Key=latest_key, Body=body, ContentType="text/plain")
        s3.put_object(Bucket=paths.bucket, Key=current_key, Body=body, ContentType="text/plain")

        return out_uri

    finally:
        spark.stop()
