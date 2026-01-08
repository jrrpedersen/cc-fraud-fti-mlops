#!/usr/bin/env python
"""
inspect_training_dataset.py

Basic sanity checks for the *offline* training dataset written to MinIO (S3A).

Key idea:
- Training dataset = offline parquet in MinIO (silver/training_dataset/...)
- Redis = Feast *online* store for online features served at inference time
  (not where the training dataset lives)

Run (from host):
  docker exec -it docker-airflow-scheduler-1 bash -lc \
    "python /opt/airflow/repo/tools/inspect_training_dataset.py --ds 2026-01-07 --pointer latest"

Run (inside container):
  docker exec -it docker-airflow-scheduler-1 bash -lc "cd /opt/airflow/repo && python tools/inspect_training_dataset.py --ds 2026-01-08 --pointer latest"

Or point at an explicit run_id:
  docker exec -it docker-airflow-scheduler-1 bash -lc \
    "python /opt/airflow/repo/tools/inspect_training_dataset.py --ds 2026-01-07 --run-id manual__...+00_00"
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import boto3
from pyspark.sql import SparkSession, functions as F


def s3_client():
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name=region,
    )


def read_pointer(bucket: str, key: str) -> str:
    s3 = s3_client()
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8").strip()


def build_spark(app_name: str) -> SparkSession:
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    spark_packages = os.getenv(
        "SPARK_JARS_PACKAGES",
        "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        # jar deps for s3a
        .config("spark.jars.packages", spark_packages)
        .config("spark.jars.ivy", "/tmp/.ivy2")
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


def s3a_uri(bucket: str, prefix: str) -> str:
    return f"s3a://{bucket}/{prefix.lstrip('/')}"


def top_null_counts(df, cols: List[str], top_k: int = 20) -> Tuple[List[Tuple[str, int]], int]:
    n = df.count()
    counts: List[Tuple[str, int]] = []
    for c in cols:
        counts.append((c, df.filter(F.col(c).isNull()).count()))
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts[:top_k], n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=os.getenv("LANDING_BUCKET", "mlops"))
    p.add_argument("--ds", required=True, help="Airflow ds, e.g. 2026-01-07")
    p.add_argument(
        "--pointer",
        choices=["latest", "current", "none"],
        default="latest",
        help="Resolve dataset location from _LATEST/_CURRENT pointer (recommended).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="If set, overrides pointer resolution. Use the exact run_id folder name, e.g. manual__...+00_00",
    )
    p.add_argument(
        "--base-prefix",
        default="silver/training_dataset",
        help="Base prefix in bucket where training datasets are written.",
    )
    args = p.parse_args()

    # Resolve prefix
    if args.run_id:
        prefix = f"{args.base_prefix}/dt={args.ds}/run_id={args.run_id}"
        resolved_from = "explicit --run-id"
    else:
        if args.pointer == "none":
            raise SystemExit("Provide --run-id or use --pointer latest/current.")
        ptr_key = f"{args.base_prefix}/dt={args.ds}/_{args.pointer.upper()}"
        prefix = read_pointer(args.bucket, ptr_key)
        if not prefix:
            raise SystemExit(f"Pointer {ptr_key} is empty or missing")
        resolved_from = ptr_key

    uri = s3a_uri(args.bucket, prefix)

    spark = build_spark("inspect_training_dataset")
    try:
        df = spark.read.parquet(uri)

        print("=" * 88)
        print("Training dataset inspection")
        print(f"Resolved from: {resolved_from}")
        print(f"URI: {uri}")
        print("=" * 88)

        # Basic shape
        rows = df.count()
        cols = df.columns
        print(f"rows: {rows}")
        print(f"cols: {len(cols)}")
        print("columns:", cols)

        # Schema
        print("\nSchema:")
        df.printSchema()

        # Time range (handles either event_ts or ts)
        ts_col = "event_ts" if "event_ts" in cols else ("ts" if "ts" in cols else None)
        if ts_col:
            rng = df.agg(F.min(ts_col).alias("min_ts"), F.max(ts_col).alias("max_ts")).collect()[0]
            print(f"\nTime range ({ts_col}): {rng['min_ts']} -> {rng['max_ts']}")
        else:
            print("\nTime range: (no event_ts/ts column found)")

        # Labels
        if "label_is_fraud" in cols:
            print("\nLabel distribution:")
            df.groupBy("label_is_fraud").count().orderBy("label_is_fraud").show(truncate=False)
        else:
            print("\nLabel distribution: (label_is_fraud not found)")

        # Uniqueness & key coverage
        for key in ["t_id", "cc_num", "merchant_id"]:
            if key in cols:
                distinct = df.select(key).distinct().count()
                print(f"distinct {key}: {distinct}")

        if "t_id" in cols:
            dup = rows - df.select("t_id").distinct().count()
            print(f"duplicate t_id rows: {dup}")

        # Null-rate quick check
        print("\nNull-rate quick check (top 20 columns by nulls):")
        top_nulls, n = top_null_counts(df, cols, top_k=20)
        for c, k in top_nulls:
            pct = (k / n * 100.0) if n else 0.0
            print(f"{c:35s} {k}/{n} ({pct:.2f}%)")

        # Feature group coverage (quick proxy checks)
        card_cols = [c for c in cols if c.startswith("card_")]
        merch_cols = [c for c in cols if c.startswith("m_")]
        if card_cols:
            proxy = card_cols[0]
            non_null = df.select(F.sum(F.when(F.col(proxy).isNotNull(), 1).otherwise(0)).alias("nn")).collect()[0]["nn"]
            print(f"\ncard_* coverage proxy using {proxy}: {non_null}/{rows}")
        if merch_cols:
            proxy = merch_cols[0]
            non_null = df.select(F.sum(F.when(F.col(proxy).isNotNull(), 1).otherwise(0)).alias("nn")).collect()[0]["nn"]
            print(f"m_* coverage proxy using {proxy}: {non_null}/{rows}")

        # Basic stats
        if "amount" in cols:
            print("\nAmount stats:")
            df.select(
                F.count("amount").alias("n"),
                F.mean("amount").alias("mean"),
                F.stddev("amount").alias("std"),
                F.min("amount").alias("min"),
                F.expr("percentile_approx(amount, 0.5)").alias("p50"),
                F.expr("percentile_approx(amount, 0.9)").alias("p90"),
                F.max("amount").alias("max"),
            ).show(truncate=False)

        # Potential leakage “smell tests” (not a proof)
        m_leak_cols = [c for c in cols if c.startswith("m_fraud_cnt_")]
        if "label_is_fraud" in cols and m_leak_cols:
            mcol = m_leak_cols[0]
            print(f"\nLeakage smell test: mean {mcol} by label_is_fraud")
            df.groupBy("label_is_fraud").agg(F.mean(mcol).alias(f"mean_{mcol}")).orderBy("label_is_fraud").show(truncate=False)

        # Samples
        sample_cols = [c for c in ["t_id", ts_col, "cc_num", "merchant_id", "amount", "country", "category", "channel", "card_present", "label_is_fraud"] if c and c in cols]
        if "label_is_fraud" in cols and sample_cols:
            print("\n5 fraud examples:")
            df.filter(F.col("label_is_fraud") == 1).select(*sample_cols).show(5, truncate=False)
            print("\n5 non-fraud examples:")
            df.filter(F.col("label_is_fraud") == 0).select(*sample_cols).show(5, truncate=False)

        print("\nDONE ✅")
        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
