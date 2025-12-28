from __future__ import annotations

from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str = "ccfraud_bronze_ingest") -> SparkSession:
    # Local mode Spark inside the Airflow container
    return (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        # keep it modest for local dev
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def jsonl_to_parquet(jsonl_path: Path, out_dir: Path, spark: Optional[SparkSession] = None) -> None:
    spark = spark or build_spark()

    df = spark.read.json(str(jsonl_path))
    # Standardize timestamp column if present
    if "ts" in df.columns:
        df = df.withColumn("ts", F.to_timestamp("ts"))

    out_dir.mkdir(parents=True, exist_ok=True)
    (df.write.mode("overwrite").parquet(str(out_dir)))


def stop_spark(spark: SparkSession) -> None:
    try:
        spark.stop()
    except Exception:
        pass
