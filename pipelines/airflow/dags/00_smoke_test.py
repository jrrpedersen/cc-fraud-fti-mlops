from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator


def check_minio() -> None:
    """
    Health check for MinIO (S3-compatible) service.
    """
    url = os.getenv("S3_ENDPOINT_URL", "http://minio:9000").rstrip("/")
    health = f"{url}/minio/health/ready"
    r = requests.get(health, timeout=5)
    r.raise_for_status()
    print(f"MinIO ready: {health} -> {r.status_code}")


def check_redis() -> None:
    """
    Simple PING check for Redis.
    """
    import redis  # installed via _PIP_ADDITIONAL_REQUIREMENTS in docker-compose

    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    client = redis.Redis(host=host, port=port, decode_responses=True, socket_connect_timeout=3)
    resp = client.ping()
    if resp is not True:
        raise RuntimeError(f"Redis ping failed: {resp}")
    print(f"Redis ping ok: {host}:{port}")


def print_env() -> None:
    """
    Print key env vars to make debugging easy in logs.
    """
    keys = [
        "AIRFLOW__CORE__EXECUTOR",
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "S3_ENDPOINT_URL",
        "REDIS_HOST",
        "REDIS_PORT",
    ]
    for k in keys:
        print(f"{k}={os.getenv(k)}")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="00_smoke_test_stack",
    description="Smoke test for local stack: MinIO + Redis connectivity",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,  # manual trigger
    catchup=False,
    tags=["smoke", "phase-1"],
) as dag:
    t_env = PythonOperator(task_id="print_env", python_callable=print_env)
    t_minio = PythonOperator(task_id="check_minio", python_callable=check_minio)
    t_redis = PythonOperator(task_id="check_redis", python_callable=check_redis)

    t_env >> [t_minio, t_redis]