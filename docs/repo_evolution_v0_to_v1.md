# Repo evolution: from v0 smoke ingestion to v1 realistic synthetic world

This repository is being built iteratively to demonstrate an end-to-end **FTI (Feature → Train → Inference)** MLOps system.

## v0 — Minimal generator + landing pipeline (now tagged)

The initial milestone focused on **platform wiring and repeatability**:

- Local stack boots via Docker Compose (Airflow + Postgres + MinIO + Redis)
- An Airflow DAG generates a simple JSONL transactions file
- The DAG uploads it to MinIO under a partitioned landing path
- A `_LATEST` pointer is written to provide a stable “latest” location per day

This version is preserved as the Git tag:

- `v0-smoke-ingestion`

Use this tag when you want to show the “first working slice” and early infrastructure bring-up.

## Why we moved on

A realistic fraud ML system requires **coherent synthetic data**:

- stable entities (banks, merchants, accounts, cards)
- transactions that reflect customer behavior (home country, travel, continuity)
- fraud labels produced by plausible attack patterns (e.g., chain attacks, geo fraud)

Without these properties, downstream features and models become “toy” signals and won’t reflect real-world behavior.

## v1 — Realistic approach: bootstrap reference world + streaming-friendly transaction generation

Starting with v1, synthetic generation is split into two parts:

### 1) Bootstrap “reference world” (slow-changing dimensions)
Generated infrequently (e.g., once per environment, or when you want a new scenario):

- `banks`
- `merchants`
- `accounts` (includes `home_country`)
- `cards` (links to accounts)

Written to MinIO under a stable world id:

```
landing/reference/world_id=<WORLD_ID>/banks.jsonl
landing/reference/world_id=<WORLD_ID>/merchants.jsonl
landing/reference/world_id=<WORLD_ID>/accounts.jsonl
landing/reference/world_id=<WORLD_ID>/cards.jsonl
landing/reference/_CURRENT          # pointer containing <WORLD_ID>
```

Airflow DAG:
- `00_bootstrap_reference_world`

### 2) Generate transactions incrementally (event stream)
Generated frequently (manual runs now; later can be scheduled or replaced by Kafka producer):

- `transactions.jsonl`
- `fraud_labels.jsonl` (can later be delayed to simulate label latency)

Written to daily partitions with immutable run ids, plus `_LATEST` pointers:

```
landing/transactions/dt=YYYY-MM-DD/run_id=<RUN_ID>/transactions.jsonl
landing/transactions/dt=YYYY-MM-DD/_LATEST

landing/fraud_labels/dt=YYYY-MM-DD/run_id=<RUN_ID>/fraud_labels.jsonl
landing/fraud_labels/dt=YYYY-MM-DD/_LATEST
```

Airflow DAG:
- `01_generate_and_land_transactions_v1`