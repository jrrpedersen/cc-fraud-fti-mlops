# Feast Feature Repo (local dev)

This folder is a Feast feature repository. It defines:

- Entities: `cc_num`, `merchant_id`
- FeatureViews:
  - `card_txn_features` (per card)
  - `merchant_risk_features` (per merchant)

## How the offline data gets here

The Airflow DAG `03_build_silver_mit_features_v1` mirrors the **latest** entity snapshots to:

- `data/offline/card_features/current/`
- `data/offline/merchant_features/current/`

These directories contain Parquet files with an `event_ts` column and one row per entity.

## Running Feast (recommended via Docker)

See `docker/feast/` for a docker-compose patch + Dockerfile.

Quick commands (once the feast container exists):

```bash
feast apply
feast materialize-incremental $(date -Iseconds)
```

For Windows / PowerShell, you can materialize with:

```powershell
docker compose run --rm feast feast materialize-incremental (Get-Date).ToString("s")
```
