# Phase 1 — Step 3b: Feast online materialization + smoke test

This milestone confirms that our **Model-Independent Transformations (MITs)** can be:
1) computed in Spark (Silver layer),
2) registered as Feast FeatureViews, and
3) **materialized into an online store (Redis)** for low-latency serving.

At the end of this step, we can fetch features **online** by entity keys (`cc_num`, `merchant_id`) and get real values back (not `None`).

---

## What we have now

From prior steps, we already have:

- **Landing (raw)** in MinIO: JSONL transactions + labels
- **Bronze** in MinIO: Parquet copies of transactions + labels
- **Silver MIT features**:
  - card-level rolling aggregates (10m / 1h / 1d / 7d) + recency
  - merchant-level rolling fraud counts (1d / 7d / 30d)
- **Feast feature repository** (`feature_repo/`) defining:
  - entities (`cc_num`, `merchant_id`)
  - feature views backed by the Silver outputs
  - Redis online store configuration

---

## Commands

### 1) Apply Feast definitions (creates/updates registry)

```powershell
cd docker
docker compose run --rm feast feast apply
```

Notes:
- You may see a Pydantic deprecation warning. It’s safe to ignore.
- If this fails, check `feature_repo/entities.py` and `feature_repo/feature_views.py` for type/path issues.

### 2) Materialize features into Redis

For local development, **full day materialization** is the most reliable:

```powershell
cd docker
docker compose run --rm feast feast materialize 2025-12-29T00:00:00 2025-12-29T23:59:59
```

Once the project is stable and you are no longer generating “time-travel” data, you can use incremental:

```powershell
cd docker
docker compose run --rm feast feast materialize-incremental (Get-Date).ToString("s")
```

Why full materialization helped:
- Earlier we generated timestamps that were *in the future*, which can cause incremental materialization to record a “watermark” that doesn’t overlap the corrected data.
- A full-day materialize resets the mismatch and populates Redis as expected.

---

## Smoke test (online feature retrieval)

We added a small script that fetches online features for one `cc_num` + `merchant_id`.

### Run via Docker (recommended)

```powershell
cd docker
docker compose run --rm feast python /feature_repo/../scripts/feast_smoke_test.py --repo-path /feature_repo --require-non-null
```

Expected output includes real values, e.g.:

- `cc_cnt_10m`: 1
- `cc_sum_amt_10m`: 66.25
- `m_fraud_cnt_7d`: 0

If you see `None` values for all features, Redis likely wasn’t populated for the requested time range/entities.

---

## Debugging checklist

### A) Verify your offline snapshots exist (inside Feast container)

```powershell
cd docker
docker compose run --rm feast bash -lc "ls -la data/offline/card_features/current | head"
docker compose run --rm feast bash -lc "ls -la data/offline/merchant_features/current | head"
```

### B) Confirm timestamps are not in the future

```powershell
cd docker
docker compose run --rm feast bash -lc "python - << 'PY'
import glob, pandas as pd
df = pd.read_parquet(glob.glob('data/offline/card_features/current/*.parquet')[0])
print('min/max event_ts:', df['event_ts'].min(), df['event_ts'].max())
PY"
```

If `max(event_ts)` is greater than “now”, Feast materialization “to now” won’t include those rows.

### C) Materialize the correct window
If your data for `ds` only covers a smaller range (e.g., 00:00–01:48), a full-day materialize is still okay; it will just include what exists.

### D) Reset registry in development (only if needed)
If incremental windows keep behaving oddly during iteration:

```powershell
# Windows: from repo root
Remove-Item .\feature_repo\data\registry.db -ErrorAction SilentlyContinue

cd docker
docker compose run --rm feast feast apply
docker compose run --rm feast feast materialize 2025-12-29T00:00:00 2025-12-29T23:59:59
```

---

## Outcome

We now have a working **Feature → (MIT) → Online Store** loop:

- Spark builds MIT feature sets (Silver)
- Feast defines/validates feature contracts
- Materialization loads Redis
- Online inference can fetch features by entity key without touching the offline store

Next step: implement the **Training pipeline** (MDTs applied in train + inference consistently) and log models/metrics to MLflow, then deploy a thin inference service that:
- logs requests (ODT inputs),
- writes events to Bronze asynchronously (durable ingestion),
- and fetches online features from Redis/Feast for real-time predictions.
