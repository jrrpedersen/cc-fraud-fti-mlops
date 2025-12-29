#!/usr/bin/env python3
"""
Feast smoke test (online retrieval)

What it does
- Reads one row from the *offline* "current" snapshots (Parquet) produced by the Silver MIT pipeline
- Uses Feast to fetch the corresponding *online* features from Redis
- Prints the result and exits non-zero if features are missing (optional)

Run (from repo root)
  python scripts/feast_smoke_test.py --repo-path feature_repo

Run via Docker (recommended, no local deps)
  cd docker
  docker compose run --rm feast python /feature_repo/../scripts/feast_smoke_test.py --repo-path /feature_repo

Notes
- This assumes you've already run:
    - Airflow DAG 03_build_silver_mit_features_v1
    - feast apply
    - feast materialize (or materialize-incremental)
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from feast import FeatureStore


DEFAULT_FEATURES = [
    "card_txn_features:cc_cnt_10m",
    "card_txn_features:cc_sum_amt_10m",
    "card_txn_features:cc_cnt_1h",
    "merchant_risk_features:m_fraud_cnt_7d",
]


@dataclass
class Inputs:
    cc_num: str
    merchant_id: str


def _pick_first_value(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in parquet columns={list(df.columns)}")
    val = df.iloc[0][col]
    return str(val)


def _read_one_parquet(glob_pat: str) -> pd.DataFrame:
    files = glob.glob(glob_pat)
    if not files:
        raise FileNotFoundError(f"No parquet files found matching: {glob_pat}")
    return pd.read_parquet(files[0])


def resolve_inputs(
    card_glob: str,
    merchant_glob: str,
    cc_num: Optional[str],
    merchant_id: Optional[str],
) -> Inputs:
    card_df = _read_one_parquet(card_glob)
    merch_df = _read_one_parquet(merchant_glob)

    cc = cc_num or _pick_first_value(card_df, "cc_num")
    mid = merchant_id or _pick_first_value(merch_df, "merchant_id")
    return Inputs(cc_num=cc, merchant_id=mid)


def run(
    repo_path: str,
    features: List[str],
    card_glob: str,
    merchant_glob: str,
    cc_num: Optional[str],
    merchant_id: Optional[str],
    require_non_null: bool,
) -> int:
    inputs = resolve_inputs(card_glob, merchant_glob, cc_num, merchant_id)

    store = FeatureStore(repo_path=repo_path)
    resp: Dict[str, Any] = store.get_online_features(
        features=features,
        entity_rows=[{"cc_num": inputs.cc_num, "merchant_id": inputs.merchant_id}],
    ).to_dict()

    print("cc_num:", inputs.cc_num)
    print("merchant_id:", inputs.merchant_id)
    print(resp)

    if require_non_null:
        missing = []
        for k, v in resp.items():
            if k in ("cc_num", "merchant_id"):
                continue
            if isinstance(v, list) and len(v) > 0 and v[0] is None:
                missing.append(k)
        if missing:
            print(f"ERROR: missing online values for: {missing}", file=sys.stderr)
            return 2

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Feast online retrieval smoke test")
    p.add_argument("--repo-path", default="feature_repo", help="Path to Feast repo (contains feature_store.yaml)")
    p.add_argument(
        "--card-parquet-glob",
        default="feature_repo/data/offline/card_features/current/*.parquet",
        help="Glob for offline 'current' card feature snapshots",
    )
    p.add_argument(
        "--merchant-parquet-glob",
        default="feature_repo/data/offline/merchant_features/current/*.parquet",
        help="Glob for offline 'current' merchant feature snapshots",
    )
    p.add_argument("--cc-num", default=None, help="Override cc_num entity id")
    p.add_argument("--merchant-id", default=None, help="Override merchant_id entity id")
    p.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feast feature references to fetch, e.g. card_txn_features:cc_cnt_10m",
    )
    p.add_argument(
        "--require-non-null",
        action="store_true",
        help="Exit non-zero if any requested feature is None",
    )

    args = p.parse_args()
    try:
        return run(
            repo_path=args.repo_path,
            features=args.features,
            card_glob=args.card_parquet_glob,
            merchant_glob=args.merchant_parquet_glob,
            cc_num=args.cc_num,
            merchant_id=args.merchant_id,
            require_non_null=args.require_non_null,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
