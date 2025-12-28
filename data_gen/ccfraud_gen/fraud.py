from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .geo import far_country, sample_ip, sample_lat_lon


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def chain_attack_events(
    rng: random.Random,
    card: dict[str, Any],
    merchants_by_country: dict[str, list[dict[str, Any]]],
    start: datetime,
    duration_min: int,
    duration_max: int,
    size_min: int,
    size_max: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate a chain attack: multiple online transactions over a short period.

    Returns (transactions, fraud_labels).
    """
    n = rng.randint(size_min, size_max)
    dur = rng.randint(duration_min, duration_max)
    t0 = start

    # Fraud country: often not home
    fraud_country = far_country(rng, exclude=set())
    txs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    # escalating amount pattern: small test, then larger
    base = rng.uniform(1.0, 10.0)
    for j in range(n):
        frac = (j / max(1, n - 1))
        # exponential-ish escalation
        amount = round(base * (1.0 + (frac * frac) * rng.uniform(20, 120)), 2)
        dt = t0 + timedelta(seconds=int((dur * 60) * frac)) + timedelta(seconds=rng.randint(0, 30))
        lat, lon = sample_lat_lon(rng, fraud_country)
        ip = sample_ip(rng, fraud_country)

        # pick a merchant in that country, prefer ecommerce category if present
        mlist = merchants_by_country.get(fraud_country) or []
        merchant = rng.choice(mlist) if mlist else {"merchant_id": "m_unknown", "category": "ecommerce", "country": fraud_country}

        tx_id = f"tx_f_chain_{card['cc_num']}_{start.timestamp():.0f}_{j}"
        tx = {
            "t_id": tx_id,
            "cc_num": card["cc_num"],
            "account_id": card["account_id"],
            "merchant_id": merchant["merchant_id"],
            "amount": amount,
            "currency": "USD",
            "country": fraud_country,
            "ip_address": ip,
            "card_present": False,
            "channel": "online",
            "category": merchant.get("category", "ecommerce"),
            "lat": lat,
            "lon": lon,
            "ts": _iso(dt),
        }
        txs.append(tx)
        labels.append(
            {
                "t_id": tx_id,
                "cc_num": card["cc_num"],
                "explanation": "chain_attack",
                "ts": tx["ts"],
            }
        )
    return txs, labels


def geo_fraud_pair(
    rng: random.Random,
    card: dict[str, Any],
    merchants_by_country: dict[str, list[dict[str, Any]]],
    start: datetime,
    gap_min: int,
    gap_max: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate a geographic fraud pair: two far-apart card-present transactions close in time.

    Returns (transactions, fraud_labels) for BOTH transactions.
    """
    c1 = far_country(rng, exclude=set())
    c2 = far_country(rng, exclude={c1})
    gap = rng.randint(gap_min, gap_max)
    t1 = start
    t2 = start + timedelta(minutes=gap)

    txs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    for idx, (country, dt) in enumerate([(c1, t1), (c2, t2)]):
        lat, lon = sample_lat_lon(rng, country)
        ip = sample_ip(rng, country)
        mlist = merchants_by_country.get(country) or []
        merchant = rng.choice(mlist) if mlist else {"merchant_id": "m_unknown", "category": "travel", "country": country}

        tx_id = f"tx_f_geo_{card['cc_num']}_{start.timestamp():.0f}_{idx}"
        tx = {
            "t_id": tx_id,
            "cc_num": card["cc_num"],
            "account_id": card["account_id"],
            "merchant_id": merchant["merchant_id"],
            "amount": round(rng.uniform(20, 600), 2),
            "currency": "USD",
            "country": country,
            "ip_address": ip,
            "card_present": True,
            "channel": "pos",
            "category": merchant.get("category", "travel"),
            "lat": lat,
            "lon": lon,
            "ts": _iso(dt),
        }
        txs.append(tx)
        labels.append(
            {
                "t_id": tx_id,
                "cc_num": card["cc_num"],
                "explanation": "geo_fraud",
                "ts": tx["ts"],
            }
        )
    return txs, labels
