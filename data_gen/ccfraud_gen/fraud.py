from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .geo import sample_ip, sample_lat_lon


def _iso(dt: datetime) -> str:
    """
    Format a datetime as an ISO-8601 UTC timestamp string.

    - If `dt` is timezone-naive, it is assumed to be UTC.
    - The returned string is always in UTC and uses the common "Z" suffix
      (e.g., "2025-12-30T12:34:00Z") instead of "+00:00".

    Args:
        dt: A datetime to format (naive or timezone-aware).

    Returns:
        An ISO-8601 timestamp string in UTC ending with "Z".
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _eligible_countries(merchants_by_country: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [c for c, ms in merchants_by_country.items() if ms]


def chain_attack_events(
    rng: random.Random, # Passing it in is nice because we can seed it outside and get reproducible randomness
    card: dict[str, Any],
    merchants_by_country: dict[str, list[dict[str, Any]]],
    start: datetime,
    duration_min: int,
    duration_max: int,
    size_min: int,
    size_max: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate a synthetic "chain attack" fraud pattern for a single card.

    A chain attack represents multiple online (card-not-present) transactions occurring
    over a short time window, often with amounts that start small (test charge) and
    escalate toward larger purchases.

    The function:
      - Chooses a fraud country (independent of the card's home country in this implementation).
      - Samples timestamps between `start` and `start + duration` with small jitter.
      - Samples lat/lon and a synthetic IP address consistent with the fraud country.
      - Picks a merchant from `merchants_by_country` for that country.
      - Produces a corresponding fraud-label record for every generated transaction.

    Args:
        rng: Random number generator (pass a seeded instance for reproducibility).
        card: Card record containing at least "cc_num" and "account_id".
        merchants_by_country: Mapping from country code -> list of merchant records.
        start: Start time for the fraud burst (naive datetimes are treated as UTC).
        duration_min: Minimum burst duration in minutes.
        duration_max: Maximum burst duration in minutes.
        size_min: Minimum number of transactions to generate in the chain.
        size_max: Maximum number of transactions to generate in the chain.

    Returns:
        A tuple (transactions, fraud_labels):
          - transactions: list of transaction dicts with fields such as t_id, cc_num,
            merchant_id, amount, country, ip_address, lat/lon, channel="online",
            card_present=False, and ts (ISO UTC string).
          - fraud_labels: list of label dicts keyed by t_id with explanation="chain_attack".

    Notes:
        - This implementation does not currently exclude the cardholder's home/current
          country when selecting `fraud_country`.
        - Amounts follow an "escalation" pattern: later transactions tend to be larger.

    Raises:
        KeyError: If required keys are missing from `card` (e.g., "cc_num", "account_id").
    """
    eligible = _eligible_countries(merchants_by_country)
    if not eligible:
        raise RuntimeError("Need at least 1 country with merchants to generate chain_attack_events.")
    n = rng.randint(size_min, size_max)
    dur = rng.randint(duration_min, duration_max)
    t0 = start

    # Fraud country: often not home
    fraud_country = rng.choice(eligible)
    txs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    # escalating amount pattern: small test, then larger
    base = rng.uniform(1.0, 10.0)
    for j in range(n):
        frac = (j / max(1, n - 1))
        # exponential-ish escalation
        amount = round(base * (1.0 + (frac * frac) * rng.uniform(20, 120)), 2)
        # Choose timestamps across the burst
        dt = t0 + timedelta(seconds=int((dur * 60) * frac)) + timedelta(seconds=rng.randint(0, 30))
        lat, lon = sample_lat_lon(rng, fraud_country)
        ip = sample_ip(rng, fraud_country)

        # pick a merchant in that country
        mlist = merchants_by_country[fraud_country]
        merchant = rng.choice(mlist)

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
    """
    Generate a synthetic "geographic fraud" pair (impossible travel) for a single card.

    A geographic fraud pair consists of two card-present (POS) transactions occurring
    close together in time but in two different, far-apart countries. This creates an
    "impossible travel" signal: the same card appears to be used in distant locations
    within a short time window.

    The function:
      - Selects two distinct countries.
      - Sets timestamps at `start` and `start + gap` minutes (gap sampled uniformly).
      - Samples lat/lon and a synthetic IP address consistent with each country.
      - Picks merchants in each country from `merchants_by_country`.
      - Produces fraud-label records for BOTH transactions.

    Args:
        rng: Random number generator (pass a seeded instance for reproducibility).
        card: Card record containing at least "cc_num" and "account_id".
        merchants_by_country: Mapping from country code -> list of merchant records.
        start: Timestamp for the first transaction (naive datetimes are treated as UTC).
        gap_min: Minimum gap between transactions in minutes.
        gap_max: Maximum gap between transactions in minutes.

    Returns:
        A tuple (transactions, fraud_labels):
          - transactions: list of two POS transaction dicts with channel="pos",
            card_present=True, and ts (ISO UTC string).
          - fraud_labels: list of two label dicts keyed by t_id with explanation="geo_fraud".

    Raises:
        KeyError: If required keys are missing from `card` (e.g., "cc_num", "account_id").
    """
    eligible = _eligible_countries(merchants_by_country)
    if len(eligible) < 2:
        raise RuntimeError("Need merchants in at least 2 countries to generate geo_fraud_pair.")

    c1 = rng.choice(eligible)
    c2 = rng.choice([c for c in eligible if c != c1])
    
    gap = rng.randint(gap_min, gap_max)
    t1 = start
    t2 = start + timedelta(minutes=gap)

    txs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    for idx, (country, dt) in enumerate([(c1, t1), (c2, t2)]):
        lat, lon = sample_lat_lon(rng, country)
        ip = sample_ip(rng, country)
        mlist = merchants_by_country[country]  # guaranteed non-empty
        merchant = rng.choice(mlist)

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
