from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .geo import sample_ip, sample_lat_lon

def _ensure_utc(dt: datetime) -> datetime:
    # Treat naive as UTC; otherwise convert to UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def clamp_ts(ts: datetime, start: datetime, end: datetime) -> datetime:
    if ts < start:
        return start
    if ts >= end:
        return end - timedelta(seconds=1)
    return ts

def _fit_start_for_duration(window_start: datetime, window_end: datetime, duration_s: int) -> Optional[datetime]:
    """
    Returns a randomizable valid start range endpoint idea:
    If duration doesn't fit, return None.
    """
    latest_start = window_end - timedelta(seconds=max(1, duration_s))
    if latest_start <= window_start:
        return None
    return latest_start

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
    rng: random.Random,
    card: dict[str, Any],
    merchants_by_country: dict[str, list[dict[str, Any]]],
    window_start: datetime,
    window_end: datetime,
    duration_min: int,
    duration_max: int,
    size_min: int,
    size_max: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    window_start = _ensure_utc(window_start)
    window_end = _ensure_utc(window_end)

    eligible = _eligible_countries(merchants_by_country)
    if not eligible:
        raise RuntimeError("Need at least 1 country with merchants to generate chain_attack_events.")

    n = rng.randint(size_min, size_max)

    # include jitter in the “must fit” budget
    jitter_s = 30
    dur_min_s = duration_min * 60
    dur_max_s = duration_max * 60
    dur_s = rng.randint(dur_min_s, dur_max_s)

    latest_start = _fit_start_for_duration(window_start, window_end, dur_s + jitter_s)
    if latest_start is None:
        # Window too small for requested chain; return empty (caller can count created_fraud)
        return [], []

    # pick t0 so whole chain fits
    span_s = int((latest_start - window_start).total_seconds())
    t0 = window_start + timedelta(seconds=rng.randrange(max(1, span_s)))

    fraud_country = rng.choice(eligible)
    mlist = merchants_by_country[fraud_country]  # non-empty by eligible logic

    txs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    base_amt = rng.uniform(1.0, 10.0)

    # Use integer seconds for reproducible IDs
    t0_id = int(t0.timestamp())

    for j in range(n):
        frac = j / max(1, n - 1)
        amount = round(base_amt * (1.0 + (frac * frac) * rng.uniform(20, 120)), 2)

        dt = t0 + timedelta(seconds=int(dur_s * frac)) + timedelta(seconds=rng.randint(0, jitter_s))
        dt = clamp_ts(dt, window_start, window_end)

        # if clamping forces everything to the same last instant, stop the chain
        if dt >= window_end - timedelta(seconds=1):
            break

        lat, lon = sample_lat_lon(rng, fraud_country)
        ip = sample_ip(rng, fraud_country)
        merchant = rng.choice(mlist)

        tx_id = f"tx_f_chain_{card['cc_num']}_{t0_id}_{j}"
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
        labels.append({"t_id": tx_id, "cc_num": card["cc_num"], "explanation": "chain_attack", "ts": tx["ts"]})

    return txs, labels


def geo_fraud_pair(
    rng: random.Random,
    card: dict[str, Any],
    merchants_by_country: dict[str, list[dict[str, Any]]],
    window_start: datetime,
    window_end: datetime,
    gap_min: int,
    gap_max: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    window_start = _ensure_utc(window_start)
    window_end = _ensure_utc(window_end)

    eligible = _eligible_countries(merchants_by_country)
    if len(eligible) < 2:
        raise RuntimeError("Need merchants in at least 2 countries to generate geo_fraud_pair.")

    c1 = rng.choice(eligible)
    c2 = rng.choice([c for c in eligible if c != c1])

    gap_s = rng.randint(gap_min, gap_max) * 60
    latest_t1 = _fit_start_for_duration(window_start, window_end, gap_s + 1)
    if latest_t1 is None:
        return [], []

    span_s = int((latest_t1 - window_start).total_seconds())
    t1 = window_start + timedelta(seconds=rng.randrange(max(1, span_s)))
    t2 = t1 + timedelta(seconds=gap_s)

    # last defense
    t1 = clamp_ts(t1, window_start, window_end)
    t2 = clamp_ts(t2, window_start, window_end)

    txs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    t1_id = int(t1.timestamp())

    for idx, (country, dt) in enumerate([(c1, t1), (c2, t2)]):
        lat, lon = sample_lat_lon(rng, country)
        ip = sample_ip(rng, country)
        merchant = rng.choice(merchants_by_country[country])

        tx_id = f"tx_f_geo_{card['cc_num']}_{t1_id}_{idx}"
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
        labels.append({"t_id": tx_id, "cc_num": card["cc_num"], "explanation": "geo_fraud", "ts": tx["ts"]})

    return txs, labels

