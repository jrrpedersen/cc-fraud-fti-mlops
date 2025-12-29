from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import TransactionConfig
from .geo import pick_next_country, sample_ip, sample_lat_lon
from .fraud import chain_attack_events, geo_fraud_pair
from .io import write_jsonl

def parse_start_utc(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _rand_amount(rng: random.Random) -> float:
    # Lognormal-ish distribution, clipped.
    amt = rng.lognormvariate(mu=3.4, sigma=0.75)
    amt = max(0.5, min(amt, 5000.0))
    return round(amt, 2)


def _channel_and_card_present(rng: random.Random, merchant_category: str) -> tuple[str, bool]:
    # Simple heuristic: ecommerce tends to be online; others mostly card-present.
    if merchant_category == "ecommerce":
        return "online", False
    if merchant_category in {"travel"}:
        # mix: bookings can be online
        if rng.random() < 0.35:
            return "online", False
        return "pos", True
    if merchant_category in {"fuel", "grocery", "restaurant", "pharmacy"}:
        return "pos", True
    # other: mixed
    if rng.random() < 0.2:
        return "online", False
    return "pos", True


def _build_indexes(world: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    accounts = {a["account_id"]: a for a in world["accounts"]}
    # cards already include account_id
    cards = {c["cc_num"]: c for c in world["cards"]}
    merchants_by_country: dict[str, list[dict[str, Any]]] = {}
    for m in world["merchants"]:
        merchants_by_country.setdefault(m["country"], []).append(m)
    return accounts, cards, merchants_by_country

def clamp_ts(ts, start, end):
    if ts < start:
        return start
    if ts >= end:
        return end - timedelta(seconds=1)
    return ts

def generate_transactions_and_labels(
    out_dir: Path,
    world: dict[str, list[dict[str, Any]]],
    seed: int,
    cfg: dict | None = None,
) -> dict[str, Path]:
    """Generate event-stream-like transactions + fraud labels using a stable world snapshot.

    Output files:
      - transactions.jsonl: event log
      - fraud_labels.jsonl: label stream (can be delayed later)

    Transaction schema (compact but realistic):
      t_id, cc_num, account_id, merchant_id, amount, currency,
      country, ip_address, card_present, channel, category, lat, lon, ts

    Fraud label schema:
      t_id, cc_num, explanation, ts
    """
    cfg_obj = TransactionConfig(**(cfg or {}))
    rng = random.Random(seed)

    accounts_by_id, cards_by_id, merchants_by_country = _build_indexes(world)

    # Determine time window
    start = parse_start_utc(cfg_obj.start_utc)
    end = start + timedelta(minutes=cfg_obj.duration_minutes)

    # Choose number of fraud transactions approximately.
    approx_fraud_txs = max(1, int(cfg_obj.n_transactions * cfg_obj.fraud_rate))
    n_base = max(0, cfg_obj.n_transactions - approx_fraud_txs)

    cards = list(cards_by_id.values())
    if not cards:
        raise ValueError("World has no cards; bootstrap world first.")

    # Decide how many base tx per card (roughly), then generate per-card timelines.
    n_cards = len(cards)
    mean_per_card = max(1, n_base // max(1, n_cards))
    # We'll sample tx counts from a light-tailed distribution to create variance.
    per_card_counts = {}
    remaining = n_base
    # pick a subset of active cards
    active_cards = rng.sample(cards, k=min(n_cards, max(1000, n_cards // 10)))
    for c in active_cards:
        # geometric-ish around mean
        k = min(200, max(0, int(rng.expovariate(1.0 / max(1.0, mean_per_card)))))
        per_card_counts[c["cc_num"]] = k
        remaining -= k
    # top up remaining by adding single tx across random active cards
    while remaining > 0:
        c = rng.choice(active_cards)
        per_card_counts[c["cc_num"]] = per_card_counts.get(c["cc_num"], 0) + 1
        remaining -= 1

    transactions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    tx_counter = 0

    # Per-card state for continuity
    last_ts: dict[str, datetime] = {}
    last_country: dict[str, str] = {}

    for cc_num, count in per_card_counts.items():
        card = cards_by_id[cc_num]
        acct = accounts_by_id[card["account_id"]]
        home = acct["home_country"]
        current_country = home
        last_country[cc_num] = current_country
        # pick count timestamps within window and sort
        offsets = sorted(rng.randint(0, max(1, int((end - start).total_seconds()))) for _ in range(count))
        prev = start
        for off in offsets:
            ts = start + timedelta(seconds=off)
            ts = clamp_ts(ts, start, end)
            if ts < prev:
                ts = prev
            # choose next country based on probabilities + neighbors
            next_country = pick_next_country(
                rng,
                home=home,
                current=current_country,
                p_home=cfg_obj.p_home,
                p_neighbor=cfg_obj.p_neighbor,
                p_international=cfg_obj.p_international,
            )
            if next_country != current_country:
                # enforce min gap between country changes
                min_gap = timedelta(hours=cfg_obj.min_hours_between_country_changes)
                if ts < prev + min_gap:
                    ts = prev + min_gap
            ts = clamp_ts(ts, start, end)
            # If we're at (or effectively at) the end of the window, stop generating more
            if ts >= end - timedelta(seconds=1):
                break
            current_country = next_country
            prev = ts

            mlist = merchants_by_country.get(current_country) or []
            merchant = rng.choice(mlist) if mlist else {"merchant_id": "m_unknown", "category": "other", "country": current_country}
            channel, card_present = _channel_and_card_present(rng, merchant.get("category", "other"))
            lat, lon = sample_lat_lon(rng, current_country)
            ip = sample_ip(rng, current_country)

            tx_id = f"tx_{seed}_{tx_counter:09d}"
            tx_counter += 1

            transactions.append(
                {
                    "t_id": tx_id,
                    "cc_num": card["cc_num"],
                    "account_id": card["account_id"],
                    "merchant_id": merchant["merchant_id"],
                    "amount": _rand_amount(rng),
                    "currency": "USD",
                    "country": current_country,
                    "ip_address": ip,
                    "card_present": card_present,
                    "channel": channel,
                    "category": merchant.get("category", "other"),
                    "lat": lat,
                    "lon": lon,
                    "ts": _iso(ts),
                }
            )
            last_ts[cc_num] = ts
            last_country[cc_num] = current_country

    # Inject fraud events (separate from base transactions for controllability)
    n_chain = int(approx_fraud_txs * cfg_obj.chain_attack_ratio)
    n_geo = max(0, approx_fraud_txs - n_chain)

    # For chain attacks, generate groups; for geo fraud, generate pairs.
    # We may slightly overshoot/undershoot approx_fraud_txs depending on group sizes.
    fraud_cards = rng.sample(cards, k=min(len(cards), max(1000, len(cards)//20)))

    # Chain attacks
    created_fraud = 0
    while created_fraud < n_chain:
        card = rng.choice(fraud_cards)
        # choose a start time somewhere in window
        base_dt = start + timedelta(seconds=rng.randint(0, max(1, int((end - start).total_seconds()))))
        txs, labs = chain_attack_events(
            rng=rng,
            card=card,
            merchants_by_country=merchants_by_country,
            start=base_dt,
            duration_min=cfg_obj.chain_duration_min_minutes,
            duration_max=cfg_obj.chain_duration_max_minutes,
            size_min=cfg_obj.chain_size_min,
            size_max=cfg_obj.chain_size_max,
        )
        transactions.extend(txs)
        labels.extend(labs)
        created_fraud += len(txs)

    # Geo fraud pairs
    for _ in range(max(1, n_geo // 2)):
        card = rng.choice(fraud_cards)
        base_dt = start + timedelta(seconds=rng.randint(0, max(1, int((end - start).total_seconds()))))
        txs, labs = geo_fraud_pair(
            rng=rng,
            card=card,
            merchants_by_country=merchants_by_country,
            start=base_dt,
            gap_min=cfg_obj.geo_gap_min_minutes,
            gap_max=cfg_obj.geo_gap_max_minutes,
        )
        transactions.extend(txs)
        labels.extend(labs)

    # Sort all transactions by timestamp for downstream processing realism
    transactions.sort(key=lambda r: r["ts"])

    out_dir.mkdir(parents=True, exist_ok=True)
    tx_path = out_dir / "transactions.jsonl"
    label_path = out_dir / "fraud_labels.jsonl"
    write_jsonl(transactions, tx_path)
    write_jsonl(labels, label_path)

    # lightweight sanity stats
    fraud_ids = {l["t_id"] for l in labels}
    realized_rate = len(fraud_ids) / max(1, len(transactions))
    print(
        f"[transactions] total={len(transactions)} labeled_fraud={len(fraud_ids)} "
        f"realized_fraud_rate~{realized_rate:.6f} (target={cfg_obj.fraud_rate})"
    )

    return {"transactions": tx_path, "fraud_labels": label_path}
