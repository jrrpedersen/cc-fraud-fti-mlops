from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TransactionEvent:
    event_timestamp: str
    transaction_id: str
    card_id: str
    merchant_id: str
    amount: float
    currency: str
    category: str
    channel: str
    lat: float
    lon: float


CATEGORIES = ["grocery", "fuel", "restaurant", "ecommerce", "travel", "electronics", "pharmacy", "other"]
CHANNELS = ["pos", "online", "atm"]
CURRENCY = "USD"


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _rand_amount(rng: random.Random) -> float:
    """
    Lognormal-ish distribution for amounts, clipped.
    """
    # mean around ~30-50 depending on sigma; tweak later
    amount = rng.lognormvariate(mu=3.5, sigma=0.8)
    amount = max(0.5, min(amount, 5000.0))
    return round(amount, 2)


def _rand_lat_lon(rng: random.Random) -> tuple[float, float]:
    """
    Roughly within continental US bounds for demo purposes.
    """
    lat = rng.uniform(25.0, 49.0)
    lon = rng.uniform(-124.0, -66.0)
    return round(lat, 6), round(lon, 6)


def generate_transactions(
    n: int,
    start_utc: datetime,
    duration_minutes: int,
    seed: int,
    n_cards: int = 500,
    n_merchants: int = 200,
) -> Iterable[TransactionEvent]:
    rng = random.Random(seed)

    # Spread event times across the duration window
    for i in range(n):
        offset_seconds = rng.randint(0, max(1, duration_minutes * 60))
        ts = start_utc + timedelta(seconds=offset_seconds)

        card_id = f"card_{rng.randint(1, n_cards):06d}"
        merchant_id = f"m_{rng.randint(1, n_merchants):05d}"
        category = rng.choice(CATEGORIES)
        channel = rng.choice(CHANNELS)

        lat, lon = _rand_lat_lon(rng)
        amount = _rand_amount(rng)

        # transaction id: deterministic-ish but unique per run
        transaction_id = f"tx_{seed}_{i:09d}"

        yield TransactionEvent(
            event_timestamp=_iso(ts),
            transaction_id=transaction_id,
            card_id=card_id,
            merchant_id=merchant_id,
            amount=amount,
            currency=CURRENCY,
            category=category,
            channel=channel,
            lat=lat,
            lon=lon,
        )


def write_jsonl(events: Iterable[TransactionEvent], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev)) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic credit-card transactions (JSONL).")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--duration-minutes", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="transactions.jsonl")
    p.add_argument("--start-utc", type=str, default=None, help="ISO8601 UTC, e.g. 2025-01-01T00:00:00Z")
    p.add_argument("--n-cards", type=int, default=500)
    p.add_argument("--n-merchants", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_utc:
        s = args.start_utc.replace("Z", "+00:00")
        start = datetime.fromisoformat(s).astimezone(timezone.utc)
    else:
        # default: now (UTC) floored to minute
        now = datetime.now(timezone.utc)
        start = now.replace(second=0, microsecond=0)

    events = generate_transactions(
        n=args.rows,
        start_utc=start,
        duration_minutes=args.duration_minutes,
        seed=args.seed,
        n_cards=args.n_cards,
        n_merchants=args.n_merchants,
    )

    out_path = Path(args.out)
    write_jsonl(events, out_path)
    print(f"Wrote {args.rows} events to {out_path.resolve()}")


if __name__ == "__main__":
    main()