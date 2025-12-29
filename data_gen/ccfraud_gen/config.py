from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceConfig:
    n_banks: int = 25
    n_merchants: int = 2000
    n_accounts: int = 20000
    n_cards: int = 40000


@dataclass(frozen=True)
class TransactionConfig:
    start_utc: str            # ISO8601 string like "2025-12-29T00:00:00Z"
    duration_minutes: int = 60

    n_transactions: int = 50_000
    
    # Behavior realism
    p_home: float = 0.85
    p_neighbor: float = 0.10
    p_international: float = 0.05
    min_hours_between_country_changes: int = 6

    # Fraud realism (approximate; exact realized rate depends on grouping)
    fraud_rate: float = 0.0005
    chain_attack_ratio: float = 0.9  # remainder is geo fraud

    # Chain attacks
    chain_size_min: int = 5
    chain_size_max: int = 15
    chain_duration_min_minutes: int = 10
    chain_duration_max_minutes: int = 120

    # Geo fraud timing buckets
    geo_gap_min_minutes: int = 5
    geo_gap_max_minutes: int = 60