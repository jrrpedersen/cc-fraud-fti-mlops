from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .geo import COUNTRIES, random_country_code # so entities can be assigned countries

# A fixed set of merchant categories
MERCHANT_CATEGORIES = [
    "grocery",
    "fuel",
    "restaurant",
    "ecommerce",
    "travel",
    "electronics",
    "pharmacy",
    "other",
]


def _id(prefix: str, i: int, width: int) -> str:
    """
    Helper to make deterministic, nicely formatted IDs.
    _id("bank", 7, 4) → "bank_0007",
    _id("m", 12, 6) → "m_000012", etc.
    """
    return f"{prefix}_{i:0{width}d}"


def make_banks(rng: random.Random, n_banks: int) -> list[dict[str, Any]]:
    """
    Creates: a list of banks, each assigned to a random country.
    """
    banks: list[dict[str, Any]] = []
    for i in range(1, n_banks + 1):
        country = rng.choice(COUNTRIES).code
        banks.append(
            {
                "bank_id": _id("bank", i, 4),
                "bank_name": f"Bank {i:04d}",
                "country": country,
            }
        )
    return banks


def make_merchants(rng: random.Random, n_merchants: int) -> list[dict[str, Any]]:
    # Every country in COUNTRIES gets at least one merchant as long as n_merchants >= len(COUNTRIES)
    merchants: list[dict[str, Any]] = []
    country_codes = [c.code for c in COUNTRIES]

    if n_merchants < len(country_codes):
        raise ValueError(
            f"n_merchants={n_merchants} is too small to cover all countries "
            f"({len(country_codes)}). Increase n_merchants or reduce COUNTRIES."
        )

    i = 1

    # 1) Guarantee at least one merchant per country
    for country in country_codes:
        category = rng.choice(MERCHANT_CATEGORIES)
        merchants.append(
            {
                "merchant_id": _id("m", i, 6),
                "merchant_name": f"Merchant {i:06d}",
                "country": country,
                "category": category,
            }
        )
        i += 1

    # 2) Add remaining merchants randomly
    while i <= n_merchants:
        country = rng.choice(country_codes)
        category = rng.choice(MERCHANT_CATEGORIES)
        merchants.append(
            {
                "merchant_id": _id("m", i, 6),
                "merchant_name": f"Merchant {i:06d}",
                "country": country,
                "category": category,
            }
        )
        i += 1

    return merchants


def make_accounts(
    rng: random.Random,
    n_accounts: int,
    banks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Creates accounts that belong to a bank and have a “home country”.
    NB: home country may differ from bank country for realism.
    """
    accounts: list[dict[str, Any]] = []
    for i in range(1, n_accounts + 1):
        bank = rng.choice(banks)
        home_country = random_country_code(rng)
        accounts.append(
            {
                "account_id": _id("acct", i, 8),
                "bank_id": bank["bank_id"],
                "home_country": home_country,
            }
        )
    return accounts


def make_cards(
    rng: random.Random,
    n_cards: int,
    accounts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """ Creates cards linked to accounts."""
    cards: list[dict[str, Any]] = []
    for i in range(1, n_cards + 1):
        acct = rng.choice(accounts)
        # Use a synthetic "cc_num" style ID
        cards.append(
            {
                "cc_num": _id("cc", i, 10),
                "account_id": acct["account_id"],
            }
        )
    return cards
