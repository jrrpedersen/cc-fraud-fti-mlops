from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .config import ReferenceConfig
from .entities import make_accounts, make_banks, make_cards, make_merchants
from .io import write_jsonl


def generate_reference_world(out_dir: Path, seed: int, cfg: dict | None = None) -> dict[str, Path]:
    """Generate slow-changing reference data.

    Outputs JSONL files and returns a mapping:
      { "banks": Path, "merchants": Path, "accounts": Path, "cards": Path }

    The concrete schema is intentionally simple but relational:
    - banks: bank_id, bank_name, country
    - merchants: merchant_id, merchant_name, country, category
    - accounts: account_id, bank_id, home_country
    - cards: cc_num, account_id
    """
    cfg_obj = ReferenceConfig(**(cfg or {}))
    rng = random.Random(seed)

    banks = make_banks(rng, cfg_obj.n_banks)
    merchants = make_merchants(rng, cfg_obj.n_merchants)
    accounts = make_accounts(rng, cfg_obj.n_accounts, banks=banks)
    cards = make_cards(rng, cfg_obj.n_cards, accounts=accounts)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "banks": out_dir / "banks.jsonl",
        "merchants": out_dir / "merchants.jsonl",
        "accounts": out_dir / "accounts.jsonl",
        "cards": out_dir / "cards.jsonl",
    }

    write_jsonl(banks, paths["banks"])
    write_jsonl(merchants, paths["merchants"])
    write_jsonl(accounts, paths["accounts"])
    write_jsonl(cards, paths["cards"])

    # lightweight sanity print
    print(f"[bootstrap] banks={len(banks)} merchants={len(merchants)} accounts={len(accounts)} cards={len(cards)}")
    return paths
