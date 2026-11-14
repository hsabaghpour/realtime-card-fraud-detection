"""A hand-written rules engine: the 'before machine learning' baseline.

Banks ran rules like these for decades. Including them answers the question a
manager will ask first: "how much better is the model than what we have?"
Each rule targets one known fraud pattern, using the same features as the
model, so the comparison is fair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISKY_NIGHT_CATEGORIES = ("gambling_net", "money_transfer")

RULES: dict[str, str] = {
    "impossible_travel": "in-person purchase implying travel faster than 900 km/h",
    "big_ticket": "amount more than 8x the card's 30-day average",
    "burst": "4 or more transactions on the card in the previous hour",
    "night_risk": "night-time gambling or money-transfer purchase",
    "merchant_swarm": "10+ different cards at the merchant in the previous hour",
    "new_device_big": "new device and amount over $500",
}


def rule_hits(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule (NaN features never fire a rule)."""
    return pd.DataFrame(
        {
            "impossible_travel": df["speed_kmh_from_prev_cp"].fillna(0) > 900,
            "big_ticket": df["amount_to_avg_30d"].fillna(0) > 8,
            "burst": df["tx_count_1h"] >= 4,
            "night_risk": (df["is_night"] == 1)
            & df["merchant_category"].isin(RISKY_NIGHT_CATEGORIES),
            "merchant_swarm": df["merchant_distinct_cards_1h"] >= 10,
            "new_device_big": (df["is_new_device"] == 1) & (df["amount"] > 500),
        },
        index=df.index,
    )


def rule_alerts(df: pd.DataFrame) -> np.ndarray:
    """1 if any rule fires."""
    return rule_hits(df).any(axis=1).to_numpy().astype(int)
