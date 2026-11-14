"""Explainability: global feature importance and per-alert reason codes.

An analyst can't act on "score 0.93". They need *why*: "travelled 3,900 km
in 10 minutes; new device; 4 transactions in the last hour". Reason codes are
the top 3 features pushing this transaction's score *up* (positive SHAP
values), rendered in plain English. The same mechanism also satisfies
regulators who require adverse-action explanations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from sentinelpay.training.model import FraudScorer


def _money(v: float) -> str:
    return f"${v:,.2f}"


# feature -> function(value) -> plain-English reason
REASON_TEXT: dict[str, Callable[[Any], str]] = {
    "amount": lambda v: f"Amount of {_money(v)}",
    "is_card_present": lambda v: (
        "Card used in person" if v else "Online purchase (card not present)"
    ),
    "merchant_category": lambda v: f"Merchant category: {v}",
    "tx_count_1h": lambda v: f"{int(v)} transactions on this card in the past hour",
    "tx_count_24h": lambda v: f"{int(v)} transactions on this card in the past 24 hours",
    "tx_count_7d": lambda v: f"{int(v)} transactions on this card in the past 7 days",
    "amount_sum_1h": lambda v: f"{_money(v)} spent on this card in the past hour",
    "amount_sum_24h": lambda v: f"{_money(v)} spent on this card in the past 24 hours",
    "amount_sum_7d": lambda v: f"{_money(v)} spent on this card in the past 7 days",
    "distinct_merchants_24h": lambda v: f"{int(v)} different merchants in the past 24 hours",
    "card_tx_count_prior": lambda v: (
        f"Card has only {int(v)} earlier transactions"
        if v < 20
        else f"Card history of {int(v)} transactions"
    ),
    "amount_zscore": lambda v: (
        "Amount far above this card's usual spend"
        if v > 0
        else "Amount unusually low for this card"
    ),
    "amount_to_avg_30d": lambda v: f"Amount is {v:,.1f}x the card's 30-day average",
    "dist_from_home_km": lambda v: f"{v:,.0f} km from the cardholder's home",
    "km_from_prev_cp": lambda v: f"{v:,.0f} km from the previous in-person purchase",
    "speed_kmh_from_prev_cp": lambda v: (
        f"Implied travel speed of {v:,.0f} km/h since the previous in-person purchase"
    ),
    "is_new_merchant": lambda v: "First purchase at this merchant" if v else "Known merchant",
    "is_new_category": lambda v: (
        "First purchase in this merchant category" if v else "Familiar merchant category"
    ),
    "is_new_device": lambda v: "Device never seen on this card" if v else "Known device",
    "hour_of_day": lambda v: f"Made at {int(v):02d}:00",
    "day_of_week": lambda v: f"Day of week {int(v)}",
    "is_night": lambda v: "Night-time transaction (00:00-05:59)" if v else "Daytime transaction",
    "seconds_since_prev_tx": lambda v: (
        f"Only {v:,.0f} seconds after the previous transaction"
        if v < 120
        else f"Only {v / 60:,.0f} minutes after the previous transaction"
        if v < 2 * 3600
        else f"{v / 3600:,.1f} hours after the previous transaction"
    ),
    "merchant_distinct_cards_1h": lambda v: (
        f"{int(v)} different cards used at this merchant in the past hour"
    ),
    "merchant_tx_count_prior": lambda v: f"Merchant has {int(v)} earlier transactions",
    "merchant_fraud_rate_prior": lambda v: f"Merchant's known fraud rate is {v:.2%}",
}


# What a *missing* value means. Missing is informative (LightGBM learns from it),
# so the reason must say what it implies, not "not available".
MISSING_TEXT: dict[str, str] = {
    "dist_from_home_km": "Online purchase: no physical location to verify",
    "km_from_prev_cp": "Online purchase: no physical location to verify",
    "speed_kmh_from_prev_cp": "Online purchase: no physical location to verify",
    "amount_zscore": "Too little card history to judge whether the amount is normal",
    "amount_to_avg_30d": "No purchases on this card in the past 30 days",
    "seconds_since_prev_tx": "First transaction ever seen on this card",
}


def describe(feature: str, value: Any) -> str:
    """Plain-English reason for one feature value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING_TEXT.get(feature, f"{feature} not available")
    return REASON_TEXT.get(feature, lambda v: f"{feature} = {v}")(value)


def reason_codes(
    scorer: FraudScorer, df: pd.DataFrame, top_n: int = 3
) -> list[list[dict[str, Any]]]:
    """Top ``top_n`` risk-increasing features per row, with readable text."""
    contrib = scorer.contributions(df)[:, :-1]  # drop the bias column
    names = scorer.feature_names
    out = []
    for i in range(len(df)):
        row = df.iloc[i]
        reasons: list[dict[str, Any]] = []
        for j in np.argsort(-contrib[i]):
            if contrib[i, j] <= 0 or len(reasons) == top_n:
                break
            text = describe(names[j], row[names[j]])
            if any(r["text"] == text for r in reasons):
                # e.g. three geo features all saying "online purchase": merge them
                # into the first one so the analyst sees three *different* reasons.
                continue
            reasons.append(
                {
                    "feature": names[j],
                    "value": None if pd.isna(row[names[j]]) else row[names[j]],
                    "contribution": round(float(contrib[i, j]), 4),
                    "text": text,
                }
            )
        out.append(reasons)
    return out


def global_importance(scorer: FraudScorer, df: pd.DataFrame) -> dict[str, float]:
    """Mean |SHAP| per feature: how much each feature moves scores on average."""
    contrib = np.abs(scorer.contributions(df)[:, :-1]).mean(axis=0)
    ranked = sorted(zip(scorer.feature_names, contrib, strict=True), key=lambda kv: -kv[1])
    return {name: round(float(v), 5) for name, v in ranked}
