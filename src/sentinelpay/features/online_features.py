"""Online (single-transaction) feature computation in pure Python/pandas.

Why a second implementation? At serving time we score ONE transaction in
milliseconds. Starting a Spark job per transaction is impossible, so real
systems compute features from a fast store of each card's and merchant's
recent history. That creates the classic **training/serving skew** risk: if
this code computes a feature even slightly differently from the Spark code
the model was trained on, the model silently degrades in production.

The defence is twofold:
1. Both implementations share every constant via ``registry.py``.
2. ``tests/test_feature_parity.py`` computes features both ways for a sample
   of real transactions and asserts they match.

Semantics are identical to ``spark_features.py``: only history rows with
``epoch <= t - 1`` are used, and only fraud labels with
``label_available_ts < t``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sentinelpay.features.registry import (
    AMOUNT_AVG_WINDOW_S,
    FEATURE_NAMES,
    MERCHANT_RATE_PRIOR,
    MERCHANT_RATE_STRENGTH,
    MIN_TRAVEL_HOURS,
    NIGHT_HOURS,
    WINDOWS_S,
)
from sentinelpay.utils.geo import haversine_km


def to_epoch(ts: pd.Timestamp | pd.Series) -> int | pd.Series:
    """Whole seconds since 1970 (floor), matching Spark's ``cast(ts as long)``."""
    if isinstance(ts, pd.Series):
        return (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    return int((pd.Timestamp(ts) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1))


def _none_if_nan(x: Any) -> Any:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def online_features(
    tx: Mapping[str, Any], card_history: pd.DataFrame, merchant_history: pd.DataFrame
) -> dict[str, Any]:
    """Compute every registry feature for one transaction.

    Args:
        tx: the transaction (silver columns; ``ts`` a UTC timestamp).
        card_history: earlier transactions of the same card (silver columns).
            Rows at or after ``tx['ts']`` are ignored, so passing the full card
            history is safe.
        merchant_history: transactions at the same merchant (needs ``card_id``,
            ``ts``, ``is_fraud``, ``label_available_ts``). Same filtering.

    Returns:
        ``{feature_name: value}`` for every name in the registry, in order.
    """
    t = to_epoch(tx["ts"])
    is_cp = tx["channel"] == "card_present"
    amount = float(tx["amount"])

    # Strictly-before filter: the point-in-time rule.
    ch = card_history.assign(epoch=to_epoch(card_history["ts"]))
    ch = ch[ch["epoch"] <= t - 1]
    mh = merchant_history.assign(epoch=to_epoch(merchant_history["ts"]))
    mh = mh[mh["epoch"] <= t - 1]

    f: dict[str, Any] = {
        "amount": amount,
        "is_card_present": int(is_cp),
        "merchant_category": tx["merchant_category"],
    }

    # -- velocity ------------------------------------------------------------
    for label, seconds in WINDOWS_S.items():
        w = ch[ch["epoch"] >= t - seconds]
        f[f"tx_count_{label}"] = len(w)
        f[f"amount_sum_{label}"] = float(w["amount"].sum())
    f["distinct_merchants_24h"] = int(
        ch.loc[ch["epoch"] >= t - WINDOWS_S["24h"], "merchant_id"].nunique()
    )

    # -- amount behaviour ----------------------------------------------------
    f["card_tx_count_prior"] = len(ch)
    std = ch["amount"].std(ddof=1) if len(ch) >= 2 else None  # sample std, like Spark
    f["amount_zscore"] = (
        (amount - ch["amount"].mean()) / std if std is not None and std > 0 else None
    )
    last_30d = ch.loc[ch["epoch"] >= t - AMOUNT_AVG_WINDOW_S, "amount"]
    f["amount_to_avg_30d"] = amount / last_30d.mean() if len(last_30d) else None

    # -- geography -----------------------------------------------------------
    f["dist_from_home_km"] = (
        float(
            haversine_km(
                tx["customer_lat"], tx["customer_lon"], tx["merchant_lat"], tx["merchant_lon"]
            )
        )
        if is_cp
        else None
    )
    prev_cp = ch[ch["channel"] == "card_present"]
    if is_cp and len(prev_cp):
        # Latest earlier in-person purchase; ties on epoch broken by the largest
        # (lat, lon), exactly like Spark's max(struct(epoch, lat, lon)).
        prev = prev_cp.sort_values(["epoch", "merchant_lat", "merchant_lon"]).iloc[-1]
        km = float(
            haversine_km(
                prev["merchant_lat"], prev["merchant_lon"], tx["merchant_lat"], tx["merchant_lon"]
            )
        )
        hours = (t - prev["epoch"]) / 3600.0
        f["km_from_prev_cp"] = km
        f["speed_kmh_from_prev_cp"] = km / max(hours, MIN_TRAVEL_HOURS)
    else:
        f["km_from_prev_cp"] = None
        f["speed_kmh_from_prev_cp"] = None

    # -- novelty -------------------------------------------------------------
    f["is_new_merchant"] = int(not (ch["merchant_id"] == tx["merchant_id"]).any())
    f["is_new_category"] = int(not (ch["merchant_category"] == tx["merchant_category"]).any())
    device = tx.get("device_id")
    has_device = not is_cp and device is not None and not pd.isna(device)
    f["is_new_device"] = int(not (ch["device_id"] == device).any()) if has_device else 0

    # -- time ----------------------------------------------------------------
    ts = pd.Timestamp(tx["ts"])
    f["hour_of_day"] = ts.hour
    f["day_of_week"] = ts.dayofweek + 1  # pandas 0 = Monday -> ISO 1 = Monday
    f["is_night"] = int(ts.hour in NIGHT_HOURS)
    f["seconds_since_prev_tx"] = float(t - ch["epoch"].max()) if len(ch) else None

    # -- merchant risk -------------------------------------------------------
    f["merchant_distinct_cards_1h"] = int(
        mh.loc[mh["epoch"] >= t - WINDOWS_S["1h"], "card_id"].nunique()
    )
    f["merchant_tx_count_prior"] = len(mh)
    label_epoch = to_epoch(mh["label_available_ts"]) if len(mh) else pd.Series(dtype="int64")
    known_frauds = int(((mh["is_fraud"] == 1) & (label_epoch <= t - 1)).sum()) if len(mh) else 0
    f["merchant_fraud_rate_prior"] = (
        known_frauds + MERCHANT_RATE_PRIOR * MERCHANT_RATE_STRENGTH
    ) / (len(mh) + MERCHANT_RATE_STRENGTH)

    return {name: _none_if_nan(_py(f[name])) for name in FEATURE_NAMES}


def _py(x: Any) -> Any:
    """NumPy scalars -> plain Python, so results compare and serialise cleanly."""
    return x.item() if isinstance(x, np.generic) else x
