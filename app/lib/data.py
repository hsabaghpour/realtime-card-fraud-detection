"""Cached loaders for pipeline outputs.

The GUI reads only what the pipeline wrote (Delta tables via the ``deltalake``
package, plus JSON reports). It never starts Spark, so it can be hosted on a
small free machine.

Each cached loader takes the path it reads as an argument, so the cache key
changes whenever the data location does (no stale results across datasets).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from deltalake import DeltaTable

from sentinelpay.utils.config import Paths, load_settings

SILVER_COLUMNS = [
    "transaction_id",
    "card_id",
    "merchant_id",
    "merchant_category",
    "amount",
    "ts",
    "customer_lat",
    "customer_lon",
    "merchant_lat",
    "merchant_lon",
    "channel",
    "device_id",
    "is_fraud",
    "fraud_pattern",
    "label_available_ts",
]


def paths() -> Paths:
    """Current output locations (cheap to compute, so not cached)."""
    return load_settings().paths


def outputs_exist() -> bool:
    p = paths()
    return Path(p.silver).exists() and Path(p.report("dq_metrics.json")).exists()


@st.cache_data(show_spinner="Loading silver transactions…")
def _read_delta(path: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return DeltaTable(path).to_pandas(columns=list(columns) if columns else None)


def silver() -> pd.DataFrame:
    df = _read_delta(paths().silver, tuple(SILVER_COLUMNS))
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def quarantine() -> pd.DataFrame:
    return _read_delta(paths().quarantine)


@st.cache_data
def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def report(name: str) -> dict[str, Any]:
    return _read_json(paths().report(name))
