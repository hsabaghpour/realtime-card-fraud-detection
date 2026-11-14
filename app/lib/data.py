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


@st.cache_data(show_spinner="Loading transactions…")
def _read_delta(path: str, columns: tuple[str, ...] | None, stamp: float) -> pd.DataFrame:
    # ``stamp`` is part of the cache key, so a rebuilt table is reloaded.
    return DeltaTable(path).to_pandas(columns=list(columns) if columns else None)


def _delta(path: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    # Last commit time of the table. (Not the Delta version: `make demo` deletes
    # and recreates tables, so the version restarts at 0.)
    stamp = max(p.stat().st_mtime for p in Path(path, "_delta_log").glob("*.json"))
    return _read_delta(path, columns, stamp)


def silver() -> pd.DataFrame:
    df = _delta(paths().silver, tuple(SILVER_COLUMNS))
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def quarantine() -> pd.DataFrame:
    return _delta(paths().quarantine)


def gold_exists() -> bool:
    p = paths()
    return Path(p.gold).exists() and Path(p.report("feature_summary.json")).exists()


def gold() -> pd.DataFrame:
    from sentinelpay.features.registry import FEATURE_NAMES

    return _delta(paths().gold, ("transaction_id", "is_fraud", *FEATURE_NAMES))


MODEL_REPORTS = (
    "metrics.json",
    "test_predictions.parquet",
    "test_alerts.parquet",
    "registry.json",
    "promotion.json",
)


def model_outputs_exist() -> bool:
    return all(Path(paths().report(n)).exists() for n in MODEL_REPORTS)


@st.cache_data
def _read_parquet(path: str, mtime: float) -> pd.DataFrame:  # mtime: refresh after retraining
    return pd.read_parquet(path)


def report_table(name: str) -> pd.DataFrame:
    path = paths().report(name)
    return _read_parquet(path, Path(path).stat().st_mtime)


@st.cache_data
def _read_json(path: str, mtime: float | None = None) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def report(name: str) -> dict[str, Any]:
    path = paths().report(name)
    # Keyed on modification time so a re-run of the pipeline shows up on reload.
    return _read_json(path, Path(path).stat().st_mtime)
