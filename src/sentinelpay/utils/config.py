"""Configuration loading.

Resolution order (later wins):
1. ``configs/base.yaml``
2. ``configs/<env>.yaml`` where env = ``SENTINELPAY_ENV`` (default ``local``)
3. Environment variables: ``SAMPLE_SIZE``, ``SEED``, ``SENTINELPAY_DATA_ROOT``,
   ``SENTINELPAY_REPORTS_DIR``
4. An explicit ``overrides`` dict (used by tests and the CLI)

The result is an immutable, typed ``Settings`` object, so the rest of the code
never touches raw dicts or environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(os.environ.get("SENTINELPAY_CONFIG_DIR", PROJECT_ROOT / "configs"))


@dataclass(frozen=True)
class DataSettings:
    """Synthetic data generation settings."""

    sample_size: int
    start_date: str
    history_days: int
    production_days: int
    fraud_rate: float
    dirty_rate: float
    label_delay_median_days: float
    chargeback_window_days: int


@dataclass(frozen=True)
class SparkSettings:
    """Local Spark session settings (ignored on Databricks except partitions)."""

    master: str
    driver_memory: str
    shuffle_partitions: int
    ui_enabled: bool


@dataclass(frozen=True)
class Paths:
    """All storage locations, derived from two roots so they move together."""

    data_root: str
    reports_dir: str

    @property
    def landing_history(self) -> str:
        """Raw parquet drop zone for the history period (the 'upstream' feed)."""
        return f"{self.data_root}/landing/history"

    def landing_production(self, scenario: str) -> str:
        """Raw parquet drop zone for one production-period drift scenario."""
        return f"{self.data_root}/landing/production/scenario={scenario}"

    @property
    def bronze(self) -> str:
        return f"{self.data_root}/delta/bronze_transactions"

    @property
    def silver(self) -> str:
        return f"{self.data_root}/delta/silver_transactions"

    @property
    def quarantine(self) -> str:
        return f"{self.data_root}/delta/silver_quarantine"

    @property
    def gold(self) -> str:
        return f"{self.data_root}/delta/gold_features"

    def report(self, name: str) -> str:
        """Path of a JSON report under the reports directory."""
        return f"{self.reports_dir}/{name}"


@dataclass(frozen=True)
class Settings:
    """Top-level, immutable application settings."""

    env: str
    seed: int
    data: DataSettings
    spark: SparkSettings
    paths: Paths


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _env_overrides() -> dict[str, Any]:
    """Map the supported environment variables onto config keys."""
    out: dict[str, Any] = {}
    if v := os.environ.get("SAMPLE_SIZE"):
        out.setdefault("data", {})["sample_size"] = int(v)
    if v := os.environ.get("SEED"):
        out["seed"] = int(v)
    if v := os.environ.get("SENTINELPAY_DATA_ROOT"):
        out.setdefault("paths", {})["data_root"] = v
    if v := os.environ.get("SENTINELPAY_REPORTS_DIR"):
        out.setdefault("paths", {})["reports_dir"] = v
    return out


def _resolve(path: str) -> str:
    """Make relative local paths absolute (relative to the project root).

    Absolute paths (including Databricks ``/Volumes/...``) and URIs are left alone.
    """
    if "://" in path or Path(path).is_absolute():
        return path
    return str((PROJECT_ROOT / path).resolve())


def load_settings(env: str | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load settings for ``env`` (default: ``$SENTINELPAY_ENV`` or ``local``)."""
    env = env or os.environ.get("SENTINELPAY_ENV", "local")
    raw = _read_yaml(CONFIG_DIR / "base.yaml")
    raw = _deep_merge(raw, _read_yaml(CONFIG_DIR / f"{env}.yaml"))
    raw = _deep_merge(raw, _env_overrides())
    raw = _deep_merge(raw, overrides or {})

    return Settings(
        env=env,
        seed=int(raw["seed"]),
        data=DataSettings(**raw["data"]),
        spark=SparkSettings(**raw["spark"]),
        paths=Paths(
            data_root=_resolve(raw["paths"]["data_root"]),
            reports_dir=_resolve(raw["paths"]["reports_dir"]),
        ),
    )
