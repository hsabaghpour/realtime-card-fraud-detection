"""Load the gold feature table and split it by time.

Why a time-based split? A random split puts transactions from the same week,
card and attack into both train and test, so the model is tested on things it
has effectively already seen, and it gets to learn from the future to
predict the past. Production is always "train on the past, predict the
future", so evaluation must be too.

    |-- warm-up --|------ train ------|-- validation --|---- test ----|
     dropped:       fit the models      early stopping,  final, untouched
     features are                       calibration,     numbers in the
     "cold"                             threshold choice README / GUI
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from deltalake import DeltaTable

from sentinelpay.features.registry import FEATURE_NAMES
from sentinelpay.utils.config import ModelSettings, Settings

KEEP = ("transaction_id", "card_id", "merchant_id", "ts", "is_fraud", "fraud_pattern")


def load_gold(settings: Settings) -> pd.DataFrame:
    """Gold features as pandas (500k x 26 fits comfortably in memory)."""
    columns = list(dict.fromkeys([*KEEP, *FEATURE_NAMES]))
    df = DeltaTable(settings.paths.gold).to_pandas(columns=columns)
    return df.sort_values(["ts", "transaction_id"], kind="stable").reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict[str, dict]:
        def info(d: pd.DataFrame) -> dict:
            return {
                "rows": len(d),
                "frauds": int(d["is_fraud"].sum()),
                "start": str(d["ts"].min().date()),
                "end": str(d["ts"].max().date()),
                "days": int(d["ts"].dt.normalize().nunique()),
            }

        return {
            "train": info(self.train),
            "validation": info(self.validation),
            "test": info(self.test),
        }


def time_split(df: pd.DataFrame, cfg: ModelSettings) -> Split:
    """Split by calendar day: [warm-up dropped] train | validation | test."""
    day = df["ts"].dt.normalize()
    first, last = day.min(), day.max()
    test_start = last - pd.Timedelta(days=cfg.test_days - 1)
    val_start = test_start - pd.Timedelta(days=cfg.validation_days)
    train_start = first + pd.Timedelta(days=cfg.warmup_days)
    if not train_start < val_start:
        raise ValueError("not enough history for warm-up + validation + test windows")
    return Split(
        train=df[(day >= train_start) & (day < val_start)].reset_index(drop=True),
        validation=df[(day >= val_start) & (day < test_start)].reset_index(drop=True),
        test=df[day >= test_start].reset_index(drop=True),
    )
