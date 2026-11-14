"""Generator tests: determinism, schema, realism and drift (no Spark needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelpay.data import generator as g
from sentinelpay.utils.geo import haversine_km


@pytest.fixture
def history(settings):
    df, summary = g.generate_period(settings, "history")
    return df, summary


@pytest.fixture
def clean_history(history):
    df, _ = history
    return df[~df["transaction_id"].str.startswith("HX")].drop_duplicates("transaction_id")


def test_same_seed_same_data(settings):
    a, _ = g.generate_period(settings, "history")
    b, _ = g.generate_period(settings, "history")
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_different_data(settings):
    from dataclasses import replace

    a, _ = g.generate_period(settings, "history")
    b, _ = g.generate_period(replace(settings, seed=settings.seed + 1), "history")
    assert not a["amount"].equals(b["amount"])


def test_schema_and_types(history):
    df, _ = history
    assert tuple(df.columns) == g.RAW_COLUMNS
    assert str(df["ts"].dtype).startswith("datetime64") and df["ts"].dt.tz is not None
    assert df["is_fraud"].dtype == np.int32
    assert set(df["channel"]) == {"online", "card_present"}


def test_fraud_rate_and_all_patterns_present(history, settings):
    _, summary = history
    # Every pattern gets at least one event, so on this tiny test sample a
    # single compromised-merchant ring (~27 rows) pushes the rate above the
    # configured 0.8%. At the default 500k rows it lands at ~0.78%.
    assert 0.004 <= summary["fraud_rate"] <= 0.02
    assert set(summary["fraud_by_pattern"]) == set(g.FRAUD_PATTERNS)


def test_rows_within_period(clean_history, settings):
    start = pd.Timestamp(settings.data.start_date, tz="UTC")
    end = start + pd.Timedelta(days=settings.data.history_days)
    assert clean_history["ts"].min() >= start
    assert clean_history["ts"].max() < end


def test_labels_arrive_after_transaction(clean_history, settings):
    delay = clean_history["label_available_ts"] - clean_history["ts"]
    assert (delay >= pd.Timedelta(days=1)).all()
    legit = clean_history["is_fraud"] == 0
    assert (delay[legit] == pd.Timedelta(days=settings.data.chargeback_window_days)).all()


def test_device_only_for_online(clean_history):
    online = clean_history["channel"] == "online"
    assert clean_history.loc[~online, "device_id"].isna().all()
    assert clean_history.loc[online, "device_id"].notna().all()


def test_card_testing_is_tiny_then_large(clean_history):
    ct = clean_history[clean_history["fraud_pattern"] == "card_testing"]
    assert (ct["amount"] <= 3.0).mean() > 0.6
    assert ct["amount"].max() > 100


def test_impossible_travel_is_far_from_previous_location(clean_history):
    df = clean_history[clean_history["channel"] == "card_present"].sort_values(["card_id", "ts"])
    prev_lat = df.groupby("card_id")["merchant_lat"].shift()
    prev_lon = df.groupby("card_id")["merchant_lon"].shift()
    dist = haversine_km(prev_lat, prev_lon, df["merchant_lat"], df["merchant_lon"])
    first_hop = df["fraud_pattern"].eq("impossible_travel") & df.groupby("card_id")[
        "fraud_pattern"
    ].shift().ne("impossible_travel")
    assert (dist[first_hop] > 1000).mean() > 0.9


def test_night_burst_happens_at_night(clean_history):
    nb = clean_history[clean_history["fraud_pattern"] == "night_burst"]
    assert nb["ts"].dt.hour.between(1, 5).all()


def test_dirty_records_are_injected(history, settings):
    df, summary = history
    dirty = summary["dirty_records"]
    assert dirty["duplicates"] > 0
    assert df["transaction_id"].duplicated().sum() == dirty["duplicates"]
    assert df["card_id"].isna().sum() == dirty["null_card_id"]
    assert (df["amount"] <= 0).sum() == dirty["non_positive_amount"]


def test_history_rejects_drift_scenario(settings):
    with pytest.raises(ValueError):
        g.generate_period(settings, "history", scenario="spending_shift")


def test_unknown_scenario_rejected(settings):
    with pytest.raises(ValueError):
        g.generate_period(settings, "production", scenario="nope")


@pytest.fixture
def production(settings):
    pop = g.build_population(settings.data.sample_size, settings.data.history_days, settings.seed)
    return {
        s: g.generate_period(settings, "production", s, population=pop) for s in g.DRIFT_SCENARIOS
    }


def _legit_median_amount(df: pd.DataFrame) -> float:
    # Median of legit rows: robust to a single huge fraud on a tiny sample.
    return float(df.loc[(df["is_fraud"] == 0) & (df["amount"] > 0), "amount"].median())


def test_production_follows_history(settings):
    df, _ = g.generate_period(settings, "production", "none")
    start = pd.Timestamp(settings.data.start_date, tz="UTC") + pd.Timedelta(
        days=settings.data.history_days
    )
    assert df["ts"].min() >= start


def test_spending_shift_raises_amounts(production):
    shifted = _legit_median_amount(production["spending_shift"][0])
    baseline = _legit_median_amount(production["none"][0])
    assert shifted > 1.25 * baseline


def test_merchant_mix_shift_raises_online_share(production):
    shifted = production["merchant_mix_shift"][1]["online_share"]
    assert shifted > 1.5 * production["none"][1]["online_share"]


def test_new_fraud_pattern_appears_only_in_that_scenario(production):
    assert g.NEW_FRAUD_PATTERN in production["new_fraud_pattern"][1]["fraud_by_pattern"]
    assert g.NEW_FRAUD_PATTERN not in production["none"][1]["fraud_by_pattern"]


def test_run_writes_landing_and_summary(settings):
    from pathlib import Path

    summary = g.run(settings, ["none"])
    assert Path(settings.paths.landing_history, "transactions.parquet").exists()
    assert Path(settings.paths.landing_production("none"), "transactions.parquet").exists()
    assert Path(settings.paths.report("generation_summary.json")).exists()
    assert summary["history"]["rows"] > settings.data.sample_size
