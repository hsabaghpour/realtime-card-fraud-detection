"""Feature correctness: point-in-time guarantees, window edges and each feature group."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sentinelpay.features import registry
from sentinelpay.features.spark_features import build_features
from tests.feature_helpers import LA, NYC, silver_df, tx

pytestmark = pytest.mark.spark

HOUR = 3600
DAY = 86_400


def features(spark, rows) -> pd.DataFrame:
    """Build gold for ``rows`` and return it indexed by transaction_id."""
    return build_features(silver_df(spark, rows)).toPandas().set_index("transaction_id")


# --------------------------------------------------------------------------- #
# Point-in-time guarantees
# --------------------------------------------------------------------------- #
def test_future_transactions_never_change_past_features(spark):
    past = [
        tx(0, amount=20),
        tx(600, amount=35, merchant="M2"),
        tx(2 * HOUR, amount=500, channel="online", device="D1", merchant="M3"),
        tx(3 * HOUR, amount=40, card="CARD2", merchant="M1"),
        tx(4 * HOUR, amount=60, is_fraud=1, label_delay_days=1),
    ]
    future = [
        tx(5 * HOUR, amount=9_999, merchant="M1", is_fraud=1, label_delay_days=0.001),
        tx(5 * HOUR, amount=1, card="CARD2", merchant="M2", where=LA),
        tx(30 * DAY, amount=70, merchant="M1"),
    ]
    before = features(spark, past)
    after = features(spark, past + future).loc[before.index]
    cols = list(registry.FEATURE_NAMES)
    pd.testing.assert_frame_equal(before[cols], after[cols], check_dtype=False)


def test_window_includes_exactly_one_hour_back_but_not_more(spark):
    rows = [
        tx(-HOUR - 1),  # 3601 s before: outside the 1h window
        tx(-HOUR),  # exactly 3600 s before: inside
        tx(-HOUR / 2),
        tx(0, tx_id="TARGET"),
    ]
    f = features(spark, rows).loc["TARGET"]
    assert f["tx_count_1h"] == 2
    assert f["tx_count_24h"] == 3


def test_same_second_transactions_do_not_see_each_other(spark):
    f = features(spark, [tx(0, tx_id="A"), tx(0, tx_id="B")])
    assert f.loc["A", "card_tx_count_prior"] == 0
    assert f.loc["B", "card_tx_count_prior"] == 0


def test_label_not_yet_available_is_not_used(spark):
    rows = [
        tx(0, merchant="M9", is_fraud=1, label_delay_days=10),
        tx(5 * DAY, merchant="M9", card="CARD2", tx_id="BEFORE_LABEL"),
        tx(11 * DAY, merchant="M9", card="CARD3", tx_id="AFTER_LABEL"),
    ]
    f = features(spark, rows)
    p, k = registry.MERCHANT_RATE_PRIOR, registry.MERCHANT_RATE_STRENGTH
    rate = f["merchant_fraud_rate_prior"]
    assert rate["BEFORE_LABEL"] == pytest.approx((0 + p * k) / (1 + k))  # fraud not yet known
    assert rate["AFTER_LABEL"] == pytest.approx((1 + p * k) / (2 + k))  # now it is


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def test_velocity_sums_and_distinct_merchants(spark):
    rows = [
        tx(-2 * DAY, amount=1000, merchant="OLD"),
        tx(-3 * HOUR, amount=10, merchant="A"),
        tx(-2 * HOUR, amount=20, merchant="B"),
        tx(-30 * 60, amount=30, merchant="B"),
        tx(0, amount=5, tx_id="TARGET"),
    ]
    f = features(spark, rows).loc["TARGET"]
    assert f["amount_sum_1h"] == pytest.approx(30)
    assert f["amount_sum_24h"] == pytest.approx(60)
    assert f["amount_sum_7d"] == pytest.approx(1060)
    assert f["distinct_merchants_24h"] == 2


def test_first_transaction_has_empty_history(spark):
    f = features(spark, [tx(0, tx_id="FIRST")]).loc["FIRST"]
    assert f["tx_count_7d"] == 0 and f["amount_sum_7d"] == 0
    assert pd.isna(f["amount_zscore"]) and pd.isna(f["amount_to_avg_30d"])
    assert pd.isna(f["seconds_since_prev_tx"]) and pd.isna(f["km_from_prev_cp"])
    assert f["is_new_merchant"] == 1 and f["is_new_category"] == 1


def test_amount_zscore_needs_two_prior_transactions(spark):
    rows = [tx(0, amount=10), tx(60, amount=30, tx_id="SECOND"), tx(120, amount=60, tx_id="THIRD")]
    f = features(spark, rows)
    assert pd.isna(f.loc["SECOND", "amount_zscore"])
    # history [10, 30]: mean 20, sample std 14.142
    assert f.loc["THIRD", "amount_zscore"] == pytest.approx((60 - 20) / 14.1421356, rel=1e-6)
    assert f.loc["THIRD", "amount_to_avg_30d"] == pytest.approx(3.0)


def test_impossible_travel_speed(spark):
    rows = [tx(0, where=NYC), tx(10 * 60, where=LA, tx_id="LA")]
    f = features(spark, rows).loc["LA"]
    assert f["km_from_prev_cp"] == pytest.approx(3936, rel=0.01)
    assert f["speed_kmh_from_prev_cp"] == pytest.approx(3936 * 6, rel=0.01)
    assert f["dist_from_home_km"] == pytest.approx(3936, rel=0.01)


def test_geo_features_ignore_online_transactions(spark):
    rows = [
        tx(0, where=NYC),
        tx(60, where=LA, channel="online", device="D1", tx_id="ONLINE"),
        tx(120, where=NYC, tx_id="BACK_IN_NYC"),
    ]
    f = features(spark, rows)
    assert pd.isna(f.loc["ONLINE", "dist_from_home_km"])
    assert pd.isna(f.loc["ONLINE", "speed_kmh_from_prev_cp"])
    # The online merchant's HQ in LA must not count as a physical location.
    assert f.loc["BACK_IN_NYC", "km_from_prev_cp"] == pytest.approx(0, abs=1e-6)


def test_novelty_flags(spark):
    rows = [
        tx(0, merchant="M1", category="grocery_pos"),
        tx(60, merchant="M1", category="grocery_pos", tx_id="SAME_MERCHANT"),
        tx(120, merchant="M2", category="grocery_pos", tx_id="NEW_MERCHANT"),
        tx(180, channel="online", device="D1", merchant="W", category="shopping_net", tx_id="D1A"),
        tx(240, channel="online", device="D1", merchant="W", category="shopping_net", tx_id="D1B"),
        tx(300, channel="online", device="D2", merchant="W", category="shopping_net", tx_id="D2"),
    ]
    f = features(spark, rows)
    assert f.loc["SAME_MERCHANT", "is_new_merchant"] == 0
    assert f.loc["NEW_MERCHANT", ["is_new_merchant", "is_new_category"]].tolist() == [1, 0]
    assert f.loc["D1A", ["is_new_device", "is_new_category"]].tolist() == [1, 1]
    assert f.loc["D1B", "is_new_device"] == 0
    assert f.loc["D2", "is_new_device"] == 1
    assert f.loc["SAME_MERCHANT", "is_new_device"] == 0  # card-present: no device


def test_time_features(spark):
    rows = [tx(-HOUR * 12 - 1800, tx_id="NIGHT"), tx(0, tx_id="NOON")]  # 23:30 Sun, 12:00 Mon
    f = features(spark, rows)
    assert f.loc["NOON", ["hour_of_day", "day_of_week", "is_night"]].tolist() == [12, 1, 0]
    assert f.loc["NIGHT", ["hour_of_day", "day_of_week"]].tolist() == [23, 7]
    assert f.loc["NOON", "seconds_since_prev_tx"] == 12 * HOUR + 1800
    early = features(spark, [tx(-9 * HOUR, tx_id="3AM")]).loc["3AM"]
    assert early["is_night"] == 1


def test_compromised_merchant_distinct_cards(spark):
    rows = [tx(-60 * i, card=f"C{i}", merchant="SHOP") for i in range(1, 6)]
    rows += [
        tx(-2 * HOUR, card="C_OLD", merchant="SHOP"),  # too old for the 1h window
        tx(0, card="C0", merchant="SHOP", tx_id="T"),
    ]
    f = features(spark, rows).loc["T"]
    assert f["merchant_distinct_cards_1h"] == 5
    assert f["merchant_tx_count_prior"] == 6


# --------------------------------------------------------------------------- #
# Registry contract
# --------------------------------------------------------------------------- #
def test_gold_columns_match_registry(spark):
    gold = build_features(silver_df(spark, [tx(0)]))
    expected = [*registry.ID_COLUMNS, *registry.LABEL_COLUMNS, *registry.FEATURE_NAMES]
    assert gold.columns == expected


def test_labels_are_never_features():
    leaky = {"is_fraud", "fraud_pattern", "label_available_ts", "transaction_id", "card_id"}
    assert not leaky & set(registry.FEATURE_NAMES)


def test_registry_entries_are_documented():
    names = registry.FEATURE_NAMES
    assert len(names) == len(set(names)), "duplicate feature names"
    for f in registry.FEATURES:
        assert f.description and f.business_meaning, f.name


def test_features_doc_is_up_to_date():
    doc = Path(__file__).resolve().parents[1] / "docs" / "features.md"
    assert doc.read_text() == registry.to_markdown() + "\n", "run `make docs`"
