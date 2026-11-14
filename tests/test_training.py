"""End-to-end training on a small generated dataset: split, train, explain, register, promote."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from sentinelpay.training.data import time_split
from sentinelpay.training.explain import describe
from sentinelpay.utils.config import load_settings
from sentinelpay.utils.io import read_json

pytestmark = pytest.mark.spark


@pytest.fixture
def trained(spark, settings):
    """Pipeline on 20k rows, then one training run + promotion."""
    from sentinelpay.data import generator
    from sentinelpay.features import spark_features
    from sentinelpay.pipeline import bronze, silver
    from sentinelpay.training import promote, train

    s = replace(settings, data=replace(settings.data, sample_size=20_000))
    generator.run(s, ["none"])
    bronze.run(spark, s)
    silver.run(spark, s)
    spark_features.run(spark, s)
    metrics = train.run(s)
    decision = promote.run(s)
    return s, metrics, decision


def test_time_split_is_ordered_and_disjoint():
    ts = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    df = pd.DataFrame({"ts": ts, "is_fraud": 0})
    cfg = replace(load_settings().model, warmup_days=10, validation_days=20, test_days=30)
    split = time_split(df, cfg)
    assert split.train["ts"].min() == ts[10]  # warm-up dropped
    assert split.train["ts"].max() < split.validation["ts"].min()
    assert split.validation["ts"].max() < split.test["ts"].min()
    assert len(split.test) == 30 and len(split.validation) == 20 and len(split.train) == 40


def test_reason_text_is_plain_english():
    assert describe("speed_kmh_from_prev_cp", 15800.4) == (
        "Implied travel speed of 15,800 km/h since the previous in-person purchase"
    )
    assert describe("is_new_device", 1) == "Device never seen on this card"
    assert describe("speed_kmh_from_prev_cp", None).startswith("Online purchase")


def test_training_produces_reports_and_beats_baselines(trained):
    s, metrics, _ = trained
    lgbm = metrics["models"]["lightgbm"]
    assert metrics["data_is_synthetic"] is True
    assert 0 < lgbm["pr_auc"] <= 1
    assert lgbm["pr_auc"] > metrics["models"]["logistic_regression"]["pr_auc"] - 0.05
    # The model must be cheaper than doing nothing.
    assert lgbm["at_threshold"]["expected_cost"] < metrics["no_model_cost"]
    for name in (
        "metrics.json",
        "test_predictions.parquet",
        "test_alerts.parquet",
        "promotion.json",
        "registry.json",
    ):
        assert pd.io.common.file_exists(s.paths.report(name)), name


def test_threshold_is_chosen_on_validation_not_test(trained):
    s, metrics, _ = trained
    # Reproduce: the stored threshold must minimise cost on VALIDATION scores.
    from sentinelpay.training import evaluate as ev
    from sentinelpay.training import registry
    from sentinelpay.training.data import load_gold

    split = time_split(load_gold(s), s.model)
    scorer = registry.load_scorer(s, registry.CHAMPION)
    val_scores = scorer.predict_proba(split.validation)
    expected = ev.best_threshold(
        split.validation["is_fraud"].to_numpy(),
        val_scores,
        s.model.cost_missed_fraud,
        s.model.cost_alert_review,
    )
    assert metrics["models"]["lightgbm"]["threshold"] == pytest.approx(expected)


def test_alerts_carry_reason_codes(trained):
    s, _, _ = trained
    alerts = pd.read_parquet(s.paths.report("test_alerts.parquet"))
    reasons = json.loads(alerts.iloc[0]["reasons"])
    assert 1 <= len(reasons) <= 3
    assert all(r["contribution"] > 0 and r["text"] for r in reasons)


def test_first_model_becomes_champion_and_identical_retrain_does_not(trained):
    from sentinelpay.training import promote, train

    s, _, decision = trained
    assert decision["promoted"] and decision["champion_version"] == 1

    train.run(s)  # deterministic: same data, same seed -> same model as v1
    again = promote.run(s)
    assert again["challenger_version"] == 2
    assert not again["promoted"]
    assert again["champion_version"] == 1

    state = read_json(s.paths.report("registry.json"))
    aliases = {v["version"]: v["aliases"] for v in state["versions"]}
    assert aliases == {2: ["challenger"], 1: ["champion"]}


def test_registered_model_scores_like_the_in_memory_one(trained):
    """The MLflow-loaded pyfunc must give identical scores (no serialisation drift)."""
    import mlflow

    from sentinelpay.training import registry
    from sentinelpay.training.data import load_gold

    s, _, _ = trained
    registry.setup(s)
    test = time_split(load_gold(s), s.model).test.head(200)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{s.mlflow.model_name}@champion")
    scorer = pyfunc.unwrap_python_model().scorer
    pd.testing.assert_series_equal(
        pd.Series(pyfunc.predict(test[scorer.feature_names])),
        pd.Series(scorer.predict_proba(test)),
    )
