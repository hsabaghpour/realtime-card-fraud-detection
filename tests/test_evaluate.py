"""Metric and threshold maths, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelpay.training import evaluate as ev
from sentinelpay.training.promote import decide
from sentinelpay.training.rules import rule_alerts

# 10 transactions, 3 frauds. Scores sorted high -> low for readability.
Y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])


def test_expected_cost_counts_misses_and_reviews():
    alert = S >= 0.65  # alerts: 0.9, 0.8, 0.7 -> 2 frauds caught, 1 missed
    assert ev.expected_cost(Y, alert, cost_missed=100, cost_review=10) == 1 * 100 + 3 * 10


def test_cost_curve_matches_brute_force():
    curve = ev.cost_curve(Y, S, cost_missed=100, cost_review=10)
    for t, c in zip(curve.thresholds, curve.costs, strict=True):
        assert c == ev.expected_cost(Y, t <= S, 100, 10)


def test_best_threshold_by_hand():
    # Alert top 4 -> catch all 3 frauds: 0 x 100 + 4 x 10 = 40 (cheapest).
    t = ev.best_threshold(Y, S, cost_missed=100, cost_review=10)
    assert t == pytest.approx(0.6)
    # If reviews are very expensive, alerting nothing is cheapest.
    assert ev.best_threshold(Y, S, cost_missed=1, cost_review=1000) == np.inf


def test_tied_scores_are_alerted_together():
    y = np.array([1, 0, 0])
    s = np.array([0.5, 0.5, 0.1])
    curve = ev.cost_curve(y, s, 100, 1)
    assert list(curve.alerts) == [0, 2, 3]  # never "1 of the 2 tied rows"


def test_precision_at_k():
    assert ev.precision_at_k(Y, S, 2) == 1.0
    assert ev.precision_at_k(Y, S, 4) == 0.75


def test_recall_at_fpr():
    # FPR <= 1/7 allows one false positive: alert 0.9, 0.8, 0.7, 0.6 -> recall 1.0
    assert ev.recall_at_fpr(Y, S, 1 / 7) == 1.0
    # FPR 0 allows no false positive: alert 0.9, 0.8 -> recall 2/3
    assert ev.recall_at_fpr(Y, S, 0.0) == pytest.approx(2 / 3)


def test_operating_point():
    op = ev.operating_point(Y, S >= 0.6, 100, 10, n_days=2)
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (3, 1, 0, 6)
    assert op["precision"] == 0.75 and op["recall"] == 1.0
    assert op["alerts_per_day"] == 2.0


def test_recall_by_pattern():
    patterns = np.array(["a", "b", "none", "a", "none", "none", "none", "none", "none", "none"])
    out = ev.recall_by_pattern(patterns, Y, S >= 0.75)
    assert out == {"a": 0.5, "b": 1.0}


@pytest.mark.parametrize(
    ("challenger", "champion", "promoted"),
    [
        ({"pr_auc": 0.9, "expected_cost": 100}, None, True),
        ({"pr_auc": 0.9, "expected_cost": 100}, {"pr_auc": 0.8, "expected_cost": 200}, True),
        ({"pr_auc": 0.9, "expected_cost": 300}, {"pr_auc": 0.8, "expected_cost": 200}, False),
        ({"pr_auc": 0.7, "expected_cost": 100}, {"pr_auc": 0.8, "expected_cost": 200}, False),
        ({"pr_auc": 0.8, "expected_cost": 200}, {"pr_auc": 0.8, "expected_cost": 200}, False),
    ],
)
def test_promotion_rule(challenger, champion, promoted):
    assert decide(challenger, champion)[0] is promoted


def test_rules_engine():
    df = pd.DataFrame(
        {
            "speed_kmh_from_prev_cp": [5000, None, 10, 10],
            "amount_to_avg_30d": [1, 1, 1, 1],
            "tx_count_1h": [0, 5, 0, 0],
            "is_night": [0, 0, 1, 1],
            "merchant_category": ["x", "x", "gambling_net", "grocery_pos"],
            "merchant_distinct_cards_1h": [0, 0, 0, 0],
            "is_new_device": [0, 0, 0, 0],
            "amount": [10, 10, 10, 10],
        }
    )
    assert rule_alerts(df).tolist() == [1, 1, 1, 0]


def test_readme_results_block_is_generated(tmp_path):
    from sentinelpay.reporting import END, START, render_results, update_readme

    model = {
        "pr_auc": 0.9,
        "recall_at_fpr": {"0.5%": 0.8},
        "at_threshold": {
            "recall": 0.9,
            "precision": 0.5,
            "alerts_per_day": 10.0,
            "expected_cost": 1000.0,
        },
        "recall_by_pattern": {"card_testing": 1.0},
    }
    rules = {"at_threshold": model["at_threshold"], "recall_by_pattern": {}}
    metrics = {
        "generated_at": "now",
        "split": {"test": {"start": "a", "end": "b", "rows": 10, "frauds": 1, "days": 31}},
        "models": {"lightgbm": model, "logistic_regression": model, "rules": rules},
        "costs": {"missed_fraud": 250, "alert_review": 5},
        "no_model_cost": 5000.0,
        "savings": {"vs_rules": 123.0},
    }
    block = render_results(metrics)
    assert block.startswith(START) and block.endswith(END)
    assert "| **LightGBM (champion)** | 0.900 | 80.0% |" in block

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold\n{END}\noutro\n")
    assert update_readme(readme, metrics)
    text = readme.read_text()
    assert "old" not in text and text.startswith("intro") and text.endswith("outro\n")
    assert "$123" in text
