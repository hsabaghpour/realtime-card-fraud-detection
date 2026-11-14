"""Training pipeline: gold -> models -> evaluation -> MLflow -> reports.

Steps
1. Load gold features and split by time (train | validation | test).
2. Fit the baseline (logistic regression) and LightGBM (early stopping on
   validation, Platt calibration on validation).
3. Pick each model's alert threshold by minimising expected cost on
   VALIDATION, then report everything on the untouched TEST window.
4. Compare against a hand-written rules engine and "no model at all".
5. Explain: global SHAP importance and reason codes for the top test alerts.
6. Log params, metrics, plots and the model to MLflow; register it as the
   ``challenger``.
7. Write reports/ files the GUI reads (no MLflow needed to view results).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render plots to files, never open windows

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from sentinelpay.features.registry import FEATURE_NAMES  # noqa: E402
from sentinelpay.training import evaluate as ev  # noqa: E402
from sentinelpay.training import registry  # noqa: E402
from sentinelpay.training.data import load_gold, time_split  # noqa: E402
from sentinelpay.training.explain import global_importance, reason_codes  # noqa: E402
from sentinelpay.training.model import train_baseline, train_lightgbm  # noqa: E402
from sentinelpay.training.rules import RULES, rule_alerts  # noqa: E402
from sentinelpay.utils.config import Settings  # noqa: E402
from sentinelpay.utils.io import write_json  # noqa: E402
from sentinelpay.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

TOP_ALERTS = 300  # test alerts saved with reason codes for the GUI


def _evaluate_model(
    name: str,
    y_val: np.ndarray,
    s_val: np.ndarray,
    test: pd.DataFrame,
    s_test: np.ndarray,
    costs: tuple[float, float],
    n_days: float,
) -> dict[str, Any]:
    threshold = ev.best_threshold(y_val, s_val, *costs)
    y = test["is_fraud"].to_numpy()
    alert = s_test >= threshold
    return {
        "model": name,
        **ev.ranking_metrics(y, s_test),
        "threshold": threshold,
        "at_threshold": ev.operating_point(y, alert, *costs, n_days),
        "recall_by_pattern": ev.recall_by_pattern(test["fraud_pattern"].to_numpy(), y, alert),
    }


def _plots(test: pd.DataFrame, scores: dict[str, np.ndarray], lgbm: dict, importance: dict):
    y = test["is_fraud"].to_numpy()

    fig_pr, ax = plt.subplots(figsize=(6, 4.5))
    for (name, s), color in zip(scores.items(), ["#2a78d6", "#eb6834"], strict=False):
        p, r, _ = precision_recall_curve(y, s)
        auc = ev.ranking_metrics(y, s)["pr_auc"]
        ax.plot(r, p, color=color, lw=2, label=f"{name} (PR-AUC {auc:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    op = lgbm["at_threshold"]
    fig_cm, ax = plt.subplots(figsize=(4.5, 4))
    cm = np.array([[op["tn"], op["fp"]], [op["fn"], op["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black", fontsize=12)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["no alert", "alert"],
        yticklabels=["legit", "fraud"],
        title=f"LightGBM @ threshold {lgbm['threshold']:.3f}",
    )

    top = list(importance.items())[:15][::-1]
    fig_shap, ax = plt.subplots(figsize=(6, 5))
    ax.barh([k for k, _ in top], [v for _, v in top], color="#2a78d6")
    ax.set(xlabel="mean |SHAP| (log-odds)", title="Global feature importance (test)")
    fig_shap.tight_layout()
    return {"pr_curve.png": fig_pr, "confusion_matrix.png": fig_cm, "shap_importance.png": fig_shap}


def run(settings: Settings) -> dict[str, Any]:
    """Train, evaluate, log and register a challenger model. Returns the metrics."""
    cfg = settings.model
    costs = (cfg.cost_missed_fraud, cfg.cost_alert_review)

    split = time_split(load_gold(settings), cfg)
    train, val, test = split.train, split.validation, split.test
    describe = split.describe()
    n_test_days = describe["test"]["days"]
    log.info(
        "split: train %s rows (%s frauds) | validation %s | test %s",
        f"{len(train):,}",
        describe["train"]["frauds"],
        f"{len(val):,}",
        f"{len(test):,}",
    )

    # -- fit --------------------------------------------------------------------
    baseline = train_baseline(train, settings.seed)
    scorer, lgbm_info = train_lightgbm(train, val, cfg.lgbm, settings.seed)

    y_val = val["is_fraud"].to_numpy()
    scores_val = {
        "lightgbm": scorer.predict_proba(val),
        "logistic_regression": baseline.predict_proba(val[list(FEATURE_NAMES)])[:, 1],
    }
    scores_test = {
        "lightgbm": scorer.predict_proba(test),
        "logistic_regression": baseline.predict_proba(test[list(FEATURE_NAMES)])[:, 1],
    }

    # -- evaluate on the untouched test window ----------------------------------
    models = {
        name: _evaluate_model(
            name, y_val, scores_val[name], test, scores_test[name], costs, n_test_days
        )
        for name in scores_test
    }
    scorer.threshold = models["lightgbm"]["threshold"]

    y_test = test["is_fraud"].to_numpy()
    rules_alert = rule_alerts(test)
    models["rules"] = {
        "model": "rules",
        "rules": RULES,
        "at_threshold": ev.operating_point(y_test, rules_alert, *costs, n_test_days),
        "recall_by_pattern": ev.recall_by_pattern(
            test["fraud_pattern"].to_numpy(), y_test, rules_alert
        ),
    }
    no_model_cost = float(y_test.sum() * cfg.cost_missed_fraud)

    # -- explain ----------------------------------------------------------------
    sample = test.sample(min(cfg.shap_sample, len(test)), random_state=settings.seed)
    importance = global_importance(scorer, sample)
    top = test.assign(score=scores_test["lightgbm"]).nlargest(TOP_ALERTS, "score")
    top = top.assign(reasons=[json.dumps(r, default=str) for r in reason_codes(scorer, top)])

    # -- MLflow -----------------------------------------------------------------
    registry.setup(settings)
    lgbm = models["lightgbm"]
    with mlflow.start_run(run_name=f"lightgbm-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        mlflow.log_params(
            {
                **cfg.lgbm,
                **lgbm_info,
                "seed": settings.seed,
                "cost_missed_fraud": costs[0],
                "cost_alert_review": costs[1],
            }
        )
        mlflow.log_params({f"{k}_window": f"{v['start']}..{v['end']}" for k, v in describe.items()})
        mlflow.log_metrics(
            {
                "test_pr_auc": lgbm["pr_auc"],
                "test_roc_auc": lgbm["roc_auc"],
                "test_expected_cost": lgbm["at_threshold"]["expected_cost"],
                "test_precision": lgbm["at_threshold"]["precision"],
                "test_recall": lgbm["at_threshold"]["recall"],
                "threshold": lgbm["threshold"],
                "baseline_test_pr_auc": models["logistic_regression"]["pr_auc"],
                "rules_test_expected_cost": models["rules"]["at_threshold"]["expected_cost"],
                **{
                    f"test_recall_at_fpr_{k.rstrip('%')}pct": v
                    for k, v in lgbm["recall_at_fpr"].items()
                },
            }
        )
        mlflow.log_dict({"features": list(FEATURE_NAMES)}, "features.json")
        mlflow.log_dict(importance, "shap_importance.json")
        for fname, fig in _plots(test, scores_test, lgbm, importance).items():
            mlflow.log_figure(fig, fname)
            plt.close(fig)
        model_uri, version = registry.log_and_register(settings, scorer, test)
        run_id = run.info.run_id

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_is_synthetic": True,
        "split": describe,
        "costs": {"missed_fraud": costs[0], "alert_review": costs[1]},
        "models": models,
        "no_model_cost": no_model_cost,
        "savings": {
            "vs_no_model": no_model_cost - lgbm["at_threshold"]["expected_cost"],
            "vs_rules": models["rules"]["at_threshold"]["expected_cost"]
            - lgbm["at_threshold"]["expected_cost"],
        },
        "lightgbm_training": lgbm_info,
        "shap_importance": importance,
        "mlflow": {"run_id": run_id, "model_uri": model_uri, "model_version": version},
    }

    # -- reports for the GUI ------------------------------------------------------
    paths = settings.paths
    write_json(paths.report("metrics.json"), metrics)
    preds = test[["transaction_id", "ts", "amount", "is_fraud", "fraud_pattern"]].assign(
        score_lightgbm=scores_test["lightgbm"],
        score_logistic_regression=scores_test["logistic_regression"],
        rule_alert=rules_alert,
    )
    preds.to_parquet(paths.report("test_predictions.parquet"), index=False)
    top[
        [
            "transaction_id",
            "ts",
            "card_id",
            "merchant_id",
            "amount",
            "merchant_category",
            "is_fraud",
            "fraud_pattern",
            "score",
            "reasons",
        ]
    ].to_parquet(paths.report("test_alerts.parquet"), index=False)

    log.info(
        "model v%s: test PR-AUC %.3f (baseline %.3f) | recall %.1f%% at %.2f alerts/day | "
        "cost $%s vs rules $%s",
        version,
        lgbm["pr_auc"],
        models["logistic_regression"]["pr_auc"],
        100 * lgbm["at_threshold"]["recall"],
        lgbm["at_threshold"]["alerts_per_day"],
        f"{lgbm['at_threshold']['expected_cost']:,.0f}",
        f"{models['rules']['at_threshold']['expected_cost']:,.0f}",
    )
    return metrics
