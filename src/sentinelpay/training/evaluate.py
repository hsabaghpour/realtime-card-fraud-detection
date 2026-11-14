"""Evaluation metrics for a rare-event classifier, plus cost-based thresholds.

Why not accuracy? With 0.8% fraud, "nothing is ever fraud" is 99.2% accurate
and useless. Fraud teams care about:

* **PR-AUC** (average precision): ranking quality focused on the rare class.
* **Recall at a fixed false-positive rate**: "if we may bother at most 0.5% of
  genuine customers, what share of fraud do we catch?"
* **Precision at top-k**: of the k riskiest transactions an analyst team can
  review, how many are really fraud?
* **Expected cost**: missed fraud x its cost + alerts x review cost. The alert
  threshold is chosen to minimise this, so it's a business decision, not 0.5.

All functions take plain arrays (``y`` in {0, 1}, ``scores`` higher = riskier)
and are unit-tested against hand-computed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FPR_TARGETS = (0.001, 0.005, 0.01)
TOP_K = (100, 500, 1000)


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Highest recall achievable while keeping the false-positive rate <= ``max_fpr``."""
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of fraud among the ``k`` highest-scored transactions."""
    k = min(k, len(y))
    top = np.argsort(-scores, kind="stable")[:k]
    return float(np.asarray(y)[top].mean()) if k else 0.0


def expected_cost(
    y: np.ndarray, alert: np.ndarray, cost_missed: float, cost_review: float
) -> float:
    """Missed frauds x cost_missed + every alert x cost_review."""
    y, alert = np.asarray(y).astype(bool), np.asarray(alert).astype(bool)
    missed = int((y & ~alert).sum())
    return float(missed * cost_missed + alert.sum() * cost_review)


@dataclass(frozen=True)
class CostCurve:
    """Expected cost for every distinct threshold (alert when score >= threshold)."""

    thresholds: np.ndarray  # descending; the first entry (inf) means "no alerts"
    alerts: np.ndarray
    true_positives: np.ndarray
    costs: np.ndarray

    @property
    def best(self) -> tuple[float, float]:
        i = int(np.argmin(self.costs))
        return float(self.thresholds[i]), float(self.costs[i])


def cost_curve(
    y: np.ndarray, scores: np.ndarray, cost_missed: float, cost_review: float
) -> CostCurve:
    """Cost at every candidate threshold in O(n log n) with cumulative sums.

    Sort by score descending; alerting on the top i transactions catches
    cumsum(y)[i] frauds. Thresholds are only placed where the score changes,
    so tied scores are always alerted together (as ``score >= t`` would).
    """
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, ys = scores[order], y[order]
    tp = np.cumsum(ys)
    last_of_tie = np.r_[s[1:] != s[:-1], True]  # positions where the next score differs
    alerts = np.flatnonzero(last_of_tie) + 1
    tp = tp[last_of_tie]
    thresholds = s[last_of_tie]

    positives = y.sum()
    alerts = np.r_[0, alerts]
    tp = np.r_[0.0, tp]
    thresholds = np.r_[np.inf, thresholds]
    costs = (positives - tp) * cost_missed + alerts * cost_review
    return CostCurve(thresholds=thresholds, alerts=alerts, true_positives=tp, costs=costs)


def best_threshold(
    y: np.ndarray, scores: np.ndarray, cost_missed: float, cost_review: float
) -> float:
    """Threshold that minimises expected cost on (validation) data."""
    return cost_curve(y, scores, cost_missed, cost_review).best[0]


def operating_point(
    y: np.ndarray, alert: np.ndarray, cost_missed: float, cost_review: float, n_days: float
) -> dict[str, Any]:
    """Confusion matrix and business numbers for one set of alert decisions."""
    y, alert = np.asarray(y).astype(bool), np.asarray(alert).astype(bool)
    tp = int((y & alert).sum())
    fp = int((~y & alert).sum())
    fn = int((y & ~alert).sum())
    tn = int((~y & ~alert).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "alerts": tp + fp,
        "alerts_per_day": (tp + fp) / n_days,
        "expected_cost": expected_cost(y, alert, cost_missed, cost_review),
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Threshold-free metrics of how well the scores rank fraud above legit."""
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_fpr": {f"{t:.1%}": recall_at_fpr(y, scores, t) for t in FPR_TARGETS},
        "precision_at_k": {str(k): precision_at_k(y, scores, k) for k in TOP_K},
    }


def recall_by_pattern(patterns: np.ndarray, y: np.ndarray, alert: np.ndarray) -> dict[str, float]:
    """Share of each fraud pattern's transactions that were alerted."""
    patterns, y, alert = np.asarray(patterns), np.asarray(y).astype(bool), np.asarray(alert)
    out = {}
    for p in sorted(set(patterns[y])):
        mask = y & (patterns == p)
        out[str(p)] = float(alert[mask].mean())
    return out
