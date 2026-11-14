"""Champion/challenger promotion.

Re-scores the current champion and the challenger on the SAME test window
(today's data, not whatever each was evaluated on when trained) and promotes
the challenger only if it beats the champion on both PR-AUC and expected
cost. The decision and the full registry state are written to reports/ so
the GUI can show them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sklearn.metrics import average_precision_score

from sentinelpay.training import evaluate as ev
from sentinelpay.training import registry
from sentinelpay.training.data import load_gold, time_split
from sentinelpay.training.model import FraudScorer
from sentinelpay.utils.config import Settings
from sentinelpay.utils.io import write_json
from sentinelpay.utils.logging import get_logger

log = get_logger(__name__)


def _score(scorer: FraudScorer, test, costs) -> dict[str, float]:
    y = test["is_fraud"].to_numpy()
    s = scorer.predict_proba(test)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "expected_cost": ev.expected_cost(y, s >= scorer.threshold, *costs),
        "threshold": scorer.threshold,
    }


def decide(challenger: dict[str, float], champion: dict[str, float] | None) -> tuple[bool, str]:
    """Promotion rule, kept pure so it can be unit-tested."""
    if champion is None:
        return True, "no champion yet: first model becomes champion"
    better_auc = challenger["pr_auc"] > champion["pr_auc"]
    cheaper = challenger["expected_cost"] < champion["expected_cost"]
    if better_auc and cheaper:
        return True, "challenger beats champion on PR-AUC and expected cost"
    reasons = []
    if not better_auc:
        reasons.append("PR-AUC not higher")
    if not cheaper:
        reasons.append("expected cost not lower")
    return False, "champion kept: " + " and ".join(reasons)


def run(settings: Settings) -> dict[str, Any]:
    client = registry.setup(settings)
    name = settings.mlflow.model_name
    costs = (settings.model.cost_missed_fraud, settings.model.cost_alert_review)

    challenger_v = registry.alias_version(settings, registry.CHALLENGER)
    if challenger_v is None:
        raise RuntimeError("no challenger registered; run `sentinelpay train` first")
    champion_v = registry.alias_version(settings, registry.CHAMPION)

    test = time_split(load_gold(settings), settings.model).test
    challenger = _score(registry.load_scorer(settings, registry.CHALLENGER), test, costs)
    champion = None
    if champion_v is not None and champion_v != challenger_v:
        champion = _score(registry.load_scorer(settings, registry.CHAMPION), test, costs)

    promote, reason = decide(challenger, champion)
    if champion_v == challenger_v:
        promote, reason = False, "challenger is already the champion"
    if promote:
        client.set_registered_model_alias(name, registry.CHAMPION, challenger_v)

    decision = {
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "challenger_version": challenger_v,
        "previous_champion_version": champion_v,
        "promoted": promote,
        "champion_version": challenger_v if promote else champion_v,
        "reason": reason,
        "challenger": challenger,
        "champion": champion,
        "test_window": [str(test["ts"].min().date()), str(test["ts"].max().date())],
    }
    write_json(settings.paths.report("promotion.json"), decision)
    write_json(settings.paths.report("registry.json"), registry.registry_state(settings))
    log.info(
        "promotion: %s -> champion is v%s (%s)",
        "PROMOTED" if promote else "kept",
        decision["champion_version"],
        reason,
    )
    return decision
