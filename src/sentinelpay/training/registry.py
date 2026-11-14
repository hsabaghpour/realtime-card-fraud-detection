"""MLflow integration: experiment setup, model logging and the model registry.

Champion / challenger
---------------------
Every training run registers a new model version and gives it the alias
``challenger``. The alias ``champion`` marks the version production scoring
uses. ``promote()`` re-scores BOTH on the same, current test window and moves
``champion`` only if the challenger is better on PR-AUC *and* on expected
cost. Retraining therefore can never silently make production worse, and
rolling back is just moving an alias.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

import sentinelpay  # noqa: E402
from sentinelpay.training.model import FraudScorer  # noqa: E402
from sentinelpay.utils.config import Settings  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)
# MLflow warns that pickled models can run code when loaded. True in general;
# here models are only ever loaded from our own registry (see design decisions).
logging.getLogger("mlflow.pyfunc").setLevel(logging.ERROR)

CHAMPION = "champion"
CHALLENGER = "challenger"


class FraudPyfunc(mlflow.pyfunc.PythonModel):
    """Thin MLflow wrapper so any MLflow tool (serving, batch UDFs) can call the scorer."""

    def __init__(self, scorer: FraudScorer) -> None:
        self.scorer = scorer

    def predict(self, context, model_input: pd.DataFrame, params=None):  # noqa: ANN001
        return self.scorer.predict_proba(model_input)


def setup(settings: Settings) -> MlflowClient:
    """Point MLflow at the configured store and make sure the experiment exists."""
    cfg = settings.mlflow
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_registry_uri(cfg.tracking_uri)
    if mlflow.get_experiment_by_name(cfg.experiment) is None:
        location = Path(cfg.artifact_root).as_uri() if cfg.artifact_root else None
        mlflow.create_experiment(cfg.experiment, artifact_location=location)
    mlflow.set_experiment(cfg.experiment)
    return MlflowClient()


def _requirements() -> list[str]:
    import lightgbm
    import numpy
    import sklearn

    return [
        f"lightgbm=={lightgbm.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={numpy.__version__}",
    ]


def log_and_register(
    settings: Settings, scorer: FraudScorer, example: pd.DataFrame
) -> tuple[str, int]:
    """Log the scorer as an MLflow model inside the active run; tag it ``challenger``."""
    client = MlflowClient()
    sample = example[scorer.feature_names].head(5)
    # Declare integer features as doubles in the signature: at serving time a
    # missing value turns an int column into float, which an int schema rejects.
    ints = sample.select_dtypes("integer").columns
    sample = sample.astype({c: "float64" for c in ints})
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=FraudPyfunc(scorer),
        signature=infer_signature(sample, scorer.predict_proba(sample)),
        input_example=sample,
        pip_requirements=_requirements(),
        code_paths=[str(Path(sentinelpay.__file__).parent)],
        registered_model_name=settings.mlflow.model_name,
    )
    version = int(info.registered_model_version)
    client.set_registered_model_alias(settings.mlflow.model_name, CHALLENGER, version)
    return info.model_uri, version


def load_scorer(settings: Settings, alias: str) -> FraudScorer | None:
    """The scorer behind ``alias`` in the registry, or None if the alias is unset."""
    setup(settings)
    try:
        model = mlflow.pyfunc.load_model(f"models:/{settings.mlflow.model_name}@{alias}")
    except MlflowException:
        return None
    return model.unwrap_python_model().scorer


def alias_version(settings: Settings, alias: str) -> int | None:
    try:
        mv = MlflowClient().get_model_version_by_alias(settings.mlflow.model_name, alias)
    except MlflowException:
        return None
    return int(mv.version)


def registry_state(settings: Settings) -> dict[str, Any]:
    """All versions with aliases and key test metrics (written for the GUI)."""
    client = MlflowClient()
    name = settings.mlflow.model_name
    # search_model_versions() doesn't populate aliases; the registered model does.
    by_version: dict[int, list[str]] = {}
    for alias, v in client.get_registered_model(name).aliases.items():
        by_version.setdefault(int(v), []).append(alias)
    versions = []
    for mv in client.search_model_versions(f"name='{name}'"):
        run = client.get_run(mv.run_id)
        m = run.data.metrics
        versions.append(
            {
                "version": int(mv.version),
                "aliases": sorted(by_version.get(int(mv.version), [])),
                "created": pd.Timestamp(mv.creation_timestamp, unit="ms", tz="UTC").isoformat(
                    timespec="seconds"
                ),
                "run_id": mv.run_id,
                "test_pr_auc": m.get("test_pr_auc"),
                "test_expected_cost": m.get("test_expected_cost"),
                "threshold": m.get("threshold"),
            }
        )
    versions.sort(key=lambda v: -v["version"])
    return {
        "model_name": name,
        "tracking_uri": settings.mlflow.tracking_uri,
        "champion": alias_version(settings, CHAMPION),
        "challenger": alias_version(settings, CHALLENGER),
        "versions": versions,
    }
