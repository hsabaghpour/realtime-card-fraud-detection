"""The fraud models: a logistic-regression baseline and calibrated LightGBM.

Class imbalance
---------------
Only ~0.8% of transactions are fraud. Unweighted, a model can do well on its
loss by mostly ignoring fraud. LightGBM therefore up-weights fraud with
``scale_pos_weight = sqrt(#legit / #fraud)`` (about 11x). The square root is
a common middle ground: enough to focus on fraud without the full ~125x
weight, which makes trees overfit the few fraud rows.

Calibration
-----------
Up-weighting fraud inflates the raw probabilities (the model believes fraud
is ~11x more common than it is). Platt scaling fits a 1-D logistic
regression on the validation set's raw log-odds, mapping them back to honest
probabilities: a score of 0.3 then really means ~30% of such transactions are
fraud. Platt is a monotone transform, so ranking metrics (PR-AUC) are
unchanged; only the probability scale is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sentinelpay.features.registry import CATEGORICAL, FEATURE_NAMES

NUMERIC = [f for f in FEATURE_NAMES if f not in CATEGORICAL]


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Compress heavy tails (amounts, speeds) so a linear model can use them."""
    return np.sign(x) * np.log1p(np.abs(x))


def train_baseline(train: pd.DataFrame, seed: int) -> Pipeline:
    """Logistic regression: the simple, explainable model to beat."""
    numeric = make_pipeline(
        FunctionTransformer(_signed_log1p),
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    pre = ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )
    model = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            ),
        ]
    )
    model.fit(train[list(FEATURE_NAMES)], train["is_fraud"])
    return model


@dataclass
class FraudScorer:
    """Everything needed to score a transaction: booster + calibration + threshold.

    Deliberately free of MLflow so the same object can be used by the batch
    and streaming scorers, the GUI and tests. MLflow wraps it (see
    ``registry.py``).
    """

    booster: lgb.Booster
    categories: dict[str, list[str]]
    platt_a: float = 1.0
    platt_b: float = 0.0
    threshold: float = 0.5
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_names].copy()
        for col, cats in self.categories.items():
            # Unknown categories (e.g. a brand-new MCC) become missing, which
            # LightGBM handles, instead of crashing the scorer.
            X[col] = pd.Categorical(X[col], categories=cats)
        return X

    def raw_logit(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._frame(df), raw_score=True)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud probability for each row."""
        z = self.platt_a * self.raw_logit(df) + self.platt_b
        return 1.0 / (1.0 + np.exp(-z))

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        """Per-feature SHAP values (log-odds) for each row; last column = bias.

        LightGBM's ``pred_contrib`` is exact TreeSHAP (Lundberg et al.), the
        same algorithm as ``shap.TreeExplainer``, computed natively.
        """
        return self.booster.predict(self._frame(df), pred_contrib=True)


def train_lightgbm(
    train: pd.DataFrame, validation: pd.DataFrame, params: dict[str, Any], seed: int
) -> tuple[FraudScorer, dict[str, Any]]:
    """Fit LightGBM with early stopping on validation, then Platt-calibrate."""
    categories = {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL}
    scorer = FraudScorer(booster=None, categories=categories)  # type: ignore[arg-type]
    X_tr, X_va = scorer._frame(train), scorer._frame(validation)
    y_tr, y_va = train["is_fraud"].to_numpy(), validation["is_fraud"].to_numpy()

    pos_weight = float(np.sqrt((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    clf = lgb.LGBMClassifier(
        **params,
        scale_pos_weight=pos_weight,
        random_state=seed,
        n_jobs=4,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_X=(X_va,),
        eval_y=(y_va,),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, first_metric_only=False, verbose=False)],
    )
    scorer.booster = clf.booster_

    # Platt scaling on validation log-odds.
    z_va = scorer.raw_logit(validation).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(z_va, y_va)
    scorer.platt_a = float(platt.coef_[0, 0])
    scorer.platt_b = float(platt.intercept_[0])

    info = {
        "best_iteration": int(clf.best_iteration_ or clf.n_estimators),
        "scale_pos_weight": round(pos_weight, 3),
        "platt_a": scorer.platt_a,
        "platt_b": scorer.platt_b,
    }
    return scorer, info
