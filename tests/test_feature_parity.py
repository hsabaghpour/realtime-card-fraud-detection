"""Training/serving parity: online (pandas) features == offline (Spark) features.

Runs the real pipeline on a small generated dataset, then recomputes the
features of a sample of transactions with the online function from raw
history alone, and requires every feature to match.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from deltalake import DeltaTable

from sentinelpay.features import spark_features
from sentinelpay.features.online_features import online_features
from sentinelpay.features.registry import FEATURE_NAMES

pytestmark = pytest.mark.spark

SAMPLE = 300


def _same(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return b is not None and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6)


@pytest.fixture
def pipeline(spark, settings):
    from sentinelpay.data import generator
    from sentinelpay.pipeline import bronze, silver

    generator.run(settings, ["none"])
    bronze.run(spark, settings)
    silver.run(spark, settings)
    spark_features.run(spark, settings)
    silver_pd = DeltaTable(settings.paths.silver).to_pandas()
    gold_pd = DeltaTable(settings.paths.gold).to_pandas()
    return silver_pd, gold_pd


def test_online_features_match_spark(pipeline):
    silver_pd, gold_pd = pipeline
    # Every fraud row (the cases that matter most) plus a random sample of the rest.
    fraud = gold_pd[gold_pd["is_fraud"] == 1]
    legit = gold_pd[gold_pd["is_fraud"] == 0].sample(SAMPLE, random_state=0)
    sample = pd.concat([fraud, legit])

    by_card = dict(tuple(silver_pd.groupby("card_id")))
    by_merchant = dict(tuple(silver_pd.groupby("merchant_id")))
    silver_by_id = silver_pd.set_index("transaction_id")

    mismatches = []
    for _, g in sample.iterrows():
        tx = silver_by_id.loc[g["transaction_id"]].to_dict()
        online = online_features(tx, by_card[g["card_id"]], by_merchant[g["merchant_id"]])
        for name in FEATURE_NAMES:
            spark_value = None if pd.isna(g[name]) else g[name]
            if not _same(spark_value, online[name]):
                mismatches.append((g["transaction_id"], name, spark_value, online[name]))

    assert not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:5]}"
    assert len(sample) > SAMPLE  # fraud rows were included
