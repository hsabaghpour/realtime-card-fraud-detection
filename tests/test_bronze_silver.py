"""Bronze ingestion and silver validation / quarantine / de-duplication."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import functions as F

from sentinelpay.pipeline import bronze, silver
from sentinelpay.pipeline.schemas import BRONZE_SCHEMA
from sentinelpay.utils.io import read_json

pytestmark = pytest.mark.spark

TS = datetime(2025, 3, 1, 12, 0, 0)
INGEST_OLD = datetime(2025, 3, 2, 0, 0, 0)
INGEST_NEW = datetime(2025, 3, 3, 0, 0, 0)


def _row(tx_id: str, **overrides):
    """A valid bronze row; override any field to break it."""
    row = {
        "transaction_id": tx_id,
        "card_id": "CARD0000001",
        "customer_id": "CUST000001",
        "merchant_id": "MER000001",
        "merchant_category": "grocery_pos",
        "amount": 42.5,
        "currency": "USD",
        "ts": TS,
        "customer_lat": 40.7,
        "customer_lon": -74.0,
        "merchant_lat": 40.71,
        "merchant_lon": -74.01,
        "channel": "card_present",
        "device_id": None,
        "is_fraud": 0,
        "fraud_pattern": "none",
        "label_available_ts": TS,
        "ingest_ts": INGEST_OLD,
        "source_file": "file:/landing/a.parquet",
    }
    row.update(overrides)
    return tuple(row[f.name] for f in BRONZE_SCHEMA.fields)


@pytest.fixture
def bronze_df(spark):
    rows = [
        _row("T1"),
        _row("T2"),
        # T2 re-sent later with a corrected amount: the newest copy must win.
        _row("T2", amount=99.0, ingest_ts=INGEST_NEW, source_file="file:/landing/b.parquet"),
        _row("T3", card_id=None),
        _row("T4", amount=-5.0),
        _row("T5", amount=0.0),
        _row("T6", merchant_lat=123.4),
        _row("T7", customer_lon=None),
        _row("T8", ts=None),
        _row("T9", channel="atm"),
        _row("T10", is_fraud=2),
        _row("T11", card_id=None, amount=-1.0),  # two problems at once
    ]
    return spark.createDataFrame(rows, BRONZE_SCHEMA)


def test_valid_rows_reach_silver_once(bronze_df):
    result = silver.build_silver(bronze_df)
    ids = sorted(r.transaction_id for r in result.silver.select("transaction_id").collect())
    assert ids == ["T1", "T2"]


def test_dedup_keeps_latest_ingested_copy(bronze_df):
    result = silver.build_silver(bronze_df)
    t2 = result.silver.filter(F.col("transaction_id") == "T2").first()
    assert t2.amount == 99.0


@pytest.mark.parametrize(
    ("tx_id", "reason"),
    [
        ("T3", "null_card_id"),
        ("T4", "non_positive_amount"),
        ("T5", "non_positive_amount"),
        ("T6", "invalid_merchant_coordinates"),
        ("T7", "invalid_customer_coordinates"),
        ("T8", "null_ts"),
        ("T9", "invalid_channel"),
        ("T10", "invalid_label"),
        ("T11", "null_card_id;non_positive_amount"),
    ],
)
def test_quarantine_reasons(bronze_df, tx_id, reason):
    q = silver.build_silver(bronze_df).quarantine
    row = q.filter(F.col("transaction_id") == tx_id).first()
    assert row is not None, f"{tx_id} should be quarantined"
    assert row.reject_reason == reason


def test_silver_adds_partition_date(bronze_df):
    row = silver.build_silver(bronze_df).silver.first()
    assert str(row.tx_date) == "2025-03-01"


def test_dq_metrics_counts(bronze_df):
    m = silver.dq_metrics(bronze_df, silver.build_silver(bronze_df))
    assert m["bronze_rows"] == 12
    assert m["quarantined_rows"] == 9
    assert m["duplicates_removed"] == 1
    assert m["silver_rows"] == 2
    assert m["quarantine_by_reason"]["non_positive_amount"] == 3
    assert m["quarantine_by_reason"]["null_card_id"] == 2


def test_end_to_end_generate_bronze_silver(spark, settings):
    """Tiny sample through generate -> bronze -> silver, plus idempotent re-ingest."""
    from sentinelpay.data import generator

    summary = generator.run(settings, ["none"])
    history = summary["history"]

    appended = bronze.run(spark, settings)
    assert appended == history["rows"]
    assert bronze.run(spark, settings) == 0  # same files again -> nothing appended

    metrics = silver.run(spark, settings)
    dirty = history["dirty_records"]
    assert metrics["silver_rows"] == history["clean_rows"]
    assert metrics["duplicates_removed"] == dirty["duplicates"]
    assert metrics["quarantined_rows"] == sum(v for k, v in dirty.items() if k != "duplicates")
    assert (
        read_json(settings.paths.report("dq_metrics.json"))["silver_rows"] == history["clean_rows"]
    )
    assert metrics["silver_ts_range"][0].startswith(settings.data.start_date)
