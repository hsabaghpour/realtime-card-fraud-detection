"""Helpers to build tiny hand-written silver tables for feature tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

from sentinelpay.pipeline.schemas import BRONZE_SCHEMA

T0 = datetime(2025, 3, 3, 12, 0, 0, tzinfo=UTC)  # a Monday
NYC = (40.71, -74.01)
LA = (34.05, -118.24)

_counter = 0


def tx(
    seconds: float = 0,
    *,
    card: str = "CARD1",
    merchant: str = "M1",
    category: str = "grocery_pos",
    amount: float = 50.0,
    channel: str = "card_present",
    device: str | None = None,
    where: tuple[float, float] = NYC,
    home: tuple[float, float] = NYC,
    is_fraud: int = 0,
    label_delay_days: float = 45,
    tx_id: str | None = None,
) -> dict:
    """One silver row at ``T0 + seconds``; override any attribute by keyword."""
    global _counter
    _counter += 1
    ts = T0 + timedelta(seconds=seconds)
    return {
        "transaction_id": tx_id or f"T{_counter:05d}",
        "card_id": card,
        "customer_id": f"CUST-{card}",
        "merchant_id": merchant,
        "merchant_category": category,
        "amount": float(amount),
        "currency": "USD",
        "ts": ts,
        "customer_lat": home[0],
        "customer_lon": home[1],
        "merchant_lat": where[0],
        "merchant_lon": where[1],
        "channel": channel,
        "device_id": device,
        "is_fraud": is_fraud,
        "fraud_pattern": "x" if is_fraud else "none",
        "label_available_ts": ts + timedelta(days=label_delay_days),
        "ingest_ts": T0,
        "source_file": "test",
    }


def silver_df(spark: SparkSession, rows: list[dict]) -> DataFrame:
    """Rows -> silver-shaped DataFrame.

    Timestamps travel as UTC strings and are parsed by Spark (session time zone
    UTC). Passing Python datetimes would make PySpark apply the test machine's
    local time zone.
    """
    ts_cols = [f.name for f in BRONZE_SCHEMA.fields if isinstance(f.dataType, TimestampType)]
    string_schema = StructType(
        [
            StructField(f.name, StringType()) if f.name in ts_cols else f
            for f in BRONZE_SCHEMA.fields
        ]
    )
    data = [
        tuple(
            r[f.name].strftime("%Y-%m-%d %H:%M:%S") if f.name in ts_cols else r[f.name]
            for f in BRONZE_SCHEMA.fields
        )
        for r in rows
    ]
    df = spark.createDataFrame(data, string_schema)
    for c in ts_cols:
        df = df.withColumn(c, F.to_timestamp(c))
    return df.withColumn("tx_date", F.to_date("ts"))
