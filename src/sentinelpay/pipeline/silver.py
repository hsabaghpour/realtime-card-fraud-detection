"""Silver layer: validated, de-duplicated, typed transactions.

Steps:
1. **Validate** every row against explicit rules. A row that fails any rule
   goes to ``silver_quarantine`` with *all* of its reject reasons (e.g.
   ``null_card_id;non_positive_amount``) so data owners can fix the source.
2. **De-duplicate** the valid rows on ``transaction_id``, keeping the most
   recently ingested copy (re-sent messages are common in payment feeds).
3. **Write** silver and quarantine as Delta tables, plus data-quality metrics
   to ``reports/dq_metrics.json``.

The transformation (``build_silver``) is a pure DataFrame -> DataFrame
function, so it is unit-tested without touching storage.

Silver is rebuilt in full from bronze on each run, which is simple and
correct at this scale. At production scale this becomes an incremental
``MERGE INTO`` keyed on ``transaction_id`` (see docs/design_decisions.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from sentinelpay.pipeline.schemas import CHANNELS
from sentinelpay.utils.config import Settings
from sentinelpay.utils.io import write_json
from sentinelpay.utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("transaction_id", "card_id", "customer_id", "merchant_id", "ts")


def _valid_coordinates(lat: str, lon: str) -> Column:
    return (
        F.col(lat).isNotNull()
        & F.col(lon).isNotNull()
        & F.col(lat).between(-90, 90)
        & F.col(lon).between(-180, 180)
    )


def validation_rules() -> list[tuple[str, Column]]:
    """(reject_reason, condition that is TRUE when the row is invalid)."""
    rules: list[tuple[str, Column]] = [(f"null_{c}", F.col(c).isNull()) for c in REQUIRED_COLUMNS]
    rules += [
        ("non_positive_amount", F.col("amount").isNull() | (F.col("amount") <= 0)),
        ("invalid_customer_coordinates", ~_valid_coordinates("customer_lat", "customer_lon")),
        ("invalid_merchant_coordinates", ~_valid_coordinates("merchant_lat", "merchant_lon")),
        ("invalid_channel", F.col("channel").isNull() | ~F.col("channel").isin(*CHANNELS)),
        ("invalid_label", F.col("is_fraud").isNull() | ~F.col("is_fraud").isin(0, 1)),
    ]
    return rules


def add_reject_reason(df: DataFrame) -> DataFrame:
    """Add ``reject_reason``: ';'-joined failing rules, or '' for a valid row.

    Each condition is wrapped in ``coalesce(..., True)`` so a rule that
    evaluates to NULL (three-valued SQL logic) counts as a failure, never as
    a silent pass.
    """
    reasons = [
        F.when(F.coalesce(cond, F.lit(True)), F.lit(name)) for name, cond in validation_rules()
    ]
    return df.withColumn("reject_reason", F.concat_ws(";", *reasons))


def deduplicate(df: DataFrame) -> DataFrame:
    """Keep one row per ``transaction_id``: the latest ingested copy."""
    w = Window.partitionBy("transaction_id").orderBy(
        F.col("ingest_ts").desc(), F.col("source_file").desc()
    )
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


@dataclass
class SilverResult:
    silver: DataFrame
    quarantine: DataFrame


def build_silver(bronze: DataFrame) -> SilverResult:
    """Pure transformation: bronze -> (silver, quarantine)."""
    checked = add_reject_reason(bronze)
    quarantine = checked.filter(F.col("reject_reason") != "").withColumn(
        "quarantine_ts", F.current_timestamp()
    )
    silver = (
        deduplicate(checked.filter(F.col("reject_reason") == "").drop("reject_reason"))
        .withColumn("amount", F.round("amount", 2))
        .withColumn("merchant_category", F.lower(F.trim("merchant_category")))
        .withColumn("tx_date", F.to_date("ts"))
    )
    return SilverResult(silver=silver, quarantine=quarantine)


def dq_metrics(bronze: DataFrame, result: SilverResult) -> dict[str, Any]:
    """Data-quality summary written to reports/dq_metrics.json."""
    bronze_rows = bronze.count()
    silver_rows = result.silver.count()
    quarantined = result.quarantine.count()
    valid_rows = bronze_rows - quarantined

    by_reason = {
        r["reason"]: r["n"]
        for r in result.quarantine.select(F.explode(F.split("reject_reason", ";")).alias("reason"))
        .groupBy("reason")
        .agg(F.count("*").alias("n"))
        .orderBy("reason")
        .collect()
    }
    null_rates = (
        bronze.select(
            [F.round(F.avg(F.col(c).isNull().cast("double")), 6).alias(c) for c in bronze.columns]
        )
        .first()
        .asDict()
    )
    # Cast to string inside Spark (session tz = UTC). Collecting raw timestamps
    # would convert them to the driver machine's local time zone.
    stats = result.silver.agg(
        F.min("ts").cast("string").alias("min_ts"),
        F.max("ts").cast("string").alias("max_ts"),
        F.avg("is_fraud").alias("fraud_rate"),
    ).first()

    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bronze_rows": bronze_rows,
        "quarantined_rows": quarantined,
        "quarantine_by_reason": by_reason,
        "duplicates_removed": valid_rows - silver_rows,
        "silver_rows": silver_rows,
        "pass_rate": round(silver_rows / bronze_rows, 6) if bronze_rows else None,
        "null_rate_bronze": {k: v for k, v in null_rates.items() if v},
        "silver_ts_range": [str(stats["min_ts"]), str(stats["max_ts"])],
        "silver_fraud_rate": round(stats["fraud_rate"] or 0.0, 6),
    }


def run(spark: SparkSession, settings: Settings) -> dict[str, Any]:
    """Rebuild silver + quarantine from bronze and write DQ metrics."""
    paths = settings.paths
    bronze = spark.read.format("delta").load(paths.bronze).cache()
    result = build_silver(bronze)

    (
        result.silver.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(paths.silver)
    )
    (
        result.quarantine.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(paths.quarantine)
    )

    # Compute metrics from the written tables (cheap, and what downstream sees).
    written = SilverResult(
        silver=spark.read.format("delta").load(paths.silver),
        quarantine=spark.read.format("delta").load(paths.quarantine),
    )
    metrics = dq_metrics(bronze, written)
    bronze.unpersist()
    out = write_json(paths.report("dq_metrics.json"), metrics)
    log.info(
        "silver: %s rows, %s quarantined, %s duplicates removed -> %s",
        f"{metrics['silver_rows']:,}",
        f"{metrics['quarantined_rows']:,}",
        f"{metrics['duplicates_removed']:,}",
        out,
    )
    return metrics
