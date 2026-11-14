"""Point-in-time feature engineering in PySpark (silver -> gold).

Point-in-time rule
------------------
Every feature for a transaction at time ``t`` uses only rows with
``epoch <= t - 1`` (strictly before, at one-second resolution). This is done
with window frames ``rangeBetween(-W, -1)`` on the transaction's epoch
seconds. Two consequences worth knowing:

* A window "1h" covers ``[t - 3600, t - 1]``: a transaction exactly one hour
  earlier IS included, one at ``t - 3601`` is not.
* Two transactions on the same card in the same second don't see each other.
  That's deliberate: in a real-time system neither has been processed when
  the other arrives.

Label-based features (merchant fraud rate) additionally use only labels with
``label_available_ts < t``: a chargeback confirmed next week is unknown today.
Using it would be *label leakage*, the classic way fraud models look great
offline and then fail in production.

The same semantics are implemented in pure pandas in ``online_features.py``;
``tests/test_feature_parity.py`` proves both give the same numbers.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from sentinelpay.features.registry import (
    AMOUNT_AVG_WINDOW_S,
    FEATURE_NAMES,
    ID_COLUMNS,
    LABEL_COLUMNS,
    MERCHANT_RATE_PRIOR,
    MERCHANT_RATE_STRENGTH,
    MIN_TRAVEL_HOURS,
    NIGHT_HOURS,
    WINDOWS_S,
)
from sentinelpay.utils.config import Settings
from sentinelpay.utils.io import write_json
from sentinelpay.utils.logging import get_logger

log = get_logger(__name__)

EARTH_RADIUS_KM = 6371.0
UNBOUNDED = Window.unboundedPreceding


def haversine_km(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    """Great-circle distance in km between two points (Spark expression)."""
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = (
        F.sin(dlat / 2) ** 2
        + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2
    )
    return F.lit(2 * EARTH_RADIUS_KM) * F.asin(F.sqrt(a))


def _prior(*keys: str, seconds: int | None = None) -> Window:
    """Window over earlier rows of the same key(s): all history or the last ``seconds``."""
    w = Window.partitionBy(*keys).orderBy("epoch")
    return w.rangeBetween(UNBOUNDED if seconds is None else -seconds, -1)


def _merchant_known_frauds(df: DataFrame) -> DataFrame:
    """Per transaction: frauds at its merchant whose label was known before it.

    Trick: treat every confirmed fraud as an *event* at its
    ``label_available_ts`` and every transaction as a *query* at its ``ts``,
    stack both in one timeline per merchant, and take a running sum of fraud
    events strictly before each query. One sort per merchant, no self-join.
    """
    label_events = df.filter(F.col("is_fraud") == 1).select(
        "merchant_id",
        F.col("label_available_ts").cast("long").alias("epoch"),
        F.lit(1).alias("fraud_event"),
        F.lit(None).cast("string").alias("transaction_id"),
    )
    queries = df.select(
        "merchant_id",
        "epoch",
        F.lit(0).alias("fraud_event"),
        "transaction_id",
    )
    timeline = label_events.unionByName(queries)
    known = F.coalesce(F.sum("fraud_event").over(_prior("merchant_id")), F.lit(0))
    return (
        timeline.withColumn("merchant_known_frauds", known)
        .filter(F.col("transaction_id").isNotNull())
        .select("transaction_id", "merchant_known_frauds")
    )


def build_features(silver: DataFrame) -> DataFrame:
    """Pure transformation: silver transactions -> gold feature table."""
    df = silver.withColumn("epoch", F.col("ts").cast("long"))
    is_cp = F.col("channel") == "card_present"
    card_all = _prior("card_id")

    # -- transaction ---------------------------------------------------------
    df = df.withColumn("is_card_present", is_cp.cast("int"))

    # -- velocity ------------------------------------------------------------
    for label, seconds in WINDOWS_S.items():
        w = _prior("card_id", seconds=seconds)
        df = df.withColumn(f"tx_count_{label}", F.count("*").over(w).cast("int"))
        df = df.withColumn(f"amount_sum_{label}", F.coalesce(F.sum("amount").over(w), F.lit(0.0)))
    df = df.withColumn(
        "distinct_merchants_24h",
        F.size(F.collect_set("merchant_id").over(_prior("card_id", seconds=WINDOWS_S["24h"]))),
    )

    # -- amount behaviour ----------------------------------------------------
    mean_prior = F.avg("amount").over(card_all)
    std_prior = F.stddev_samp("amount").over(card_all)  # null with < 2 rows
    avg_30d = F.avg("amount").over(_prior("card_id", seconds=AMOUNT_AVG_WINDOW_S))
    df = (
        df.withColumn("card_tx_count_prior", F.count("*").over(card_all).cast("int"))
        .withColumn(
            "amount_zscore",
            F.when(std_prior > 0, (F.col("amount") - mean_prior) / std_prior),
        )
        .withColumn("amount_to_avg_30d", F.col("amount") / avg_30d)
    )

    # -- geography -----------------------------------------------------------
    # Previous card-present purchase: max over a struct orders by epoch first,
    # so this picks the latest earlier in-person location deterministically.
    prev_cp = F.max(F.when(is_cp, F.struct("epoch", "merchant_lat", "merchant_lon"))).over(card_all)
    df = df.withColumn("_prev_cp", prev_cp)
    km_prev = haversine_km(
        F.col("_prev_cp.merchant_lat"),
        F.col("_prev_cp.merchant_lon"),
        F.col("merchant_lat"),
        F.col("merchant_lon"),
    )
    hours_prev = (F.col("epoch") - F.col("_prev_cp.epoch")) / 3600.0
    df = (
        df.withColumn(
            "dist_from_home_km",
            F.when(
                is_cp,
                haversine_km(
                    F.col("customer_lat"),
                    F.col("customer_lon"),
                    F.col("merchant_lat"),
                    F.col("merchant_lon"),
                ),
            ),
        )
        .withColumn("km_from_prev_cp", F.when(is_cp, km_prev))
        .withColumn(
            "speed_kmh_from_prev_cp",
            F.when(is_cp, km_prev / F.greatest(hours_prev, F.lit(MIN_TRAVEL_HOURS))),
        )
        .drop("_prev_cp")
    )

    # -- novelty -------------------------------------------------------------
    def first_time(*keys: str) -> Column:
        return (F.count("*").over(_prior("card_id", *keys)) == 0).cast("int")

    df = (
        df.withColumn("is_new_merchant", first_time("merchant_id"))
        .withColumn("is_new_category", first_time("merchant_category"))
        .withColumn(
            "is_new_device",
            F.when(~is_cp & F.col("device_id").isNotNull(), first_time("device_id")).otherwise(0),
        )
    )

    # -- time ----------------------------------------------------------------
    hour = F.hour("ts")
    df = (
        df.withColumn("hour_of_day", hour)
        # Spark: 1 = Sunday ... 7 = Saturday  ->  ISO: 1 = Monday ... 7 = Sunday
        .withColumn("day_of_week", ((F.dayofweek("ts") + 5) % 7 + 1).cast("int"))
        .withColumn("is_night", hour.isin(*NIGHT_HOURS).cast("int"))
        .withColumn(
            "seconds_since_prev_tx",
            (F.col("epoch") - F.max("epoch").over(card_all)).cast("double"),
        )
    )

    # -- merchant risk -------------------------------------------------------
    merchant_all = _prior("merchant_id")
    df = df.withColumn(
        "merchant_distinct_cards_1h",
        F.size(F.collect_set("card_id").over(_prior("merchant_id", seconds=WINDOWS_S["1h"]))),
    ).withColumn("merchant_tx_count_prior", F.count("*").over(merchant_all).cast("int"))

    df = df.join(_merchant_known_frauds(df), "transaction_id", "left")
    df = df.withColumn(
        "merchant_fraud_rate_prior",
        (F.col("merchant_known_frauds") + F.lit(MERCHANT_RATE_PRIOR * MERCHANT_RATE_STRENGTH))
        / (F.col("merchant_tx_count_prior") + F.lit(MERCHANT_RATE_STRENGTH)),
    )

    return df.select(*ID_COLUMNS, *LABEL_COLUMNS, *FEATURE_NAMES)


def feature_summary(gold: DataFrame) -> dict[str, Any]:
    """Per-feature null rate and mean for legit vs fraud (reports/feature_summary.json)."""
    numeric = [c for c in FEATURE_NAMES if c != "merchant_category"]
    aggs = [F.count("*").alias("rows")]
    for c in numeric:
        aggs += [
            F.avg(F.col(c).isNull().cast("double")).alias(f"{c}__null"),
            F.avg(c).alias(f"{c}__mean"),
        ]
    by_label = {r["is_fraud"]: r.asDict() for r in gold.groupBy("is_fraud").agg(*aggs).collect()}
    legit, fraud = by_label.get(0, {}), by_label.get(1, {})

    def r(x: float | None) -> float | None:
        return None if x is None else round(float(x), 4)

    return {
        "rows": int((legit.get("rows") or 0) + (fraud.get("rows") or 0)),
        "features": {
            c: {
                "null_rate_legit": r(legit.get(f"{c}__null")),
                "mean_legit": r(legit.get(f"{c}__mean")),
                "mean_fraud": r(fraud.get(f"{c}__mean")),
            }
            for c in numeric
        },
    }


def run(spark: SparkSession, settings: Settings) -> dict[str, Any]:
    """Build the gold feature table from silver (full rebuild, partitioned by date)."""
    paths = settings.paths
    silver = spark.read.format("delta").load(paths.silver)
    gold = build_features(silver)
    (
        gold.repartition("tx_date")  # one file per day instead of many tiny ones
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("tx_date")
        .save(paths.gold)
    )
    summary = feature_summary(spark.read.format("delta").load(paths.gold))
    write_json(paths.report("feature_summary.json"), summary)
    log.info(
        "gold: %s rows x %d features -> %s", f"{summary['rows']:,}", len(FEATURE_NAMES), paths.gold
    )
    return summary
