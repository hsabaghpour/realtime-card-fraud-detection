"""Bronze layer: land the raw feed in Delta exactly as received, plus lineage.

Bronze is append-only and never "fixes" data; that is silver's job. Keeping
the raw records means we can always replay silver/gold if a rule changes.

Ingestion is idempotent at the file level: each row records its source file,
and files already present in the table are skipped. Re-running the step (e.g.
an Airflow retry) therefore never double-loads data.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from sentinelpay.pipeline.schemas import RAW_SCHEMA
from sentinelpay.utils.config import Settings
from sentinelpay.utils.logging import get_logger

log = get_logger(__name__)


def read_raw(spark: SparkSession, source_dir: str) -> DataFrame:
    """Read landing parquet with the enforced raw schema and attach lineage."""
    return (
        spark.read.schema(RAW_SCHEMA)
        .parquet(source_dir)
        .select(
            *RAW_SCHEMA.fieldNames(),
            F.current_timestamp().alias("ingest_ts"),
            F.col("_metadata.file_path").alias("source_file"),
        )
    )


def already_ingested(spark: SparkSession, bronze_path: str) -> set[str]:
    """Source files that are already in the bronze table."""
    if not DeltaTable.isDeltaTable(spark, bronze_path):
        return set()
    rows = spark.read.format("delta").load(bronze_path).select("source_file").distinct().collect()
    return {r.source_file for r in rows}


class StaleBronzeError(RuntimeError):
    """Bronze holds data from landing files that have since been replaced."""


def check_not_stale(done: set[str], source_dir: str) -> None:
    """Fail if bronze contains files that are no longer in the landing zone.

    That happens when the data was regenerated (e.g. a different SAMPLE_SIZE)
    on top of an existing lakehouse. Appending would silently mix the old and
    new datasets, so we stop and tell the user how to rebuild instead.
    """
    current = {p.name for p in Path(source_dir).glob("*.parquet")}
    stale = sorted({Path(urlparse(f).path).name for f in done} - current)
    if stale:
        raise StaleBronzeError(
            f"bronze contains data from landing files that no longer exist ({', '.join(stale)}). "
            "The landing data was regenerated. Rebuild from scratch with `make demo`, "
            "or run `make clean` first."
        )


def ingest(spark: SparkSession, source_dir: str, bronze_path: str) -> int:
    """Append new files from ``source_dir`` to bronze. Returns rows appended."""
    raw = read_raw(spark, source_dir)
    done = already_ingested(spark, bronze_path)
    check_not_stale(done, source_dir)
    if done:
        raw = raw.filter(~F.col("source_file").isin(list(done)))

    # Schema is enforced twice: the explicit read schema above, and Delta's
    # own schema enforcement, which rejects an append whose columns or types
    # differ from the existing table (no silent schema drift).
    n = raw.count()
    if n == 0:
        log.info("bronze: nothing new to ingest from %s", source_dir)
        return 0
    raw.write.format("delta").mode("append").save(bronze_path)
    log.info("bronze: appended %s rows from %s", f"{n:,}", source_dir)
    return n


def run(spark: SparkSession, settings: Settings) -> int:
    """Ingest the history-period landing zone into bronze.

    The production period is not ingested here; it arrives later through the
    streaming path (Phase 4).
    """
    return ingest(spark, settings.paths.landing_history, settings.paths.bronze)
