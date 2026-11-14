"""Spark session factory that works both locally and on Databricks."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession

from sentinelpay.utils.config import CONFIG_DIR, Settings


def is_databricks() -> bool:
    """True when running inside a Databricks runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(settings: Settings, app_name: str = "sentinelpay") -> SparkSession:
    """Return a Delta-enabled Spark session.

    On Databricks the platform owns the session (and Delta is built in), so we
    just attach to it. Locally we build one with the open-source Delta jars,
    UTC session time zone (so window maths on timestamps is unambiguous), and
    a small shuffle partition count suited to a laptop.
    """
    if is_databricks():
        spark = SparkSession.builder.getOrCreate()
        spark.conf.set("spark.sql.shuffle.partitions", str(settings.spark.shuffle_partitions))
        return spark

    from delta import configure_spark_with_delta_pip

    # Python workers must run the same interpreter as the driver; otherwise
    # Spark picks whatever `python3` is first on PATH and fails on a version
    # mismatch (e.g. driver in .venv on 3.11, system python 3.14).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    # Local mode never needs the network interface; binding to loopback avoids
    # the "hostname resolves to a loopback address" warning on laptops.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    log4j = CONFIG_DIR / "log4j2.properties"

    builder = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark.master)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", settings.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", str(settings.spark.shuffle_partitions))
        .config("spark.ui.enabled", str(settings.spark.ui_enabled).lower())
        .config("spark.ui.showConsoleProgress", "false")
        # Arrow makes Spark -> pandas conversion fast and avoids a timestamp bug
        # in PySpark's slow (non-Arrow) path with tz-aware pandas columns.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.extraJavaOptions", f"-Dlog4j.configurationFile=file:{log4j}")
        # Keep Delta's own snapshot work light on small local tables.
        .config("spark.databricks.delta.snapshotPartitions", "2")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
