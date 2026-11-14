"""Explicit schemas: the data contract between the raw feed and the lakehouse.

Reading with an explicit schema (instead of inferring one) means a producer
that silently changes a column type fails loudly at ingestion instead of
corrupting downstream features.
"""

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

RAW_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("card_id", StringType()),
        StructField("customer_id", StringType()),
        StructField("merchant_id", StringType()),
        StructField("merchant_category", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("ts", TimestampType()),
        StructField("customer_lat", DoubleType()),
        StructField("customer_lon", DoubleType()),
        StructField("merchant_lat", DoubleType()),
        StructField("merchant_lon", DoubleType()),
        StructField("channel", StringType()),
        StructField("device_id", StringType()),
        StructField("is_fraud", IntegerType()),
        StructField("fraud_pattern", StringType()),
        StructField("label_available_ts", TimestampType()),
    ]
)

# Bronze = raw columns + ingestion lineage.
BRONZE_SCHEMA = StructType(
    [
        *RAW_SCHEMA.fields,
        StructField("ingest_ts", TimestampType(), nullable=False),
        StructField("source_file", StringType(), nullable=False),
    ]
)

CHANNELS = ("card_present", "online")
