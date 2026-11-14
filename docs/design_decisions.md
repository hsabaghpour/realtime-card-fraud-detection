# Design decisions

Each entry records what was decided, why, and what the trade-off is. It grows with each
phase.

## Data

### Synthetic data instead of a public dataset
**Decision:** Generate data with a deterministic, seeded generator. A loader for the
public Kaggle dataset is planned as an optional extra.
**Why:** Kaggle downloads need a login, which breaks the one-command demo. Public datasets
also lack the things this project needs to demonstrate: late-arriving labels,
controllable drift, and ground truth on *which* fraud pattern each fraud row belongs to.
That ground truth lets us measure which patterns the model catches.
**Trade-off:** The model learns patterns we designed, so absolute metrics are
optimistic. The README says this plainly, and the value of the project is in the
engineering, not the headline number.

### Shared population, independent random streams per period
The population (customers, cards, merchants) comes from `rng([seed, 0])`, history from
`rng([seed, 1])`, and each production scenario from `rng([seed, 2, i])`. All periods share
the same cardholders, as in reality, and adding a drift scenario never changes the history
data.

### Legitimate travel is part of normal behaviour
About 25% of cards take one out-of-town trip per period. Without it, "far from home"
would be a perfect fraud signal and the impossible-travel feature would be trivially
easy. With it, the model has to use the *speed* between transactions, not just distance.

### Late-arriving labels
Real fraud labels arrive through chargebacks, days or weeks later. Each row carries
`label_available_ts`:
* fraud: transaction time + log-normal delay (median 10 days, clipped to 1–60 days)
* legit: transaction time + 45 days, when the chargeback window closes and the
  transaction is treated as confirmed legitimate

Any feature or metric that uses labels must only use labels where
`label_available_ts < now`. This prevents label leakage in training (Phase 2) and gives
realistic delayed performance monitoring (Phase 4).

### Dirty records are injected on purpose
About 0.2% of the raw feed is duplicates (re-sent messages) or malformed rows (null card,
non-positive amount, impossible coordinates, null timestamp). The silver layer has real
work to do, and the tests check that every injected problem is caught exactly.

## Lakehouse

### Medallion layers on Delta Lake
* **Bronze** keeps raw data exactly as received plus lineage (`ingest_ts`,
  `source_file`). It never "fixes" anything, so silver and gold can always be rebuilt
  if a rule changes.
* **Silver** is the validated, de-duplicated, typed version.
* **Gold** (Phase 2) is the feature table.

Delta gives ACID appends, schema enforcement, and time travel to reproduce the exact
data a model was trained on.

### Schema enforced twice
The raw feed is read with an explicit schema (no inference), and Delta rejects appends
whose schema differs from the table. A producer that changes a column type fails loudly
at ingestion instead of quietly corrupting features.

### Idempotent bronze ingestion
Each bronze row records its source file, and files already in the table are skipped. An
Airflow retry or a re-run of `make demo` never double-loads data. At larger scale this
would use Databricks Auto Loader, which tracks ingested files in a checkpoint.

### Quarantine instead of drop
Invalid rows go to `silver_quarantine` with **all** their failing rules
(`null_card_id;non_positive_amount`), not just the first one. Data owners can fix every
problem in one pass, and `reports/dq_metrics.json` tracks counts per reason. Each rule is
wrapped in `coalesce(rule, true)` so a rule that evaluates to SQL `NULL` counts as a
failure instead of silently passing.

### De-duplication keeps the latest ingested copy
Payment feeds re-send messages. Silver keeps one row per `transaction_id`, ordered by
`ingest_ts` (latest wins), so a corrected re-send replaces the original.

### Silver is rebuilt in full (for now)
At 500k rows a full rebuild from bronze takes ~15 seconds and is obviously correct. At
production scale, silver would be updated incrementally with `MERGE INTO` on
`transaction_id`, reading only new bronze data via Delta Change Data Feed or Structured
Streaming.

## Runtime

### Spark 4 + Delta 4, Java 17
PySpark 4.0 supports Java 17 and 21. `scripts/find_java.sh` checks the *actual* version of
each candidate JDK. macOS's `java_home -v 17` silently returns the newest JDK when 17
isn't registered, which caused a real bug during development. `docker compose up` (Phase 6)
removes the Java requirement entirely.

### Pin the Python worker to the driver interpreter
Spark launches Python workers using whatever `python3` is first on `PATH`. If that differs
from the driver's interpreter (e.g. a 3.11 venv and a 3.14 system Python), jobs fail with
`PYTHON_VERSION_MISMATCH`. `get_spark()` sets `PYSPARK_PYTHON` to `sys.executable`.

### UTC everywhere
The Spark session time zone is UTC, and timestamps are formatted inside Spark before
collecting. Collecting raw timestamps converts them to the driver machine's local zone,
which made the DQ report show `2024-12-31` for data starting `2025-01-01`. This was caught
by a test.

### One config, many environments
`configs/base.yaml` holds the defaults. `local.yaml` / `databricks.yaml` override them
(selected by `SENTINELPAY_ENV`), then environment variables, then explicit overrides. The
code only sees an immutable, typed `Settings` object. On Databricks, `get_spark()`
attaches to the platform's session and paths point at a Unity Catalog volume. No code
changes are needed.

### Reproducible dependencies
`pyproject.toml` holds compatible version ranges. `requirements.lock` holds the exact
resolved versions of every transitive dependency, and `make install` uses the lock file.
pandas is held below 3.0 because PySpark 4.0 was not built against it.
