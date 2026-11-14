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
Each bronze row records its source file, and files already in the table are skipped, so
an Airflow retry never double-loads data. At larger scale this would use Databricks Auto
Loader, which tracks ingested files in a checkpoint.

File-level idempotency only works if a file name always means the same content. An early
version always wrote `transactions.parquet`. Regenerating with a different `SAMPLE_SIZE`
reused the name, bronze skipped the "already ingested" file, and silver was silently
rebuilt from the *old* data. Two fixes:
* Landing files are named after their content (`transactions_seed42_n500000.parquet`).
* Bronze raises `StaleBronzeError` if it holds rows from files no longer in the landing
  zone, instead of appending and mixing two datasets. `make demo` always rebuilds the
  lakehouse from scratch, and a regression test covers the scenario.

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

## Features (Phase 2)

### Point-in-time correctness: strictly-before windows
Every feature for a transaction at time `t` uses only rows with `epoch <= t - 1`. In
Spark this is a window frame `rangeBetween(-W, -1)` over epoch seconds, partitioned by
card or merchant. Consequences, each covered by a test:
* The "1h" window covers `[t - 3600, t - 1]`. A transaction exactly one hour earlier
  counts; one 3,601 seconds earlier doesn't.
* Two transactions on the same card in the same second don't see each other. In a
  real-time system neither is processed when the other arrives.
* Adding *future* transactions never changes any past feature. The test builds features,
  adds later rows (including a fraud whose label arrives almost instantly), rebuilds,
  and asserts the earlier rows are byte-identical.

Why it matters: a feature that accidentally includes the current or a later
transaction makes offline metrics look excellent and then collapses in production.

### Merchant fraud rate without label leakage
The merchant's historical fraud rate may only count frauds **known** before `t`, i.e.
`label_available_ts < t`. A naive self-join (each transaction × all earlier transactions
at its merchant) is quadratic. Instead, each confirmed fraud becomes an *event* at its
`label_available_ts` and each transaction a *query* at its `ts`. Both go in one timeline
per merchant, and a running sum strictly before each query gives the known-fraud count:
one sort per merchant, no join explosion.

Two further choices:
* **Denominator = all earlier transactions**, not only labelled ones. Legit transactions
  are only "confirmed" when the 45-day chargeback window closes, while frauds are
  confirmed after about 10 days. Dividing by labelled transactions would make every
  merchant look close to 100% fraud for its first six weeks (label-maturity bias).
* **Bayesian smoothing** towards a 0.5% prior with strength 50, so a new merchant with
  one fraud out of two transactions doesn't show 50%.

### Geography uses card-present purchases only
Online merchants only have a registered HQ address, which says nothing about where the
cardholder is. Distance from home, distance from the previous purchase and implied speed
are therefore computed only for card-present transactions, against the previous
*card-present* purchase. A test checks that an online purchase "in LA" doesn't break the
NYC → NYC sequence around it. Speed uses a 1-minute minimum elapsed time, so two
purchases in the same minute can't divide by zero.

### One feature registry, two implementations, a parity test
Training uses Spark over months of data. Real-time scoring needs one transaction in
milliseconds, which Spark can't do. So there are two implementations, `spark_features.py`
and `online_features.py` (pandas), and the classic risk of **training/serving skew**: a
feature computed slightly differently online than in training silently degrades the
model.

Defences:
* Both read every constant (windows, night hours, smoothing) from `registry.py`.
* `test_feature_parity.py` runs the real pipeline on generated data and recomputes every
  feature for every fraud row plus 300 random rows with the online function, using only
  raw history. All values must match.
* The parity test was itself checked by mutation: injecting a label-leakage bug into the
  online code made it fail with 44 mismatches.

The registry also generates `docs/features.md`, and a test fails if the doc is stale.

### Gold is a full rebuild, partitioned by date
At this size, rebuilding gold takes about 35 seconds. At production scale the same window
logic would run incrementally (Structured Streaming with state, or a feature store such
as Databricks Feature Engineering that keeps per-card aggregates and serves them online).
Phase 4 discusses this trade-off for streaming scoring.

### Hour of day is UTC
The synthetic data has no time zones, so hour features use UTC. A real system would use
the merchant's (or cardholder's) local time, since "3 am" only means "unusual" locally.

## Model (Phase 3)

### Making the synthetic data honest
The first model scored a PR-AUC of **0.992** with 99% recall. That would make an
experienced fraud data scientist suspicious, not impressed. It happened because the
generator made fraud perfectly separable: every card tester used a new device, nobody
genuine shopped at 3 am, every big purchase was fraud. Two fixes in the generator:
* **Hard negatives:** night owls, genuine big-ticket purchases, genuinely new devices.
  These are realistic look-alikes, so every signal is a *risk indicator*, not proof.
* **Stealthier fraud:** smaller amount spikes, travel "only" 400–1,000 km/h on the slow
  end (overlapping real flights), attackers using hijacked sessions instead of new
  devices, and slower merchant attacks.

SHAP then exposed a **simulator artefact**: "card has little history" was the #2
feature, because the generator picked fraud victims uniformly while genuine spending
comes mostly from busy cards. A model should learn fraud, not the simulator, so victims
are now picked in proportion to card activity, and the feature dropped to #7.

The result is PR-AUC ≈ 0.94 for LightGBM vs. ≈ 0.71 for the baseline. That is still
higher than on real fraud data (where 0.3–0.7 is common), and the README says so. The
*comparisons* between approaches are the meaningful part.

### Time-based split with a warm-up
`[14-day warm-up dropped] | train (~3.5 months) | validation (May) | test (June)`.
* **Not random:** random splits leak (same attack, same card, same week in train and
  test) and let the model learn from the future.
* **Warm-up dropped:** in the first two weeks every card looks "new", which says
  nothing about fraud.
* **Validation is used for three things only:** early stopping, calibration and
  choosing the threshold. Test is touched once, for the final numbers. The threshold
  chosen in May is ~$1,200 worse on June data than June's own optimum. That gap is the
  honest price of not peeking, and the Threshold page shows it.
* **Limitation:** training uses ground-truth labels. In production, the most recent
  ~45 days would only have *partially* matured labels (chargebacks still arriving), so
  you either train on matured data only or accept some label noise. Phase 4 monitoring
  handles label delay explicitly.

### Imbalance: square-root class weight, then Platt calibration
* `scale_pos_weight = sqrt(#legit / #fraud)` ≈ 11 focuses the trees on fraud without the
  full ~125x weight, which overfits the few fraud rows.
* Weighting inflates probabilities, so **Platt scaling** (a 1-D logistic regression on
  the validation log-odds) maps them back. A score of 0.3 then means ~30% of such
  transactions are fraud. Platt is monotone, so it can't change PR-AUC or the ranking.
  It only fixes the probability scale, which matters for cost-based thresholds and for
  analysts reading scores. Isotonic regression was rejected because its step function
  creates ties that degrade ranking.

### Threshold = minimum expected cost, chosen on validation
`cost = missed frauds × $250 + alerts × $5`, both configurable. The cost curve is
computed for every distinct threshold in O(n log n) with cumulative sums, and tied
scores are always alerted together. The GUI lets a manager change both costs and see
the optimal threshold move.

### Baselines that answer the manager's first question
* **Logistic regression** (signed-log transform, median imputation with missing
  indicators, one-hot category, balanced class weight): the simple, explainable model.
* **A rules engine** with six hand-written rules, one per known pattern: "what we had
  before ML". It catches 44% of fraud at 13% precision. The gap to the model is the
  business case.
* **No model:** every fraud is missed.

### SHAP via LightGBM's native TreeSHAP
`booster.predict(X, pred_contrib=True)` computes exact TreeSHAP, the same algorithm as
`shap.TreeExplainer`, with no extra dependency. (The `shap` package pulls in `numba`,
which often lags new NumPy releases.) Reason codes are the top 3 *positive*
contributions per alert in plain English. Two details:
* **Missing values are informative:** an online purchase has no physical location,
  and LightGBM learns that this raises risk. The reason says "Online purchase: no
  physical location to verify", not "dist_from_home_km not available".
* **Duplicate reasons are merged:** three geo features saying the same thing become one
  reason, so the analyst sees three *different* reasons.

### MLflow: one model object, champion/challenger aliases
* `FraudScorer` (booster + categories + Platt + threshold) is a plain Python class with
  no MLflow dependency. Batch scoring, streaming, the GUI and tests all use it. MLflow
  wraps it as a `pyfunc`, and a test checks the registered model scores identically to
  the in-memory one.
* The signature declares integer features as doubles, because at serving time a missing
  value turns an int column into a float, which an int schema would reject.
* Aliases, not stages: training registers a version as `challenger`. `promote`
  re-scores challenger **and** champion on the same current test window and moves
  `champion` only if the challenger wins on PR-AUC **and** cost. Training is
  deterministic, so retraining on the same data creates an identical v2 that is
  correctly *not* promoted. Rollback is moving an alias.
* `pyfunc` models are pickled, and MLflow rightly warns that unpickling runs code. Models
  are only ever loaded from this project's own registry, so the warning is silenced
  with a comment saying why.
* On Databricks the same code uses the Unity Catalog registry (three-level name in
  `configs/databricks.yaml`).

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
