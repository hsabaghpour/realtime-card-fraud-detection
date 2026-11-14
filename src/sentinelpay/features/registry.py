"""Feature registry: the single source of truth for every model feature.

Both feature implementations (Spark for training/batch, pandas for online
scoring), the docs (docs/features.md) and the GUI read from here. Adding a
feature means adding it here first; tests fail if an implementation produces
a column the registry doesn't know about, or misses one it does.

Shared constants (window sizes, night hours, smoothing) also live here, so
the two implementations cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DType = Literal["int", "float", "category"]

# --- shared constants ---------------------------------------------------------
WINDOWS_S: dict[str, int] = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400}
AMOUNT_AVG_WINDOW_S = 30 * 86_400
NIGHT_HOURS = range(0, 6)  # 00:00-05:59 UTC
MIN_TRAVEL_HOURS = 1 / 60  # floor on elapsed time so speed can't divide by ~0
# Bayesian smoothing for the merchant fraud rate: a merchant with little history
# is pulled towards a portfolio-wide prior instead of showing 0% or 100%.
MERCHANT_RATE_PRIOR = 0.005
MERCHANT_RATE_STRENGTH = 50.0


@dataclass(frozen=True)
class Feature:
    """One model feature and what it means to the business."""

    name: str
    group: str
    dtype: DType
    description: str  # how it is computed
    business_meaning: str  # why it helps catch fraud
    owner: str = "fraud-data-science"


FEATURES: tuple[Feature, ...] = (
    # -- transaction ---------------------------------------------------------
    Feature(
        "amount",
        "transaction",
        "float",
        "Transaction amount in USD.",
        "Baseline size of the purchase; very small and very large amounts are both risky.",
    ),
    Feature(
        "is_card_present",
        "transaction",
        "int",
        "1 if the card was physically present.",
        "Online (card-not-present) transactions carry most card fraud.",
    ),
    Feature(
        "merchant_category",
        "transaction",
        "category",
        "Merchant category code.",
        "Some categories (gambling, money transfer, gift cards) are favoured by fraudsters.",
    ),
    # -- velocity ------------------------------------------------------------
    Feature(
        "tx_count_1h",
        "velocity",
        "int",
        "Card's transactions in the previous hour.",
        "Bursts of activity are typical of card testing and cash-out attacks.",
    ),
    Feature(
        "tx_count_24h",
        "velocity",
        "int",
        "Card's transactions in the previous 24 hours.",
        "Unusually busy day for this card.",
    ),
    Feature(
        "tx_count_7d",
        "velocity",
        "int",
        "Card's transactions in the previous 7 days.",
        "Normal weekly activity level; context for the shorter windows.",
    ),
    Feature(
        "amount_sum_1h",
        "velocity",
        "float",
        "Card's total spend in the previous hour.",
        "Fraudsters try to spend as much as possible before the card is blocked.",
    ),
    Feature(
        "amount_sum_24h",
        "velocity",
        "float",
        "Card's total spend in the previous 24 hours.",
        "Daily spend far above normal suggests account takeover.",
    ),
    Feature(
        "amount_sum_7d",
        "velocity",
        "float",
        "Card's total spend in the previous 7 days.",
        "Weekly spend context.",
    ),
    Feature(
        "distinct_merchants_24h",
        "velocity",
        "int",
        "Different merchants the card used in the previous 24 hours.",
        "Card testing hits many merchants in a short time.",
    ),
    # -- amount behaviour ----------------------------------------------------
    Feature(
        "card_tx_count_prior",
        "amount",
        "int",
        "All earlier transactions on the card.",
        "How much history we have; novelty and z-scores mean little for a brand-new card.",
    ),
    Feature(
        "amount_zscore",
        "amount",
        "float",
        "(amount - card's historical mean) / card's historical std. Null with < 2 prior txns.",
        "How unusual this amount is for this particular cardholder.",
    ),
    Feature(
        "amount_to_avg_30d",
        "amount",
        "float",
        "Amount divided by the card's average amount over the previous 30 days.",
        "A purchase many times the usual ticket size is a classic fraud signal.",
    ),
    # -- geography -----------------------------------------------------------
    Feature(
        "dist_from_home_km",
        "geo",
        "float",
        "Distance from cardholder's home to the merchant (card-present only).",
        "In-person purchases far from home are rarer and riskier.",
    ),
    Feature(
        "km_from_prev_cp",
        "geo",
        "float",
        "Distance from the card's previous card-present purchase (card-present only).",
        "Large jumps between physical locations suggest a cloned card.",
    ),
    Feature(
        "speed_kmh_from_prev_cp",
        "geo",
        "float",
        "km_from_prev_cp divided by the hours since that purchase (min 1 minute).",
        "Impossible travel: a card can't be in two distant cities minutes apart.",
    ),
    # -- novelty -------------------------------------------------------------
    Feature(
        "is_new_merchant",
        "novelty",
        "int",
        "1 if the card never used this merchant before.",
        "Fraudsters shop where the real cardholder never does.",
    ),
    Feature(
        "is_new_category",
        "novelty",
        "int",
        "1 if the card never used this merchant category before.",
        "A sudden first gambling or money-transfer purchase is suspicious.",
    ),
    Feature(
        "is_new_device",
        "novelty",
        "int",
        "1 if an online purchase comes from a device never seen on this card (0 in person).",
        "A new device is the strongest sign someone else is using the card online.",
    ),
    # -- time ----------------------------------------------------------------
    Feature(
        "hour_of_day",
        "time",
        "int",
        "Hour of the transaction (0-23, UTC).",
        "Fraud clusters at night when cardholders are asleep.",
    ),
    Feature(
        "day_of_week",
        "time",
        "int",
        "ISO day of week (1 = Monday, 7 = Sunday).",
        "Captures weekly spending rhythm.",
    ),
    Feature(
        "is_night",
        "time",
        "int",
        "1 if between 00:00 and 05:59.",
        "Simple night flag that linear models can use directly.",
    ),
    Feature(
        "seconds_since_prev_tx",
        "time",
        "float",
        "Seconds since the card's previous transaction. Null for a card's first one.",
        "Seconds-apart transactions are machine-like (card testing).",
    ),
    # -- merchant risk -------------------------------------------------------
    Feature(
        "merchant_distinct_cards_1h",
        "merchant",
        "int",
        "Different cards used at this merchant in the previous hour.",
        "Many cards at one merchant in a short window signals a compromised merchant.",
    ),
    Feature(
        "merchant_tx_count_prior",
        "merchant",
        "int",
        "All earlier transactions at this merchant.",
        "Merchant size; context for its fraud rate.",
    ),
    Feature(
        "merchant_fraud_rate_prior",
        "merchant",
        "float",
        "Smoothed share of the merchant's earlier transactions confirmed as fraud, using "
        "only labels available before this transaction (respects chargeback delay).",
        "Merchants with a history of fraud are likely to see more.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
CATEGORICAL: tuple[str, ...] = tuple(f.name for f in FEATURES if f.dtype == "category")

# Columns kept next to the features in the gold table but NEVER used as model
# inputs: identifiers, the label, and evaluation-only ground truth.
ID_COLUMNS = ("transaction_id", "card_id", "customer_id", "merchant_id", "ts", "tx_date")
LABEL_COLUMNS = ("is_fraud", "fraud_pattern", "label_available_ts")


def by_name(name: str) -> Feature:
    """Look up a feature definition."""
    for f in FEATURES:
        if f.name == name:
            return f
    raise KeyError(name)


def to_markdown() -> str:
    """Render the registry as the docs/features.md table."""
    lines = [
        "# Feature registry",
        "",
        "_Generated from `src/sentinelpay/features/registry.py` by "
        "`sentinelpay features-doc`. Do not edit by hand._",
        "",
        "Every feature uses only data from **strictly before** the transaction it describes "
        "(point-in-time correct). Label-based features only use labels whose "
        "`label_available_ts` is before the transaction.",
        "",
    ]
    for group in dict.fromkeys(f.group for f in FEATURES):
        lines += [
            f"## {group.capitalize()}",
            "",
            "| Feature | Type | Computation | Why it catches fraud |",
            "|---|---|---|---|",
        ]
        for f in FEATURES:
            if f.group == group:
                lines.append(f"| `{f.name}` | {f.dtype} | {f.description} | {f.business_meaning} |")
        lines.append("")
    return "\n".join(lines)
