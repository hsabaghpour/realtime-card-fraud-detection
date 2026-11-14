# Data dictionary

## Raw transaction feed (landing → bronze)

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique id. `H…` history, `P…` production, `HX…`/`PX…` malformed test records |
| `card_id` | string | Card used. A customer can have more than one card |
| `customer_id` | string | Cardholder |
| `merchant_id` | string | Merchant |
| `merchant_category` | string | One of 16 categories, e.g. `grocery_pos`, `shopping_net`, `gambling_net` |
| `amount` | double | Amount in `currency` |
| `currency` | string | Always `USD` in the synthetic data |
| `ts` | timestamp (UTC) | When the transaction happened |
| `customer_lat`, `customer_lon` | double | Cardholder's home location |
| `merchant_lat`, `merchant_lon` | double | Store location (card-present) or registered HQ (online) |
| `channel` | string | `card_present` or `online` |
| `device_id` | string, nullable | Device used for online purchases; null for card-present |
| `is_fraud` | int | 1 = confirmed fraud |
| `fraud_pattern` | string | Ground-truth pattern (`none` for legit). **Never used as a model feature.** It is only for evaluation and the GUI |
| `label_available_ts` | timestamp (UTC) | When the label became known (chargeback delay). Label-based features and monitoring must respect it |

## Bronze adds

| Column | Description |
|---|---|
| `ingest_ts` | When the row was ingested |
| `source_file` | Source file path (used for idempotent ingestion) |

## Silver adds

| Column | Description |
|---|---|
| `tx_date` | `to_date(ts)`, used for partitioning downstream |

## Silver quarantine adds

| Column | Description |
|---|---|
| `reject_reason` | `;`-separated failing rules: `null_<col>`, `non_positive_amount`, `invalid_customer_coordinates`, `invalid_merchant_coordinates`, `invalid_channel`, `invalid_label` |
| `quarantine_ts` | When the row was quarantined |

## Fraud patterns

| Pattern | What happens | Signal a model can learn |
|---|---|---|
| `card_testing` | 2–6 online charges of $0.50–$3, 1–20 minutes apart, then one large purchase. Two-thirds come from a new device, the rest through a hijacked session | Burst velocity, tiny amounts, new device |
| `impossible_travel` | Card-present use in a city >1500 km away, 10 minutes to 4 hours after a legit use. The slow end overlaps with genuine flights | Implied travel speed, distance from home |
| `high_amount` | One purchase at 3–20× the card's usual spend at a merchant it never used. Overlaps with genuine big-ticket buys | Amount z-score, new merchant |
| `night_burst` | 2–5 gambling / money-transfer charges between about midnight and 6 am | Hour, risky category, velocity |
| `compromised_merchant` | 6–30 different stolen cards used at one online merchant within 3 hours, half from a few shared devices | Distinct cards per merchant per hour |
| `gift_card_cashout` | *Production drift only.* 2–4 gift-card purchases by day from the victim's own device | Deliberately hard: few existing signals fire |

Victim cards are chosen in proportion to how much each card is used, so "low-activity card" is not a fake fraud signal.

## Genuine look-alikes ("hard negatives")

Real fraud overlaps with genuine behaviour. Without these, every fraud signal would be a perfect rule and the model would look unrealistically good:

| Behaviour | Share | Looks like |
|---|---|---|
| Night owls whose spending peaks around 1 am | 4% of customers | Night-time burst |
| Genuine big-ticket purchases at 4–15× the usual amount (travel, electronics, home) | 4% of purchases in those categories | High-amount anomaly |
| Online purchases from a genuinely new device (new phone, work laptop) | 4% of online purchases | Card testing / account takeover |
| Out-of-town trips (some starting hours after a purchase at home) | 25% of cards | Impossible travel |

## Drift scenarios (production period only)

| Scenario | Change |
|---|---|
| `none` | Same behaviour as history |
| `new_fraud_pattern` | Adds `gift_card_cashout` (~0.6% of rows) |
| `spending_shift` | Amounts ×1.4, volume ×1.2 (holiday season / inflation) |
| `merchant_mix_shift` | Online share ×2.2, with more retail and electronics |
