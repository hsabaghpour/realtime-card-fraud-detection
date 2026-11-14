# Feature registry

_Generated from `src/sentinelpay/features/registry.py` by `sentinelpay features-doc`. Do not edit by hand._

Every feature uses only data from **strictly before** the transaction it describes (point-in-time correct). Label-based features only use labels whose `label_available_ts` is before the transaction.

## Transaction

| Feature | Type | Computation | Why it catches fraud |
|---|---|---|---|
| `amount` | float | Transaction amount in USD. | Baseline size of the purchase; very small and very large amounts are both risky. |
| `is_card_present` | int | 1 if the card was physically present. | Online (card-not-present) transactions carry most card fraud. |
| `merchant_category` | category | Merchant category code. | Some categories (gambling, money transfer, gift cards) are favoured by fraudsters. |

## Velocity

| Feature | Type | Computation | Why it catches fraud |
|---|---|---|---|
| `tx_count_1h` | int | Card's transactions in the previous hour. | Bursts of activity are typical of card testing and cash-out attacks. |
| `tx_count_24h` | int | Card's transactions in the previous 24 hours. | Unusually busy day for this card. |
| `tx_count_7d` | int | Card's transactions in the previous 7 days. | Normal weekly activity level; context for the shorter windows. |
| `amount_sum_1h` | float | Card's total spend in the previous hour. | Fraudsters try to spend as much as possible before the card is blocked. |
| `amount_sum_24h` | float | Card's total spend in the previous 24 hours. | Daily spend far above normal suggests account takeover. |
| `amount_sum_7d` | float | Card's total spend in the previous 7 days. | Weekly spend context. |
| `distinct_merchants_24h` | int | Different merchants the card used in the previous 24 hours. | Card testing hits many merchants in a short time. |

## Amount

| Feature | Type | Computation | Why it catches fraud |
|---|---|---|---|
| `card_tx_count_prior` | int | All earlier transactions on the card. | How much history we have; novelty and z-scores mean little for a brand-new card. |
| `amount_zscore` | float | (amount - card's historical mean) / card's historical std. Null with < 2 prior txns. | How unusual this amount is for this particular cardholder. |
| `amount_to_avg_30d` | float | Amount divided by the card's average amount over the previous 30 days. | A purchase many times the usual ticket size is a classic fraud signal. |

## Geo

| Feature | Type | Computation | Why it catches fraud |
|---|---|---|---|
| `dist_from_home_km` | float | Distance from cardholder's home to the merchant (card-present only). | In-person purchases far from home are rarer and riskier. |
| `km_from_prev_cp` | float | Distance from the card's previous card-present purchase (card-present only). | Large jumps between physical locations suggest a cloned card. |
| `speed_kmh_from_prev_cp` | float | km_from_prev_cp divided by the hours since that purchase (min 1 minute). | Impossible travel: a card can't be in two distant cities minutes apart. |

## Novelty

| Feature | Type | Computation | Why it catches fraud |
|---|---|---|---|
| `is_new_merchant` | int | 1 if the card never used this merchant before. | Fraudsters shop where the real cardholder never does. |
| `is_new_category` | int | 1 if the card never used this merchant category before. | A sudden first gambling or money-transfer purchase is suspicious. |
| `is_new_device` | int | 1 if an online purchase comes from a device never seen on this card (0 in person). | A new device is the strongest sign someone else is using the card online. |

## Time

| Feature | Type | Computation | Why it catches fraud |
|---|---|---|---|
| `hour_of_day` | int | Hour of the transaction (0-23, UTC). | Fraud clusters at night when cardholders are asleep. |
| `day_of_week` | int | ISO day of week (1 = Monday, 7 = Sunday). | Captures weekly spending rhythm. |
| `is_night` | int | 1 if between 00:00 and 05:59. | Simple night flag that linear models can use directly. |
| `seconds_since_prev_tx` | float | Seconds since the card's previous transaction. Null for a card's first one. | Seconds-apart transactions are machine-like (card testing). |

## Merchant

| Feature | Type | Computation | Why it catches fraud |
|---|---|---|---|
| `merchant_distinct_cards_1h` | int | Different cards used at this merchant in the previous hour. | Many cards at one merchant in a short window signals a compromised merchant. |
| `merchant_tx_count_prior` | int | All earlier transactions at this merchant. | Merchant size; context for its fraud rate. |
| `merchant_fraud_rate_prior` | float | Smoothed share of the merchant's earlier transactions confirmed as fraud, using only labels available before this transaction (respects chargeback delay). | Merchants with a history of fraud are likely to see more. |

