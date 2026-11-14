"""Deterministic synthetic card-transaction generator.

Why synthetic? Public fraud datasets need a login to download, which would
break the one-command demo, and none of them ship with late-arriving labels or
controllable drift. Generating the data also means we *know* the ground truth
of every fraud pattern, so the GUI can show which patterns the model catches.

Two periods are produced from one shared population of customers, cards and
merchants:

* ``history``    – ~6 months, used for training / validation / test
* ``production`` – the following month, optionally with injected drift; it is
  replayed as a stream in later phases

Everything is seeded: the population comes from ``rng([seed, 0])``, history
from ``rng([seed, 1])`` and each production scenario from
``rng([seed, 2, scenario_index])``, so the same seed always yields the same
rows, byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from sentinelpay.utils.config import Settings
from sentinelpay.utils.geo import haversine_km
from sentinelpay.utils.io import write_json
from sentinelpay.utils.logging import get_logger

log = get_logger(__name__)

Period = Literal["history", "production"]

DRIFT_SCENARIOS: tuple[str, ...] = (
    "none",
    "new_fraud_pattern",
    "spending_shift",
    "merchant_mix_shift",
)
FRAUD_PATTERNS: tuple[str, ...] = (
    "card_testing",
    "impossible_travel",
    "high_amount",
    "night_burst",
    "compromised_merchant",
)
# Only appears in the production period under the ``new_fraud_pattern`` scenario.
NEW_FRAUD_PATTERN = "gift_card_cashout"

# Final column order of the raw feed (the "contract" with the bronze layer).
RAW_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "card_id",
    "customer_id",
    "merchant_id",
    "merchant_category",
    "amount",
    "currency",
    "ts",
    "customer_lat",
    "customer_lon",
    "merchant_lat",
    "merchant_lon",
    "channel",
    "device_id",
    "is_fraud",
    "fraud_pattern",
    "label_available_ts",
)

# --------------------------------------------------------------------------- #
# Static reference data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Category:
    """A merchant category and how people typically spend in it."""

    name: str
    online: bool
    median_amount: float  # typical ticket size in USD
    sigma: float  # log-normal spread of ticket size
    share: float  # relative share of merchants / spend in this category


CATEGORIES: tuple[Category, ...] = (
    # card-present
    Category("grocery_pos", False, 45, 0.5, 0.16),
    Category("gas_transport", False, 50, 0.4, 0.12),
    Category("food_dining", False, 30, 0.6, 0.12),
    Category("shopping_pos", False, 70, 0.8, 0.10),
    Category("health_fitness", False, 50, 0.6, 0.06),
    Category("entertainment", False, 40, 0.7, 0.06),
    Category("home", False, 80, 0.8, 0.06),
    Category("kids_pets", False, 40, 0.6, 0.05),
    Category("personal_care", False, 30, 0.6, 0.05),
    Category("travel", False, 300, 0.9, 0.03),
    # online
    Category("shopping_net", True, 60, 0.9, 0.08),
    Category("misc_net", True, 35, 0.9, 0.05),
    Category("electronics_net", True, 250, 0.8, 0.03),
    Category("gambling_net", True, 50, 1.0, 0.015),
    Category("money_transfer", True, 150, 0.9, 0.015),
    Category("gift_cards", True, 50, 0.6, 0.01),
)
CAT_NAMES = np.array([c.name for c in CATEGORIES])
CAT_ONLINE = np.array([c.online for c in CATEGORIES])
CAT_MEDIAN = np.array([c.median_amount for c in CATEGORIES])
CAT_SIGMA = np.array([c.sigma for c in CATEGORIES])
CAT_SHARE = np.array([c.share for c in CATEGORIES])
CAT_INDEX = {name: i for i, name in enumerate(CAT_NAMES)}

# (name, lat, lon, relative population)
CITIES: tuple[tuple[str, float, float, float], ...] = (
    ("New York", 40.71, -74.01, 8.3),
    ("Los Angeles", 34.05, -118.24, 3.9),
    ("Chicago", 41.88, -87.63, 2.7),
    ("Houston", 29.76, -95.37, 2.3),
    ("Phoenix", 33.45, -112.07, 1.6),
    ("Philadelphia", 39.95, -75.17, 1.6),
    ("San Antonio", 29.42, -98.49, 1.5),
    ("San Diego", 32.72, -117.16, 1.4),
    ("Dallas", 32.78, -96.80, 1.3),
    ("San Francisco", 37.77, -122.42, 0.9),
    ("Seattle", 47.61, -122.33, 0.75),
    ("Denver", 39.74, -104.99, 0.7),
    ("Boston", 42.36, -71.06, 0.68),
    ("Miami", 25.76, -80.19, 0.45),
    ("Atlanta", 33.75, -84.39, 0.5),
    ("Minneapolis", 44.98, -93.27, 0.43),
    ("Portland", 45.52, -122.68, 0.65),
    ("Las Vegas", 36.17, -115.14, 0.65),
    ("Nashville", 36.16, -86.78, 0.69),
    ("Detroit", 42.33, -83.05, 0.63),
)
CITY_LAT = np.array([c[1] for c in CITIES])
CITY_LON = np.array([c[2] for c in CITIES])
CITY_WEIGHT = np.array([c[3] for c in CITIES]) / sum(c[3] for c in CITIES)

# Relative transaction volume Monday..Sunday (weekends busier for retail).
WEEKDAY_FACTOR = np.array([0.95, 0.97, 1.0, 1.02, 1.12, 1.18, 0.90])

TX_PER_CARD_PER_DAY = 0.6
CARDS_PER_CUSTOMER = 1.15
ONLINE_MERCHANT_SHARE = 0.2
N_FAVORITES = 12
FAVORITE_PROB = 0.85  # share of card-present spend at a card's usual merchants
TRIP_PROB = 0.25  # share of cards with one legitimate out-of-town trip per period
SECONDS_PER_DAY = 86_400

# Fraud mix in the history period (share of fraudulent *transactions*).
FRAUD_MIX: dict[str, float] = {
    "card_testing": 0.30,
    "impossible_travel": 0.20,
    "high_amount": 0.15,
    "night_burst": 0.15,
    "compromised_merchant": 0.20,
}
# Average transactions per fraud event, used to turn a row budget into events.
FRAUD_EVENT_SIZE: dict[str, float] = {
    "card_testing": 6.5,
    "impossible_travel": 2.0,
    "high_amount": 1.0,
    "night_burst": 4.5,
    "compromised_merchant": 27.5,
    NEW_FRAUD_PATTERN: 3.0,
}


@dataclass(frozen=True)
class DriftProfile:
    """How a production scenario differs from the history period."""

    amount_multiplier: float = 1.0
    volume_multiplier: float = 1.0
    online_share_multiplier: float = 1.0
    online_category_boost: dict[str, float] = field(default_factory=dict)
    new_pattern_rate: float = 0.0  # extra fraud rows (share of normal rows)


DRIFT_PROFILES: dict[str, DriftProfile] = {
    "none": DriftProfile(),
    # A fraud ring starts cashing out via gift cards from the victim's own
    # device during normal hours: none of the existing patterns' signals fire.
    "new_fraud_pattern": DriftProfile(new_pattern_rate=0.006),
    # Holiday season / inflation: people spend ~40% more, ~20% more often.
    "spending_shift": DriftProfile(amount_multiplier=1.4, volume_multiplier=1.2),
    # Customers move to online shopping, mostly retail and electronics.
    "merchant_mix_shift": DriftProfile(
        online_share_multiplier=2.2,
        online_category_boost={"shopping_net": 2.0, "electronics_net": 2.5},
    ),
}


# --------------------------------------------------------------------------- #
# Population (customers, cards, merchants) shared by all periods
# --------------------------------------------------------------------------- #


@dataclass
class Population:
    """All entities, stored as NumPy arrays indexed by integer id."""

    cust_city: np.ndarray
    cust_lat: np.ndarray
    cust_lon: np.ndarray
    cust_amount_mult: np.ndarray
    cust_peak_hour: np.ndarray
    cust_online_share: np.ndarray
    cust_n_devices: np.ndarray
    card_customer: np.ndarray
    card_activity: np.ndarray
    card_favorites: np.ndarray  # (n_cards, N_FAVORITES) merchant indices
    merch_category: np.ndarray
    merch_city: np.ndarray
    merch_lat: np.ndarray
    merch_lon: np.ndarray
    merch_online: np.ndarray
    city_cp_merchants: np.ndarray  # (n_cities, max_per_city) padded with -1
    city_cp_count: np.ndarray

    @property
    def n_cards(self) -> int:
        return len(self.card_customer)

    def merchants_in(self, categories: list[str]) -> np.ndarray:
        """Indices of merchants in any of the given categories."""
        cat_ids = [CAT_INDEX[c] for c in categories]
        return np.flatnonzero(np.isin(self.merch_category, cat_ids))

    def random_cp_merchant_in_city(self, rng: np.random.Generator, city: np.ndarray) -> np.ndarray:
        """One random card-present merchant per entry of ``city`` (vectorised)."""
        pick = (rng.random(len(city)) * self.city_cp_count[city]).astype(int)
        return self.city_cp_merchants[city, pick]


def build_population(sample_size: int, history_days: int, seed: int) -> Population:
    """Create customers, cards and merchants sized for ``sample_size`` transactions."""
    rng = np.random.default_rng([seed, 0])
    n_cities = len(CITIES)

    n_cards = max(60, round(sample_size / (TX_PER_CARD_PER_DAY * history_days)))
    n_customers = max(50, round(n_cards / CARDS_PER_CUSTOMER))
    n_cards = max(n_cards, n_customers)

    cust_city = rng.choice(n_cities, n_customers, p=CITY_WEIGHT)
    cust_lat = CITY_LAT[cust_city] + rng.normal(0, 0.2, n_customers)
    cust_lon = CITY_LON[cust_city] + rng.normal(0, 0.2, n_customers)
    cust_amount_mult = rng.lognormal(0, 0.35, n_customers)
    cust_peak_hour = np.clip(rng.normal(15, 2.5, n_customers), 9, 21)
    cust_online_share = rng.beta(2, 5, n_customers)
    cust_n_devices = 1 + (rng.random(n_customers) < 0.3).astype(int)

    # Every customer gets one card; ~15% get a second one.
    extra_cards = rng.integers(0, n_customers, n_cards - n_customers)
    card_customer = np.concatenate([np.arange(n_customers), extra_cards])
    card_activity = rng.lognormal(0, 0.6, n_cards)

    # Merchants: card-present ones live in a city; online ones have an HQ city.
    n_merchants = max(150, sample_size // 250)
    online_cats = np.flatnonzero(CAT_ONLINE)
    cp_cats = np.flatnonzero(~CAT_ONLINE)
    n_online = max(len(online_cats), round(n_merchants * ONLINE_MERCHANT_SHARE))
    n_cp = n_merchants - n_online

    def assign_categories(cat_ids: np.ndarray, n: int) -> np.ndarray:
        # Guarantee every category exists at least once, then sample by share.
        p = CAT_SHARE[cat_ids] / CAT_SHARE[cat_ids].sum()
        return np.concatenate([cat_ids, rng.choice(cat_ids, n - len(cat_ids), p=p)])

    merch_category = np.concatenate(
        [assign_categories(cp_cats, n_cp), assign_categories(online_cats, n_online)]
    )
    cp_city = np.concatenate(
        [np.arange(n_cities), rng.choice(n_cities, n_cp - n_cities, p=CITY_WEIGHT)]
    )
    merch_city = np.concatenate([cp_city, rng.choice(n_cities, n_online, p=CITY_WEIGHT)])
    merch_online = np.concatenate([np.zeros(n_cp, bool), np.ones(n_online, bool)])
    merch_lat = CITY_LAT[merch_city] + rng.normal(0, 0.15, n_merchants)
    merch_lon = CITY_LON[merch_city] + rng.normal(0, 0.15, n_merchants)

    # Lookup table: city -> its card-present merchants (padded with -1).
    city_cp_count = np.bincount(cp_city, minlength=n_cities)
    city_cp_merchants = np.full((n_cities, city_cp_count.max()), -1, dtype=int)
    fill = np.zeros(n_cities, dtype=int)
    for m, c in enumerate(cp_city):
        city_cp_merchants[c, fill[c]] = m
        fill[c] += 1

    # Each card has a set of "usual" merchants near the cardholder's home.
    card_favorites = np.empty((n_cards, N_FAVORITES), dtype=int)
    for card in range(n_cards):
        city = cust_city[card_customer[card]]
        local = city_cp_merchants[city, : city_cp_count[city]]
        card_favorites[card] = rng.choice(local, N_FAVORITES, replace=len(local) < N_FAVORITES)

    return Population(
        cust_city=cust_city,
        cust_lat=cust_lat,
        cust_lon=cust_lon,
        cust_amount_mult=cust_amount_mult,
        cust_peak_hour=cust_peak_hour,
        cust_online_share=cust_online_share,
        cust_n_devices=cust_n_devices,
        card_customer=card_customer,
        card_activity=card_activity,
        card_favorites=card_favorites,
        merch_category=merch_category,
        merch_city=merch_city,
        merch_lat=merch_lat,
        merch_lon=merch_lon,
        merch_online=merch_online,
        city_cp_merchants=city_cp_merchants,
        city_cp_count=city_cp_count,
    )


# --------------------------------------------------------------------------- #
# Normal (legitimate) behaviour
# --------------------------------------------------------------------------- #


def _online_merchant_probs(
    pop: Population, boost: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Online merchants and the probability that an online purchase goes to each."""
    online = np.flatnonzero(pop.merch_online)
    cats = pop.merch_category[online]
    cat_weight = CAT_SHARE.copy()
    for name, mult in boost.items():
        cat_weight[CAT_INDEX[name]] *= mult
    per_cat_count = np.bincount(cats, minlength=len(CATEGORIES))
    w = cat_weight[cats] / per_cat_count[cats]
    return online, w / w.sum()


def _device_ids(cust: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Format customer device ids like ``DEV004211-0``."""
    return np.char.add(
        np.char.add("DEV", np.char.zfill(cust.astype(str), 6)), np.char.add("-", k.astype(str))
    )


def _generate_normal(
    pop: Population,
    rng: np.random.Generator,
    n: int,
    start: pd.Timestamp,
    n_days: int,
    drift: DriftProfile,
) -> dict[str, np.ndarray]:
    """Draw ``n`` legitimate transactions, vectorised."""
    # Which card: more active cards transact more.
    counts = rng.multinomial(n, pop.card_activity / pop.card_activity.sum())
    card = np.repeat(np.arange(pop.n_cards), counts)
    cust = pop.card_customer[card]

    # When: weekly seasonality plus a slight upward trend; hour of day around
    # each customer's personal peak (which naturally yields a few night txns).
    weekday = (start.weekday() + np.arange(n_days)) % 7
    day_w = WEEKDAY_FACTOR[weekday] * (1 + 0.1 * np.arange(n_days) / n_days)
    day = rng.choice(n_days, size=n, p=day_w / day_w.sum())
    hour = (pop.cust_peak_hour[cust] + rng.normal(0, 3.5, n)) % 24
    epoch = int(start.timestamp()) + day * SECONDS_PER_DAY + (hour * 3600).astype(np.int64)

    # Where: online vs card-present, then which merchant.
    online_share = np.clip(pop.cust_online_share[cust] * drift.online_share_multiplier, 0, 0.95)
    is_online = rng.random(n) < online_share
    merchant = np.empty(n, dtype=int)

    online_merchants, online_p = _online_merchant_probs(pop, drift.online_category_boost)
    idx_on = np.flatnonzero(is_online)
    merchant[idx_on] = rng.choice(online_merchants, size=len(idx_on), p=online_p)

    idx_cp = np.flatnonzero(~is_online)
    zipf = 1 / np.arange(1, N_FAVORITES + 1) ** 0.8
    fav_pick = rng.choice(N_FAVORITES, size=len(idx_cp), p=zipf / zipf.sum())
    favorite = pop.card_favorites[card[idx_cp], fav_pick]
    local = pop.random_cp_merchant_in_city(rng, pop.cust_city[cust[idx_cp]])
    merchant[idx_cp] = np.where(rng.random(len(idx_cp)) < FAVORITE_PROB, favorite, local)

    # Legitimate travel: some cards spend a few days in another city. This
    # keeps "far from home" from being a trivially perfect fraud signal.
    n_cities = len(CITIES)
    has_trip = rng.random(pop.n_cards) < TRIP_PROB
    trip_start = rng.integers(0, max(1, n_days - 5), pop.n_cards)
    trip_len = rng.integers(2, 6, pop.n_cards)
    home_city = pop.cust_city[pop.card_customer]
    trip_city = (home_city + rng.integers(1, n_cities, pop.n_cards)) % n_cities
    on_trip = (
        ~is_online
        & has_trip[card]
        & (day >= trip_start[card])
        & (day < trip_start[card] + trip_len[card])
    )
    idx_trip = np.flatnonzero(on_trip)
    merchant[idx_trip] = pop.random_cp_merchant_in_city(rng, trip_city[card[idx_trip]])

    # How much: category ticket size x customer's spending level x noise.
    cat = pop.merch_category[merchant]
    amount = (
        CAT_MEDIAN[cat]
        * pop.cust_amount_mult[cust]
        * np.exp(rng.normal(0, CAT_SIGMA[cat]))
        * drift.amount_multiplier
    )

    # Device: online purchases come from one of the customer's own devices.
    device_k = np.where((pop.cust_n_devices[cust] == 2) & (rng.random(n) < 0.2), 1, 0)
    device = np.where(pop.merch_online[merchant], _device_ids(cust, device_k), None)

    return {
        "card": card,
        "merchant": merchant,
        "epoch": epoch,
        "amount": np.maximum(np.round(amount, 2), 0.5),
        "device_id": device.astype(object),
        "is_fraud": np.zeros(n, dtype=np.int32),
        "fraud_pattern": np.full(n, "none", dtype=object),
    }


# --------------------------------------------------------------------------- #
# Fraud injection
# --------------------------------------------------------------------------- #


class _FraudInjector:
    """Builds fraudulent transactions, one event (attack) at a time."""

    def __init__(
        self,
        pop: Population,
        rng: np.random.Generator,
        start: pd.Timestamp,
        n_days: int,
        normal: dict[str, np.ndarray],
        period_tag: str,
    ) -> None:
        self.pop = pop
        self.rng = rng
        self.start_epoch = int(start.timestamp())
        self.end_epoch = self.start_epoch + n_days * SECONDS_PER_DAY - 1
        self.n_days = n_days
        self.period_tag = period_tag
        self.rows: list[tuple[int, int, int, float, str | None, str]] = []
        self._events = 0

        # Per-card mean legit spend (for "amount far above normal").
        sums = np.bincount(normal["card"], weights=normal["amount"], minlength=pop.n_cards)
        cnts = np.bincount(normal["card"], minlength=pop.n_cards)
        self.card_mean_amount = np.where(cnts > 0, sums / np.maximum(cnts, 1), 60.0)

        # Legit card-present transactions: anchors for impossible travel.
        cp = ~pop.merch_online[normal["merchant"]]
        self.cp_card = normal["card"][cp]
        self.cp_epoch = normal["epoch"][cp]
        self.cp_merchant = normal["merchant"][cp]

    # -- helpers ------------------------------------------------------------ #
    def _new_fraud_device(self) -> str:
        return f"FDEV{self.period_tag}{self._events:06d}"

    def _random_time(self, margin_s: int = 6 * 3600) -> int:
        return int(self.rng.integers(self.start_epoch, self.end_epoch - margin_s))

    def _amount(self, median: float, sigma: float) -> float:
        return round(float(median * np.exp(self.rng.normal(0, sigma))), 2)

    def _add(self, card: int, merchant: int, epoch: int, amount: float, device, pattern: str):
        epoch = min(int(epoch), self.end_epoch)
        if not self.pop.merch_online[merchant]:
            device = None  # card-present: no customer device involved
        elif device is None:
            # Online fraud without a fraudster device = account takeover from
            # the victim's own (compromised) device.
            cust = np.array([self.pop.card_customer[card]])
            device = str(_device_ids(cust, np.array([0]))[0])
        self.rows.append((card, merchant, epoch, max(amount, 0.5), device, pattern))

    def _pick(self, candidates: np.ndarray) -> int:
        return int(self.rng.choice(candidates))

    # -- patterns ----------------------------------------------------------- #
    def card_testing(self) -> None:
        """Burst of tiny online charges to check the card works, then a big one."""
        card = int(self.rng.integers(self.pop.n_cards))
        device = self._new_fraud_device()
        small = self.pop.merchants_in(["misc_net", "gift_cards", "shopping_net"])
        big = self.pop.merchants_in(["electronics_net", "shopping_net"])
        t = self._random_time()
        for _ in range(int(self.rng.integers(3, 9))):
            t += int(self.rng.uniform(20, 300))
            self._add(
                card,
                self._pick(small),
                t,
                round(self.rng.uniform(0.5, 3.0), 2),
                device,
                "card_testing",
            )
        t += int(self.rng.uniform(300, 3600))
        self._add(card, self._pick(big), t, self._amount(600, 0.6), device, "card_testing")

    def impossible_travel(self) -> None:
        """Card used in a city >1500 km from where it was used minutes earlier."""
        i = int(self.rng.integers(len(self.cp_card)))
        card, t, last_merchant = int(self.cp_card[i]), int(self.cp_epoch[i]), self.cp_merchant[i]
        last_city = self.pop.merch_city[last_merchant]
        dist = haversine_km(CITY_LAT[last_city], CITY_LON[last_city], CITY_LAT, CITY_LON)
        far_city = self._pick(np.flatnonzero(dist > 1500))
        t += int(self.rng.uniform(10, 90) * 60)
        for _ in range(int(self.rng.integers(1, 4))):
            m = int(self.pop.random_cp_merchant_in_city(self.rng, np.array([far_city]))[0])
            self._add(card, m, t, self._amount(200, 0.6), None, "impossible_travel")
            t += int(self.rng.uniform(5, 30) * 60)

    def high_amount(self) -> None:
        """One purchase 8-25x the card's normal spend at a merchant it never used."""
        card = int(self.rng.integers(self.pop.n_cards))
        candidates = self.pop.merchants_in(["electronics_net", "travel", "shopping_pos"])
        candidates = np.setdiff1d(candidates, self.pop.card_favorites[card])
        amount = min(self.card_mean_amount[card] * self.rng.uniform(8, 25), 15_000)
        device = self._new_fraud_device() if self.rng.random() < 0.6 else None
        self._add(
            card,
            self._pick(candidates),
            self._random_time(),
            round(amount, 2),
            device,
            "high_amount",
        )

    def night_burst(self) -> None:
        """Several 1-4am transactions in high-risk categories within ~90 minutes."""
        card = int(self.rng.integers(self.pop.n_cards))
        device = self._new_fraud_device()
        risky = self.pop.merchants_in(["gambling_net", "money_transfer"])
        day = int(self.rng.integers(self.n_days - 1))
        t = self.start_epoch + day * SECONDS_PER_DAY + int(self.rng.uniform(1, 3.5) * 3600)
        for _ in range(int(self.rng.integers(3, 7))):
            self._add(card, self._pick(risky), t, self._amount(120, 0.7), device, "night_burst")
            t += int(self.rng.uniform(3, 25) * 60)

    def compromised_merchant(self) -> None:
        """A ring cashes out many stolen cards at one online merchant within an hour."""
        merchant = self._pick(
            self.pop.merchants_in(["shopping_net", "electronics_net", "misc_net"])
        )
        n_cards = min(int(self.rng.integers(15, 41)), self.pop.n_cards)
        cards = self.rng.choice(self.pop.n_cards, n_cards, replace=False)
        ring_devices = [f"{self._new_fraud_device()}-{k}" for k in range(3)]
        t0 = self._random_time()
        for card in cards:
            self._add(
                int(card),
                merchant,
                t0 + int(self.rng.uniform(0, 3600)),
                self._amount(90, 0.5),
                self.rng.choice(ring_devices),
                "compromised_merchant",
            )

    def gift_card_cashout(self) -> None:
        """Drift-only pattern: gift cards bought from the victim's own device by day."""
        card = int(self.rng.integers(self.pop.n_cards))
        cust = self.pop.card_customer[card]
        device = str(_device_ids(np.array([cust]), np.array([0]))[0])
        merchants = self.pop.merchants_in(["gift_cards"])
        day = int(self.rng.integers(self.n_days - 1))
        for _ in range(int(self.rng.integers(2, 5))):
            t = self.start_epoch + day * SECONDS_PER_DAY + int(self.rng.uniform(10, 20) * 3600)
            amount = float(self.rng.choice([50, 100, 150, 200, 250]))
            self._add(card, self._pick(merchants), t, amount, device, NEW_FRAUD_PATTERN)

    # -- driver ------------------------------------------------------------- #
    def inject(self, pattern: str, row_budget: float) -> None:
        """Generate enough events of ``pattern`` to roughly fill ``row_budget`` rows."""
        n_events = max(1, round(row_budget / FRAUD_EVENT_SIZE[pattern]))
        make = getattr(self, pattern)
        for _ in range(n_events):
            self._events += 1
            make()

    def as_arrays(self) -> dict[str, np.ndarray]:
        card, merchant, epoch, amount, device, pattern = zip(*self.rows, strict=True)
        n = len(self.rows)
        return {
            "card": np.array(card, dtype=int),
            "merchant": np.array(merchant, dtype=int),
            "epoch": np.array(epoch, dtype=np.int64),
            "amount": np.array(amount, dtype=float),
            "device_id": np.array(device, dtype=object),
            "is_fraud": np.ones(n, dtype=np.int32),
            "fraud_pattern": np.array(pattern, dtype=object),
        }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _ids(prefix: str, values: np.ndarray, width: int) -> np.ndarray:
    return np.char.add(prefix, np.char.zfill(values.astype(str), width))


def _assemble(
    pop: Population,
    parts: list[dict[str, np.ndarray]],
    rng: np.random.Generator,
    id_prefix: str,
    label_delay_median_days: float,
    chargeback_window_days: int,
) -> pd.DataFrame:
    """Combine normal + fraud rows, sort by time and attach entity attributes."""
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(cols["epoch"], kind="stable")
    cols = {k: v[order] for k, v in cols.items()}
    n = len(order)

    card, merchant = cols["card"], cols["merchant"]
    cust = pop.card_customer[card]
    ts = pd.to_datetime(cols["epoch"], unit="s", utc=True)

    # Late-arriving labels: fraud is confirmed after a chargeback delay; a legit
    # transaction counts as "confirmed legit" once the chargeback window closes.
    fraud_delay_days = np.clip(rng.lognormal(np.log(label_delay_median_days), 0.7, n), 1, 60)
    delay_days = np.where(cols["is_fraud"] == 1, fraud_delay_days, chargeback_window_days)
    label_ts = ts + pd.to_timedelta(delay_days * SECONDS_PER_DAY, unit="s").round("s")

    df = pd.DataFrame(
        {
            "transaction_id": _ids(id_prefix, np.arange(1, n + 1), 9),
            "card_id": _ids("CARD", card, 7),
            "customer_id": _ids("CUST", cust, 6),
            "merchant_id": _ids("MER", merchant, 6),
            "merchant_category": CAT_NAMES[pop.merch_category[merchant]],
            "amount": cols["amount"],
            "currency": "USD",
            "ts": ts,
            "customer_lat": np.round(pop.cust_lat[cust], 5),
            "customer_lon": np.round(pop.cust_lon[cust], 5),
            "merchant_lat": np.round(pop.merch_lat[merchant], 5),
            "merchant_lon": np.round(pop.merch_lon[merchant], 5),
            "channel": np.where(pop.merch_online[merchant], "online", "card_present"),
            "device_id": cols["device_id"],
            "is_fraud": cols["is_fraud"].astype(np.int32),
            "fraud_pattern": cols["fraud_pattern"],
            "label_available_ts": label_ts,
        }
    )
    for c in (
        "transaction_id",
        "card_id",
        "customer_id",
        "merchant_id",
        "merchant_category",
        "channel",
    ):
        df[c] = df[c].astype(object)
    return df


def inject_dirty_records(
    df: pd.DataFrame, rng: np.random.Generator, rate: float, id_prefix: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add realistic feed problems for the silver layer to catch.

    Half of the dirty budget is exact duplicates (the upstream system re-sent a
    message); the other half are malformed copies with a new id and one broken
    field. Returns the dirty frame and counts by problem type.
    """
    n_dirty = round(len(df) * rate)
    n_dup = n_dirty // 2
    n_bad = n_dirty - n_dup
    if n_dirty == 0:
        return df, {"duplicates": 0}

    dups = df.iloc[rng.choice(len(df), n_dup, replace=False)]

    bad = df.iloc[rng.choice(len(df), n_bad, replace=False)].copy().reset_index(drop=True)
    bad["transaction_id"] = _ids(f"{id_prefix}X", np.arange(1, n_bad + 1), 8).astype(object)
    bad["is_fraud"] = np.int32(0)
    bad["fraud_pattern"] = "none"
    problems = np.array(["null_card_id", "non_positive_amount", "invalid_coordinates", "null_ts"])
    kind = problems[np.arange(n_bad) % len(problems)]
    bad.loc[kind == "null_card_id", "card_id"] = None
    bad.loc[kind == "non_positive_amount", "amount"] = -bad.loc[
        kind == "non_positive_amount", "amount"
    ]
    bad.loc[kind == "invalid_coordinates", "merchant_lat"] = 123.456
    bad.loc[kind == "null_ts", "ts"] = pd.NaT

    out = pd.concat([df, dups, bad], ignore_index=True)
    out = out.sort_values("ts", kind="stable", na_position="last").reset_index(drop=True)
    counts = {"duplicates": int(n_dup)}
    counts.update({str(k): int((kind == k).sum()) for k in problems})
    return out, counts


def generate_period(
    settings: Settings,
    period: Period,
    scenario: str = "none",
    population: Population | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the raw feed for one period. Returns (dataframe, summary)."""
    if scenario not in DRIFT_SCENARIOS:
        raise ValueError(f"unknown drift scenario {scenario!r}; choose from {DRIFT_SCENARIOS}")
    if period == "history" and scenario != "none":
        raise ValueError("drift scenarios only apply to the production period")

    d = settings.data
    pop = population or build_population(d.sample_size, d.history_days, settings.seed)
    history_start = pd.Timestamp(d.start_date, tz="UTC")

    if period == "history":
        rng = np.random.default_rng([settings.seed, 1])
        start, n_days, drift = history_start, d.history_days, DRIFT_PROFILES["none"]
        n_normal, id_prefix, tag = d.sample_size, "H", "H"
    else:
        idx = DRIFT_SCENARIOS.index(scenario)
        rng = np.random.default_rng([settings.seed, 2, idx])
        start = history_start + pd.Timedelta(days=d.history_days)
        n_days, drift = d.production_days, DRIFT_PROFILES[scenario]
        base = d.sample_size * d.production_days / d.history_days
        n_normal = round(base * drift.volume_multiplier)
        id_prefix, tag = "P", f"P{idx}"

    normal = _generate_normal(pop, rng, n_normal, start, n_days, drift)

    injector = _FraudInjector(pop, rng, start, n_days, normal, tag)
    budget = d.fraud_rate * n_normal
    for pattern, share in FRAUD_MIX.items():
        injector.inject(pattern, budget * share)
    if drift.new_pattern_rate > 0:
        injector.inject(NEW_FRAUD_PATTERN, drift.new_pattern_rate * n_normal)

    df = _assemble(
        pop,
        [normal, injector.as_arrays()],
        rng,
        id_prefix,
        d.label_delay_median_days,
        d.chargeback_window_days,
    )
    # Summarise the clean ground truth before the feed gets dirtied.
    fraud = df[df["is_fraud"] == 1]
    summary = {
        "period": period,
        "scenario": scenario,
        "clean_rows": len(df),
        "start": str(start.date()),
        "days": n_days,
        "cards": pop.n_cards,
        "customers": int(len(pop.cust_city)),
        "merchants": int(len(pop.merch_category)),
        "fraud_rows": len(fraud),
        "fraud_rate": round(len(fraud) / len(df), 5),
        "fraud_by_pattern": {
            k: int(v) for k, v in fraud["fraud_pattern"].value_counts().sort_index().items()
        },
        "online_share": round(float((df["channel"] == "online").mean()), 4),
        "mean_amount": round(float(df["amount"].mean()), 2),
    }
    df, dirty = inject_dirty_records(df, rng, d.dirty_rate, id_prefix)
    summary["rows"] = len(df)
    summary["dirty_records"] = dirty
    return df, summary


def write_landing(df: pd.DataFrame, directory: str) -> Path:
    """Write the raw feed as a single parquet file in ``directory`` (overwrites)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    path = out_dir / "transactions.parquet"
    # Spark can't read nanosecond parquet timestamps, so store microseconds.
    df.to_parquet(path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    return path


def run(settings: Settings, scenarios: list[str] | None = None) -> dict[str, Any]:
    """Generate history plus the requested production scenarios and write them."""
    scenarios = scenarios or list(DRIFT_SCENARIOS)
    d = settings.data
    pop = build_population(d.sample_size, d.history_days, settings.seed)

    summaries: dict[str, Any] = {"seed": settings.seed, "sample_size": d.sample_size}
    df, summary = generate_period(settings, "history", population=pop)
    path = write_landing(df, settings.paths.landing_history)
    log.info(
        "history: %s rows (fraud rate %.3f%%) -> %s",
        f"{len(df):,}",
        100 * summary["fraud_rate"],
        path,
    )
    summaries["history"] = summary

    summaries["production"] = {}
    for scenario in scenarios:
        df, summary = generate_period(settings, "production", scenario, population=pop)
        path = write_landing(df, settings.paths.landing_production(scenario))
        log.info(
            "production[%s]: %s rows (fraud rate %.3f%%) -> %s",
            scenario,
            f"{len(df):,}",
            100 * summary["fraud_rate"],
            path,
        )
        summaries["production"][scenario] = summary

    write_json(settings.paths.report("generation_summary.json"), summaries)
    return summaries
