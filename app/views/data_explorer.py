"""Data Explorer: what the synthetic data looks like and how clean it is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data
from lib.charts import FRAUD, INK_2, LEGIT, PLOTLY_CONFIG, SERIES, style

from sentinelpay.data.generator import FRAUD_PATTERNS
from sentinelpay.utils.geo import haversine_km

PATTERN_INFO = {
    "card_testing": (
        "Card testing",
        "A fraudster checks a stolen card with a burst of tiny online charges, "
        "then makes one large purchase once they know it works.",
    ),
    "impossible_travel": (
        "Impossible travel",
        "The card is used in person in a city over 1,500 km away, minutes after a "
        "genuine purchase. Nobody travels that fast, so the card has been cloned.",
    ),
    "high_amount": (
        "High-amount anomaly",
        "One purchase 8–25× bigger than this card's usual spend, at a merchant the "
        "cardholder has never used.",
    ),
    "night_burst": (
        "Night-time burst",
        "Several gambling or money-transfer charges between 1 and 6 am, when the "
        "cardholder is normally asleep.",
    ),
    "compromised_merchant": (
        "Compromised merchant",
        "A fraud ring uses dozens of different stolen cards at one online shop within "
        "an hour, from a handful of shared devices.",
    ),
}

st.title("🔎 Data Explorer")
st.markdown(
    "SentinelPay generates realistic card transactions, hides five kinds of fraud in "
    "them, and cleans the feed before any model sees it. This page shows that data. "
    "**All data is synthetic.**"
)

if not data.outputs_exist():
    st.warning("No pipeline outputs found yet. Run `make demo` first, then reload this page.")
    st.stop()

df = data.silver()
dq = data.report("dq_metrics.json")
gen = data.report("generation_summary.json")
fraud = df[df["is_fraud"] == 1]

tab_overview, tab_patterns, tab_quality, tab_drift = st.tabs(
    ["Overview", "Fraud patterns", "Data quality", "Drift scenarios"]
)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    c = st.columns(5)
    c[0].metric("Clean transactions", f"{len(df):,}")
    c[1].metric("Fraud rate", f"{df['is_fraud'].mean():.2%}")
    c[2].metric("Cards", f"{df['card_id'].nunique():,}")
    c[3].metric("Merchants", f"{df['merchant_id'].nunique():,}")
    c[4].metric(
        "Period",
        f"{df['ts'].min():%b}–{df['ts'].max():%b %Y}",
        help="History period used for training. A further month is held back as "
        "'production' for streaming and drift monitoring.",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Daily transactions")
        daily = df.set_index("ts").resample("D").size().rename("transactions").reset_index()
        fig = go.Figure(
            go.Scatter(
                x=daily["ts"],
                y=daily["transactions"],
                mode="lines",
                line=dict(color=LEGIT, width=2),
                hovertemplate="%{x|%a %d %b %Y}<br>%{y:,} transactions<extra></extra>",
            )
        )
        st.plotly_chart(style(fig, legend=False), width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            "Weekly rhythm (busier Fridays and Saturdays) and slow growth, like a real "
            "card portfolio. Models must cope with this seasonality."
        )

    with right:
        st.subheader("Fraud per week, by pattern")
        weekly = (
            fraud.assign(week=fraud["ts"].dt.tz_convert(None).dt.to_period("W").dt.start_time)
            .groupby(["week", "fraud_pattern"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(FRAUD_PATTERNS), fill_value=0)
        )
        fig = go.Figure()
        for i, p in enumerate(FRAUD_PATTERNS):
            fig.add_bar(
                x=weekly.index,
                y=weekly[p],
                name=PATTERN_INFO[p][0],
                marker=dict(color=SERIES[i], line=dict(color="#fcfcfb", width=1)),
                hovertemplate=f"{PATTERN_INFO[p][0]}<br>"
                + "week of %{x|%d %b}: %{y} txns<extra></extra>",
            )
        fig.update_layout(barmode="stack", legend_traceorder="normal")
        st.plotly_chart(style(fig), width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            "Fraud is rare (under 1% of transactions) and arrives in bursts. A "
            "compromised-merchant attack alone can add dozens of fraudulent rows in one week."
        )

    left, right = st.columns(2)

    with left:
        st.subheader("How much is spent")
        edges = np.logspace(np.log10(0.5), np.log10(20_000), 45)
        centers = np.sqrt(edges[:-1] * edges[1:])
        fig = go.Figure()
        for label, subset, color in (
            ("Legitimate", df[df["is_fraud"] == 0], LEGIT),
            ("Fraud", fraud, FRAUD),
        ):
            counts, _ = np.histogram(subset["amount"], bins=edges)
            fig.add_scatter(
                x=centers,
                y=100 * counts / counts.sum(),
                name=label,
                mode="lines",
                line=dict(color=color, width=2, shape="spline"),
                hovertemplate=f"{label}<br>"
                + "around $%{x:,.2f}: %{y:.1f}% of txns<extra></extra>",
            )
        fig.update_xaxes(type="log", title="Amount (USD, log scale)", tickprefix="$")
        fig.update_yaxes(title="% of transactions", ticksuffix="%")
        st.plotly_chart(style(fig), width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            "Fraud has two peaks: tiny card-testing charges under $3, and unusually large "
            "purchases. Genuine spending sits in between."
        )

    with right:
        st.subheader("When it happens")
        fig = go.Figure()
        for label, subset, color in (
            ("Legitimate", df[df["is_fraud"] == 0], LEGIT),
            ("Fraud", fraud, FRAUD),
        ):
            hours = subset["ts"].dt.hour
            share = hours.value_counts(normalize=True).reindex(range(24), fill_value=0)
            fig.add_scatter(
                x=share.index,
                y=100 * share.values,
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color, line=dict(color="#fcfcfb", width=2)),
                hovertemplate=f"{label}<br>" + "%{x}:00–%{x}:59 — %{y:.1f}% of txns<extra></extra>",
            )
        fig.update_xaxes(title="Hour of day (UTC)", dtick=3)
        fig.update_yaxes(title="% of transactions", ticksuffix="%")
        st.plotly_chart(style(fig), width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            "Genuine spending peaks in the afternoon. Fraud has an extra bump between 1 and "
            "5 am, from night-time bursts. Hour of day will be a useful model feature."
        )


# --------------------------------------------------------------------------- #
# Fraud patterns
# --------------------------------------------------------------------------- #
def _example_events(pattern: str) -> pd.DataFrame:
    """One row per attack: the entity it targeted and when it started."""
    rows = fraud[fraud["fraud_pattern"] == pattern]
    key = "merchant_id" if pattern == "compromised_merchant" else "card_id"
    events = rows.groupby(key)["ts"].min().rename("start").reset_index()
    # Attacks in the first days have no genuine history to compare against,
    # so they go to the end of the list.
    events["early"] = events["start"] < df["ts"].min() + pd.Timedelta(days=3)
    return events.sort_values(["early", "start"]).reset_index(drop=True)


def _event_window(pattern: str, event: pd.Series) -> pd.DataFrame:
    if pattern == "compromised_merchant":
        lo, hi = event["start"] - pd.Timedelta(hours=1), event["start"] + pd.Timedelta(hours=2)
        sel = (df["merchant_id"] == event["merchant_id"]) & df["ts"].between(lo, hi)
    else:
        lo, hi = event["start"] - pd.Timedelta(days=3), event["start"] + pd.Timedelta(hours=12)
        sel = (df["card_id"] == event["card_id"]) & df["ts"].between(lo, hi)
    return df[sel].sort_values("ts")


with tab_patterns:
    st.markdown(
        "Pick a pattern to see **one real attack** from the data, with the cardholder's "
        "genuine transactions around it for context."
    )
    names = {PATTERN_INFO[p][0]: p for p in FRAUD_PATTERNS}
    c1, c2 = st.columns([2, 1])

    def _reset_example() -> None:
        # Different patterns have different numbers of examples.
        st.session_state["example"] = 1

    pattern = names[
        c1.radio(
            "Fraud pattern", list(names), horizontal=True, key="pattern", on_change=_reset_example
        )
    ]
    events = _example_events(pattern)
    idx = c2.number_input(
        f"Example (1–{len(events)})",
        min_value=1,
        max_value=len(events),
        step=1,
        key="example",
    )
    event = events.iloc[int(idx) - 1]
    window = _event_window(pattern, event)

    title, story = PATTERN_INFO[pattern]
    st.info(f"**{title}.** {story}")

    n_fraud = int(window["is_fraud"].sum())
    m = st.columns(4)
    m[0].metric("Fraudulent transactions", n_fraud)
    m[1].metric("Genuine transactions shown", len(window) - n_fraud)
    m[2].metric("Fraud amount", f"${window.loc[window['is_fraud'] == 1, 'amount'].sum():,.2f}")
    if pattern == "compromised_merchant":
        m[3].metric("Different cards", window.loc[window["is_fraud"] == 1, "card_id"].nunique())
    else:
        m[3].metric("Card", event["card_id"])

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Timeline")
        fig = go.Figure()
        for label, flag, color, symbol in (
            ("Genuine", 0, LEGIT, "circle"),
            ("Fraud", 1, FRAUD, "diamond"),
        ):
            part = window[window["is_fraud"] == flag]
            fig.add_scatter(
                x=part["ts"],
                y=part["amount"],
                name=label,
                mode="markers",
                marker=dict(
                    size=11, color=color, symbol=symbol, line=dict(color="#fcfcfb", width=2)
                ),
                customdata=np.stack(
                    [part["merchant_category"], part["channel"], part["card_id"]], axis=1
                )
                if len(part)
                else None,
                hovertemplate="%{x|%d %b %H:%M}<br>$%{y:,.2f} · %{customdata[0]}"
                "<br>%{customdata[1]} · %{customdata[2]}<extra>" + label + "</extra>",
            )
        fig.update_yaxes(
            type="log",
            title="Amount (USD, log scale)",
            tickvals=[1, 10, 100, 1_000, 10_000],
            ticktext=["$1", "$10", "$100", "$1k", "$10k"],
        )
        st.plotly_chart(style(fig, height=360), width="stretch", config=PLOTLY_CONFIG)
        st.caption("Each dot is one transaction. Hover for details.")

    with right:
        st.subheader("Where")
        cp = window[window["channel"] == "card_present"]
        if pattern == "compromised_merchant" or cp.empty:
            st.caption(
                "This attack happened online, so there is no physical location to map. "
                "Online merchants only have a registered address."
            )
        else:
            # Scattergeo renders as SVG: no WebGL and no map tiles from the
            # internet, so it works in every browser and offline.
            home = window.iloc[0]
            fig = go.Figure()
            fig.add_scattergeo(
                lat=cp["merchant_lat"],
                lon=cp["merchant_lon"],
                mode="lines",
                line=dict(color=INK_2, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
            fig.add_scattergeo(
                lat=[home["customer_lat"]],
                lon=[home["customer_lon"]],
                mode="markers",
                name="Cardholder's home",
                marker=dict(size=14, color=SERIES[2], symbol="square"),
                hovertemplate="Cardholder's home<extra></extra>",
            )
            for label, flag, color, symbol in (
                ("Genuine", 0, LEGIT, "circle"),
                ("Fraud", 1, FRAUD, "diamond"),
            ):
                part = cp[cp["is_fraud"] == flag]
                fig.add_scattergeo(
                    lat=part["merchant_lat"],
                    lon=part["merchant_lon"],
                    mode="markers",
                    name=label,
                    marker=dict(
                        size=12, color=color, symbol=symbol, line=dict(color="white", width=2)
                    ),
                    text=part["ts"].dt.strftime("%d %b %H:%M")
                    + " · $"
                    + part["amount"].map("{:,.2f}".format),
                    hovertemplate=label + "<br>%{text}<extra></extra>",
                )
            fig.update_geos(
                scope="usa",
                showland=True,
                landcolor="#f0efec",
                showlakes=False,
                subunitcolor="#e1e0d9",
                countrycolor="#c3c2b7",
                bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(style(fig, height=360), width="stretch", config=PLOTLY_CONFIG)

            if pattern == "impossible_travel":
                seq = cp.reset_index(drop=True)
                first = seq.index[seq["is_fraud"] == 1][0]
                if first > 0:
                    a, b = seq.iloc[first - 1], seq.iloc[first]
                    km = float(
                        haversine_km(
                            a["merchant_lat"],
                            a["merchant_lon"],
                            b["merchant_lat"],
                            b["merchant_lon"],
                        )
                    )
                    hours = (b["ts"] - a["ts"]).total_seconds() / 3600
                    st.caption(
                        f"**{km:,.0f} km in {hours * 60:,.0f} minutes**, an implied speed of "
                        f"**{km / max(hours, 1e-6):,.0f} km/h**. A passenger jet flies at about "
                        "900 km/h."
                    )
            else:
                st.caption("Card-present purchases only. The line joins them in time order.")

    st.subheader("Transactions")
    shown = ["ts", "flag", "amount", "merchant_category", "channel", "card_id", "merchant_id"]
    table = window.assign(
        flag=np.where(window["is_fraud"] == 1, "🚩 fraud", "genuine"),
    )[[*shown, "device_id"]]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "ts": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM, HH:mm:ss"),
            "flag": "Label",
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "merchant_category": "Category",
            "channel": "Channel",
            "card_id": "Card",
            "merchant_id": "Merchant",
            "device_id": "Device",
        },
    )


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
with tab_quality:
    st.markdown(
        "Real payment feeds contain broken and re-sent records. The generator injects some "
        "on purpose, and the **silver** layer must catch every one before the data is used."
    )
    c = st.columns(4)
    c[0].metric("Raw rows received", f"{dq['bronze_rows']:,}")
    c[1].metric("Quarantined", f"{dq['quarantined_rows']:,}")
    c[2].metric("Duplicates removed", f"{dq['duplicates_removed']:,}")
    c[3].metric("Pass rate", f"{dq['pass_rate']:.2%}")

    injected = gen["history"]["dirty_records"]
    caught = {
        "null_card_id": dq["quarantine_by_reason"].get("null_card_id", 0),
        "non_positive_amount": dq["quarantine_by_reason"].get("non_positive_amount", 0),
        "invalid_coordinates": dq["quarantine_by_reason"].get("invalid_merchant_coordinates", 0),
        "null_ts": dq["quarantine_by_reason"].get("null_ts", 0),
        "duplicates": dq["duplicates_removed"],
    }
    labels = {
        "duplicates": "Duplicate (re-sent message)",
        "null_card_id": "Missing card id",
        "non_positive_amount": "Zero or negative amount",
        "invalid_coordinates": "Impossible coordinates",
        "null_ts": "Missing timestamp",
    }

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Injected problems vs. caught")
        check = pd.DataFrame(
            {
                "Problem": [labels[k] for k in labels],
                "Injected": [injected.get(k, 0) for k in labels],
                "Caught": [caught[k] for k in labels],
            }
        )
        all_caught = check["Injected"] == check["Caught"]
        check["Result"] = np.where(all_caught, "✅ all caught", "⚠️ mismatch")
        st.dataframe(check, hide_index=True, width="stretch")
        st.caption(
            "Every problem the generator injected was caught, with the exact count. "
            "Automated tests check this on every run."
        )

    with right:
        st.subheader("Quarantine by reason")
        reasons = pd.Series(dq["quarantine_by_reason"]).sort_values()
        fig = go.Figure(
            go.Bar(
                x=reasons.values,
                y=reasons.index,
                orientation="h",
                marker=dict(color=LEGIT, cornerradius=4),
                hovertemplate="%{y}: %{x} rows<extra></extra>",
            )
        )
        st.plotly_chart(style(fig, height=260, legend=False), width="stretch", config=PLOTLY_CONFIG)
        st.caption("Rows are quarantined with every rule they failed, not just the first.")

    st.subheader("Sample of quarantined rows")
    q = data.quarantine()
    st.dataframe(
        q[["transaction_id", "reject_reason", "card_id", "amount", "ts", "merchant_lat"]].head(50),
        hide_index=True,
        width="stretch",
    )


# --------------------------------------------------------------------------- #
# Drift scenarios
# --------------------------------------------------------------------------- #
with tab_drift:
    st.markdown(
        "After six months of history, the generator creates one more **production** month "
        "under four scenarios. Real-world behaviour changes over time (*drift*), and a model "
        "trained on the past can quietly get worse. Phase 4 will detect these changes "
        "automatically; for now, here is what each scenario changes."
    )
    hist = gen["history"]
    rows = []
    descriptions = {
        "none": "Nothing changes: the control group.",
        "new_fraud_pattern": "A new gift-card cash-out scam the model has never seen.",
        "spending_shift": "Holiday season: people spend ~40% more, ~20% more often.",
        "merchant_mix_shift": "Shoppers move online: the online share roughly doubles.",
    }
    for name, s in [("history", hist), *gen["production"].items()]:
        rows.append(
            {
                "Period": "History (training)" if name == "history" else f"Production: {name}",
                "What changes": "Baseline used for training"
                if name == "history"
                else descriptions[name],
                "Txns / day": s["clean_rows"] / s["days"],
                "Avg amount": s["mean_amount"],
                "Online share": s["online_share"],
                "Fraud rate": s["fraud_rate"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Txns / day": st.column_config.NumberColumn(format="%.0f"),
            "Avg amount": st.column_config.NumberColumn(format="$%.2f"),
            "Online share": st.column_config.NumberColumn(format="percent"),
            "Fraud rate": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Compare each production row with the history row: spending_shift raises amounts and "
        "volume, merchant_mix_shift raises the online share, and new_fraud_pattern raises the "
        "fraud rate with a pattern absent from training."
    )
