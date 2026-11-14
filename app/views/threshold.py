"""Threshold & business trade-off: what happens when you move the alert cut-off."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from lib import data
from lib.charts import FRAUD, INK_2, LEGIT, PLOTLY_CONFIG, SERIES, style
from sklearn.metrics import precision_recall_curve

from sentinelpay.training import evaluate as ev

st.title("⚖️ Threshold & Business Trade-off")
st.markdown(
    "The model gives every transaction a **fraud probability**. The bank still has to decide "
    "*how high is high enough* to raise an alert. Set it **low** and analysts drown in false "
    "alarms. Set it **high** and fraud slips through. The best cut-off depends on what each "
    "mistake costs, so it is a business decision, not a fixed 50%."
)

if not data.model_outputs_exist():
    st.warning("No trained model yet. Run `make demo`, then reload this page.")
    st.stop()

metrics = data.report("metrics.json")
preds = data.report_table("test_predictions.parquet")
y = preds["is_fraud"].to_numpy()
s = preds["score_lightgbm"].to_numpy()
n_days = metrics["split"]["test"]["days"]
period = f"{metrics['split']['test']['start']} → {metrics['split']['test']['end']}"

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)
cost_missed = c1.number_input(
    "Cost of one missed fraud ($)",
    min_value=0.0,
    value=float(metrics["costs"]["missed_fraud"]),
    step=25.0,
    help="Average loss the bank absorbs (chargeback, fees, customer churn).",
)
cost_review = c2.number_input(
    "Cost of reviewing one alert ($)",
    min_value=0.0,
    value=float(metrics["costs"]["alert_review"]),
    step=1.0,
    help="Analyst time, plus friction for a genuine customer whose card is checked.",
)

curve = ev.cost_curve(y, s, cost_missed, cost_review)
best_t, best_cost = curve.best

# Thresholds on a log grid: the interesting region is far below 0.5.
grid = np.unique(np.round(np.geomspace(0.001, 0.99, 160), 4))
grid = np.unique(np.r_[grid, round(metrics["models"]["lightgbm"]["threshold"], 4)])


def _nearest(value: float) -> float:
    return float(grid[np.argmin(np.abs(np.log(grid) - np.log(max(value, 1e-6))))])


if "threshold" not in st.session_state:
    st.session_state["threshold"] = _nearest(metrics["models"]["lightgbm"]["threshold"])

b1, b2, _ = st.columns([1, 1, 3])
if b1.button("Jump to cost-optimal", type="primary"):
    st.session_state["threshold"] = _nearest(best_t if np.isfinite(best_t) else 0.99)
if b2.button("Reset to trained threshold"):
    st.session_state["threshold"] = _nearest(metrics["models"]["lightgbm"]["threshold"])

threshold = st.select_slider(
    "Alert threshold: raise an alert when fraud probability ≥",
    options=[float(t) for t in grid],
    key="threshold",
    format_func=lambda t: f"{t:.2%}" if t < 0.1 else f"{t:.0%}",
)

alert = s >= threshold
op = ev.operating_point(y, alert, cost_missed, cost_review, n_days)

# --------------------------------------------------------------------------- #
# Headline numbers
# --------------------------------------------------------------------------- #
k = st.columns(4)
k[0].metric("Fraud caught", f"{op['recall']:.1%}", help="Recall: share of fraud that is alerted.")
k[1].metric(
    "Alerts that are real fraud",
    f"{op['precision']:.1%}",
    help="Precision: of all alerts, how many are fraud.",
)
k[2].metric("Alerts per day", f"{op['alerts_per_day']:,.1f}")
k[3].metric(
    f"Expected cost ({n_days} days)",
    f"${op['expected_cost']:,.0f}",
    delta=f"${op['expected_cost'] - best_cost:,.0f} vs optimal",
    delta_color="inverse",
)
st.caption(
    f"Test period {period} (data the model never saw during training). The cost-optimal "
    f"threshold for these costs is **{best_t:.2%}**, costing **${best_cost:,.0f}**. "
    f"With no model, every fraud is missed: **${y.sum() * cost_missed:,.0f}**."
)

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

with left:
    st.subheader("Total cost by threshold")
    keep = np.isfinite(curve.thresholds) & (curve.thresholds >= 1e-4)
    t_line, c_line = curve.thresholds[keep], curve.costs[keep]
    idx = np.unique(np.linspace(0, len(t_line) - 1, 1500).astype(int))
    fig = go.Figure(
        go.Scatter(
            x=t_line[idx],
            y=c_line[idx],
            mode="lines",
            line=dict(color=LEGIT, width=2),
            name="Expected cost",
            hovertemplate="threshold %{x:.2%}<br>cost $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold,
        line=dict(color=FRAUD, width=2),
        annotation_text="you",
        annotation_position="top",
    )
    if np.isfinite(best_t):
        fig.add_vline(
            x=best_t,
            line=dict(color=INK_2, width=1.5, dash="dash"),
            annotation_text="optimal",
            annotation_position="bottom right",
        )
    fig.update_xaxes(type="log", title="Alert threshold (log scale)", tickformat=".1%")
    fig.update_yaxes(title=f"Expected cost over {n_days} days", tickprefix="$")
    fig = style(fig, legend=False)
    fig.update_layout(margin=dict(t=30))  # room for the "you" label above the plot
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    st.caption(
        "Left of the dip, review costs explode (too many alerts). Right of it, missed fraud "
        "dominates. The dashed line marks the cheapest threshold for the costs above."
    )

with right:
    st.subheader("Precision vs. recall")
    p, r, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(p) - 1, 1500).astype(int))
    fig = go.Figure()
    fig.add_scatter(
        x=r[idx],
        y=p[idx],
        mode="lines",
        line=dict(color=LEGIT, width=2),
        name="LightGBM",
        hovertemplate="recall %{x:.1%}<br>precision %{y:.1%}<extra></extra>",
    )
    fig.add_scatter(
        x=[op["recall"]],
        y=[op["precision"]],
        mode="markers",
        name="Your threshold",
        marker=dict(size=14, color=FRAUD, symbol="diamond", line=dict(color="white", width=2)),
        hovertemplate="your threshold<br>recall %{x:.1%}<br>precision %{y:.1%}<extra></extra>",
    )
    best_op = ev.operating_point(y, s >= best_t, cost_missed, cost_review, n_days)
    fig.add_scatter(
        x=[best_op["recall"]],
        y=[best_op["precision"]],
        mode="markers",
        name="Cost-optimal",
        marker=dict(size=12, color=SERIES[2], symbol="circle", line=dict(color="white", width=2)),
        hovertemplate="cost-optimal<br>recall %{x:.1%}<br>precision %{y:.1%}<extra></extra>",
    )
    fig.update_xaxes(title="Recall (fraud caught)", tickformat=".0%", range=[0, 1.02])
    fig.update_yaxes(title="Precision (alerts that are fraud)", tickformat=".0%", range=[0, 1.05])
    st.plotly_chart(style(fig), width="stretch", config=PLOTLY_CONFIG)
    st.caption(
        "Every threshold is a point on this curve. Catching more fraud (moving right) always "
        "means more false alarms (moving down). The question is where to stop."
    )

# --------------------------------------------------------------------------- #
# Confusion matrix in plain words
# --------------------------------------------------------------------------- #
st.subheader("What this threshold means in practice")
cm = st.columns(4)
cm[0].metric("✅ Fraud caught", f"{op['tp']:,}")
cm[1].metric("❌ Fraud missed", f"{op['fn']:,}")
cm[2].metric("⚠️ False alarms", f"{op['fp']:,}")
cm[3].metric("👍 Genuine, not bothered", f"{op['tn']:,}")
st.caption(
    f"Out of {len(y):,} test transactions ({int(y.sum()):,} fraudulent) over {n_days} days. "
    "Each false alarm is a genuine customer whose payment gets checked."
)
