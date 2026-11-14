"""Model performance: how well the champion model works, why, and versus what."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data
from lib.charts import FRAUD, LEGIT, PLOTLY_CONFIG, SERIES, style
from sklearn.metrics import precision_recall_curve, roc_curve

from sentinelpay.data.generator import FRAUD_PATTERNS

st.title("📈 Model Performance")

if not data.model_outputs_exist():
    st.warning("No trained model yet. Run `make demo`, then reload this page.")
    st.stop()

metrics = data.report("metrics.json")
preds = data.report_table("test_predictions.parquet")
alerts = data.report_table("test_alerts.parquet")
registry = data.report("registry.json")
promotion = data.report("promotion.json")

lgbm = metrics["models"]["lightgbm"]
lr = metrics["models"]["logistic_regression"]
rules = metrics["models"]["rules"]
split = metrics["split"]
y = preds["is_fraud"].to_numpy()

MODELS = {  # display name -> (prediction column, colour slot)
    "LightGBM (champion)": ("score_lightgbm", LEGIT),
    "Logistic regression (baseline)": ("score_logistic_regression", FRAUD),
}

st.markdown(
    f"Trained on **{split['train']['start']} → {split['train']['end']}**, tuned on "
    f"**{split['validation']['start']} → {split['validation']['end']}**, and evaluated "
    f"below on **{split['test']['start']} → {split['test']['end']}**: "
    f"{split['test']['rows']:,} transactions the model never saw. "
    "**Data is synthetic**, so absolute numbers are higher than on real fraud data; "
    "the comparisons between approaches are what matter."
)

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
k = st.columns(4)
k[0].metric(
    "PR-AUC",
    f"{lgbm['pr_auc']:.3f}",
    delta=f"{lgbm['pr_auc'] - lr['pr_auc']:+.3f} vs baseline",
    help="Area under the precision-recall curve: 1.0 is perfect; random scores ≈ fraud rate.",
)
k[1].metric(
    "Fraud caught at 0.5% false alarms",
    f"{lgbm['recall_at_fpr']['0.5%']:.1%}",
    help="Recall when at most 0.5% of genuine transactions may be flagged.",
)
k[2].metric(
    f"Saved vs. rules ({split['test']['days']} days)",
    f"${metrics['savings']['vs_rules']:,.0f}",
    help="Expected cost of the rules engine minus that of the model, at the same business costs.",
)
k[3].metric("Champion model", f"v{registry['champion']}")

# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
left, right = st.columns(2)


def _thin(*arrays: np.ndarray, n: int = 1500) -> list[np.ndarray]:
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n).astype(int))
    return [a[idx] for a in arrays]


with left:
    st.subheader("Precision vs. recall")
    fig = go.Figure()
    for name, (col, color) in MODELS.items():
        p, r, _ = precision_recall_curve(y, preds[col])
        r, p = _thin(r, p)
        fig.add_scatter(
            x=r,
            y=p,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=name + "<br>recall %{x:.1%}<br>precision %{y:.1%}<extra></extra>",
        )
    ra = rules["at_threshold"]
    fig.add_scatter(
        x=[ra["recall"]],
        y=[ra["precision"]],
        mode="markers",
        name="Rules engine",
        marker=dict(size=13, color=SERIES[3], symbol="square", line=dict(color="white", width=2)),
        hovertemplate="Rules engine<br>recall %{x:.1%}<br>precision %{y:.1%}<extra></extra>",
    )
    fig.update_xaxes(title="Recall (fraud caught)", tickformat=".0%", range=[0, 1.02])
    fig.update_yaxes(title="Precision (alerts that are fraud)", tickformat=".0%", range=[0, 1.05])
    st.plotly_chart(style(fig, height=360), width="stretch", config=PLOTLY_CONFIG)
    st.caption(
        "Higher and further right is better. The rules engine is a single point: it can't be "
        "tuned, and it sits far below both models."
    )

ROC_HOVER = "<br>false alarms %{x:.2%}<br>fraud caught %{y:.1%}<extra></extra>"

with right:
    st.subheader("Fraud caught vs. false-alarm rate")
    fig = go.Figure()
    for name, (col, color) in MODELS.items():
        fpr, tpr, _ = roc_curve(y, preds[col])
        fpr, tpr = _thin(fpr, tpr)
        fig.add_scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=name + ROC_HOVER,
        )
    fig.add_scatter(
        x=[ra["fpr"]],
        y=[ra["recall"]],
        mode="markers",
        name="Rules engine",
        marker=dict(size=13, color=SERIES[3], symbol="square", line=dict(color="white", width=2)),
        hovertemplate="Rules engine" + ROC_HOVER,
    )
    ticks = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    fig.update_xaxes(
        title="False-alarm rate (log scale)",
        type="log",
        tickvals=ticks,
        ticktext=[f"{t:.2%}".rstrip("0").rstrip(".") if t < 0.01 else f"{t:.0%}" for t in ticks],
        range=[np.log10(2e-4), 0],
    )
    fig.update_yaxes(title="Fraud caught (recall)", tickformat=".0%", range=[0, 1.02])
    st.plotly_chart(style(fig, height=360), width="stretch", config=PLOTLY_CONFIG)
    st.caption(
        "ROC curve with a log x-axis, so the region that matters (well under 1% false alarms) "
        "is visible. Banks operate at the far left."
    )

# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
st.subheader("Side by side, at each approach's chosen threshold")
rows = []
for label, m in (("LightGBM", lgbm), ("Logistic regression", lr), ("Rules engine", rules)):
    a = m["at_threshold"]
    rows.append(
        {
            "Approach": label,
            "PR-AUC": m.get("pr_auc"),
            "Caught @0.1% FPR": m.get("recall_at_fpr", {}).get("0.1%"),
            "Caught @0.5% FPR": m.get("recall_at_fpr", {}).get("0.5%"),
            "Caught @1% FPR": m.get("recall_at_fpr", {}).get("1.0%"),
            "Fraud caught": a["recall"],
            "Alerts that are fraud": a["precision"],
            "Alerts / day": a["alerts_per_day"],
            "Expected cost": a["expected_cost"],
        }
    )
rows.append(
    {
        "Approach": "No model",
        "Fraud caught": 0.0,
        "Alerts / day": 0.0,
        "Expected cost": metrics["no_model_cost"],
    }
)
FORMATS = {
    "PR-AUC": "{:.3f}",
    "Caught @0.1% FPR": "{:.1%}",
    "Caught @0.5% FPR": "{:.1%}",
    "Caught @1% FPR": "{:.1%}",
    "Fraud caught": "{:.1%}",
    "Alerts that are fraud": "{:.1%}",
    "Alerts / day": "{:,.1f}",
    "Expected cost": "${:,.0f}",
}
table = pd.DataFrame(rows)
for col, fmt in FORMATS.items():  # "—" where a metric doesn't apply (e.g. PR-AUC of rules)
    table[col] = table[col].map(lambda v, fmt=fmt: "—" if pd.isna(v) else fmt.format(v))
st.dataframe(table, hide_index=True, width="stretch")
st.caption(
    f"Expected cost = missed frauds × ${metrics['costs']['missed_fraud']:,.0f} + alerts × "
    f"${metrics['costs']['alert_review']:,.0f}, over the {split['test']['days']}-day test period. "
    "Each model's threshold was chosen on the validation month, never on test."
)

# --------------------------------------------------------------------------- #
# By pattern + importance
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

with left:
    st.subheader("Fraud caught, by attack type")
    names = {
        "card_testing": "Card testing",
        "impossible_travel": "Impossible travel",
        "high_amount": "High amount",
        "night_burst": "Night burst",
        "compromised_merchant": "Compromised merchant",
    }
    fig = go.Figure()
    for (label, m), color in zip(
        (("LightGBM", lgbm), ("Logistic regression", lr), ("Rules engine", rules)),
        (LEGIT, FRAUD, SERIES[3]),
        strict=True,
    ):
        rec = m["recall_by_pattern"]
        fig.add_bar(
            x=[names[p] for p in FRAUD_PATTERNS],
            y=[100 * rec.get(p, 0) for p in FRAUD_PATTERNS],
            name=label,
            marker=dict(color=color, cornerradius=4),
            hovertemplate=label + "<br>%{x}: %{y:.0f}% caught<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="Fraud caught", ticksuffix="%", range=[0, 105])
    st.plotly_chart(style(fig, height=360), width="stretch", config=PLOTLY_CONFIG)
    st.caption(
        "Rules catch the attack they were written for and little else. The model generalises "
        "across patterns; high-amount fraud is hardest because genuine big purchases look similar."
    )

with right:
    st.subheader("What drives the model")
    imp = pd.Series(metrics["shap_importance"]).head(12)[::-1]
    fig = go.Figure(
        go.Bar(
            x=imp.values,
            y=imp.index,
            orientation="h",
            marker=dict(color=LEGIT, cornerradius=4),
            hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Average impact on the score (mean |SHAP|, log-odds)")
    st.plotly_chart(style(fig, height=360, legend=False), width="stretch", config=PLOTLY_CONFIG)
    st.caption(
        "SHAP measures how much each feature moves an individual prediction; this is the "
        "average over test transactions. Definitions are on the Data Explorer's Features tab."
    )

# --------------------------------------------------------------------------- #
# Top alerts with reasons
# --------------------------------------------------------------------------- #
st.subheader("Highest-risk test transactions, and why")
top = alerts.head(25).copy()
reasons = top["reasons"].map(json.loads)
REASON_COLS = ["Reason 1", "Reason 2", "Reason 3"]
for i in range(3):
    top[f"Reason {i + 1}"] = reasons.map(lambda r, i=i: r[i]["text"] if len(r) > i else "")
top["Truth"] = np.where(top["is_fraud"] == 1, "🚩 " + top["fraud_pattern"], "genuine")
st.dataframe(
    top[["ts", "amount", "merchant_category", "score", "Truth", *REASON_COLS]],
    hide_index=True,
    width="stretch",
    column_config={
        "ts": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM, HH:mm"),
        "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
        "merchant_category": "Category",
        "score": st.column_config.ProgressColumn(
            "Fraud probability", format="%.2f", min_value=0.0, max_value=1.0
        ),
    },
)
st.caption(
    "Reason codes are the three features that pushed each score up the most (per-transaction "
    "SHAP values), translated into plain English for the analyst reviewing the alert."
)

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
st.subheader("Model registry (MLflow)")
icon = "✅" if promotion["promoted"] else "ℹ️"
st.info(
    f"{icon} **Latest promotion decision:** {promotion['reason']}. "
    f"Champion is **v{promotion['champion_version']}**; challenger was "
    f"**v{promotion['challenger_version']}**. Both were scored on the same test window "
    f"({promotion['test_window'][0]} → {promotion['test_window'][1]})."
)
reg = pd.DataFrame(registry["versions"])
reg["aliases"] = reg["aliases"].map(lambda a: ", ".join(a) if a else "")
st.dataframe(
    reg[["version", "aliases", "created", "test_pr_auc", "test_expected_cost", "threshold"]],
    hide_index=True,
    width="stretch",
    column_config={
        "version": "Version",
        "aliases": "Alias",
        "created": "Registered (UTC)",
        "test_pr_auc": st.column_config.NumberColumn("Test PR-AUC", format="%.3f"),
        "test_expected_cost": st.column_config.NumberColumn("Test cost", format="$%,.0f"),
        "threshold": st.column_config.NumberColumn("Threshold", format="%.4f"),
    },
)
st.caption(
    "A new model is registered as the *challenger* and only becomes *champion* if it beats the "
    "current champion on both PR-AUC and expected cost. Run `make mlflow` to browse every run, "
    "parameter and plot in the MLflow UI."
)
