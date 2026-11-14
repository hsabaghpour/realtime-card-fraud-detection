"""SentinelPay dashboard entry point: `make app` (or `streamlit run app/streamlit_app.py`)."""

import streamlit as st

st.set_page_config(page_title="SentinelPay", page_icon="🛡️", layout="wide")

pages = {
    "Data": [
        st.Page("views/data_explorer.py", title="Data Explorer", icon="🔎", default=True),
    ],
    "Model": [
        st.Page("views/model_performance.py", title="Model Performance", icon="📈"),
        st.Page("views/threshold.py", title="Threshold & Trade-off", icon="⚖️"),
    ],
}

with st.sidebar:
    st.markdown("### 🛡️ SentinelPay")
    st.caption(
        "Card fraud detection platform. **Preview:** data, features and model "
        "(Phases 1–3). Live scoring and drift monitoring pages arrive in later phases."
    )

st.navigation(pages).run()
