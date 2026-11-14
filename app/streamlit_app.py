"""SentinelPay dashboard entry point: `make app` (or `streamlit run app/streamlit_app.py`)."""

import streamlit as st

st.set_page_config(page_title="SentinelPay", page_icon="🛡️", layout="wide")

pages = [
    st.Page("views/data_explorer.py", title="Data Explorer", icon="🔎", default=True),
]

with st.sidebar:
    st.markdown("### 🛡️ SentinelPay")
    st.caption(
        "Card fraud detection platform. **Preview:** the data foundation (Phase 1). "
        "Model, live scoring and drift monitoring pages arrive in later phases."
    )

st.navigation(pages).run()
