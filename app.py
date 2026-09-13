"""
app.py
------
Streamlit dashboard for the SIH26105 Cyber Risk Quantification prototype.

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st

from generate_data import generate_dataset
from risk_engine import calculate_risk, total_enterprise_risk, top_risky_assets, format_inr
from ml_layer import train_likelihood_model
from optimizer import build_remediation_options, optimize_budget, risk_reduction_curve

st.set_page_config(page_title="Cyber Risk Quantification Platform", layout="wide")

DATA_PATH = "data/assets.csv"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        df = generate_dataset()
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    df = calculate_risk(df)
    df = build_remediation_options(df)
    return df


df = load_data()
total_risk = total_enterprise_risk(df)

st.title("🛡️ AI-Powered Cyber Risk Quantification Platform")
st.caption("SIH26105 — Prototype | Converting technical cyber risk into financial exposure (₹)")

# --- Traditional vs Our Approach ---
comp_col1, comp_col2 = st.columns(2)
comp_col1.error(
    "**Traditional Approach**  \n"
    "**Risk Level:** Medium  \n"
    "*No financial context, hard to act on*"
)
comp_col2.success(
    f"**Our Approach**  \n"
    f"**Expected Annual Loss:** {format_inr(total_risk)}  \n"
    "*Clear, actionable, business-ready*"
)

# --- Top summary ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Enterprise Risk (Expected Annual Loss)", format_inr(total_risk))
col2.metric("Assets Monitored", len(df))
col3.metric("High-Criticality Assets", int((df["criticality_weight"] > 0.7).sum()))

st.divider()

# --- Top risky assets ---
st.subheader("🔥 Top 5 Riskiest Assets")
st.dataframe(top_risky_assets(df), use_container_width=True, hide_index=True)

st.divider()

# --- Investment Optimization ---
st.subheader("💰 Investment Optimization")
budget = st.slider(
    "Security Budget (₹)", min_value=0, max_value=2_00_00_000, value=1_00_00_000, step=5_00_000,
    format="₹%d",
)

selected = optimize_budget(df, budget)
reduced = selected["risk_reduction_inr"].sum() if len(selected) else 0

c1, c2 = st.columns(2)
c1.metric("Recommended Assets to Remediate", len(selected))
c2.metric("Total Risk Reduced", format_inr(reduced))

if len(selected):
    st.write("Recommended remediation plan for this budget:")
    st.dataframe(
        selected[["asset_name", "asset_type", "remediation_cost_inr", "risk_reduction_inr"]]
        .rename(columns={
            "asset_name": "Asset",
            "asset_type": "Type",
            "remediation_cost_inr": "Cost (₹)",
            "risk_reduction_inr": "Risk Reduced (₹)",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.write("**Investment vs. Risk Reduction curve** (shows diminishing returns as budget grows)")
curve = risk_reduction_curve(df, max_budget=2_00_00_000)
st.line_chart(curve.set_index("budget_inr"))

st.divider()

# --- AI Decision Support / Risk Drivers ---
st.subheader("🧠 AI Risk Drivers (ML Layer)")
model, importance, mae = train_likelihood_model(df)
st.write("Which factors contribute most to attack likelihood, according to the model:")
st.bar_chart(importance.head(8))

st.divider()

# --- Simple NL query ---
st.subheader("💬 Ask the Platform")
query = st.text_input("Ask a question, e.g. 'What is our highest risk today?'")

if query:
    q = query.lower()
    if "highest risk" in q or "biggest risk" in q:
        top = df.iloc[0]
        st.info(f"Your highest financial risk is **{top['asset_name']}**, "
                f"with an Expected Annual Loss of **{format_inr(top['expected_annual_loss_inr'])}**.")
    elif "total risk" in q or "total exposure" in q:
        st.info(f"Your total enterprise cyber risk exposure is **{format_inr(total_risk)}** per year.")
    elif "budget" in q or "invest" in q:
        st.info(f"With a budget of {format_inr(budget)}, remediating the top "
                f"{len(selected)} assets reduces risk by **{format_inr(reduced)}**.")
    else:
        st.info("This prototype currently answers questions about highest risk, "
                "total exposure, and budget recommendations. (A full version would "
                "use an LLM to answer any natural-language question over this data.)")

st.divider()
st.caption("Prototype built for SIH26105 — Team demo. Data shown is synthetic, generated to "
           "illustrate the platform's logic, not real company data.")
