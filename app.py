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
from risk_engine import calculate_risk, total_enterprise_risk, top_risky_assets, format_inr, format_inr_short
from ml_layer import train_likelihood_model
from optimizer import build_remediation_options, optimize_budget, risk_reduction_curve
from data_ingestion import ingest_user_data

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


# --- Data Source Selection (Sidebar) ---
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Demo Data (Synthetic)", "Upload Your Own Data"],
    index=0,
)

if data_source == "Upload Your Own Data":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your asset/vulnerability data",
        type=["csv", "xlsx", "xls", "json", "txt"],
        help="Upload a CSV, Excel, JSON, or text file containing your asset inventory, "
             "vulnerability scan results, or any cyber risk data. The platform will "
             "automatically map your columns and fill in any missing fields."
    )
    if uploaded_file is not None:
        df, messages = ingest_user_data(uploaded_file)

        # Show mapping/warning messages to user
        for msg in messages:
            if msg.startswith("❌"):
                st.sidebar.error(msg)
            elif msg.startswith("⚠️"):
                st.sidebar.warning(msg)
            elif msg.startswith("✅"):
                st.sidebar.success(msg)
            else:
                st.sidebar.info(msg)

        if len(df) == 0:
            st.error("Could not process the uploaded file. Please check the messages in the sidebar.")
            st.stop()

        # Show a preview of the processed data
        with st.sidebar.expander("Preview processed data"):
            st.dataframe(df.head(10))

        # Run the full pipeline on user data
        df = calculate_risk(df)
        df = build_remediation_options(df)
    else:
        st.sidebar.warning("Please upload a file to continue, or switch to Demo Data.")
        st.stop()
else:
    df = load_data()

total_risk = total_enterprise_risk(df)

st.title("AI-Powered Cyber Risk Quantification Platform")
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
    f"**Expected Annual Loss:** {format_inr_short(total_risk)}  \n"
    "*Clear, actionable, business-ready*"
)

# --- Ask the Platform (Chat UI) ---

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



def get_answer(question):
    """Return an answer string based on keyword matching."""
    q = question.lower()

    # Handle follow-up questions ("why", "explain", "how do you know")
    if any(kw in q for kw in ("why", "explain", "how do you know", "how is that")):
        # Check if the previous assistant message mentioned a specific asset
        prev_msgs = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
        if prev_msgs:
            last = prev_msgs[-1]["content"]
            match = df[df["asset_name"].apply(lambda n: n in last)]
            if len(match):
                asset = match.iloc[0]
                return (
                    f"**{asset['asset_name']}** has an Expected Annual Loss of "
                    f"**{format_inr_short(asset['expected_annual_loss_inr'])}**, calculated as:\n\n"
                    f"> **Likelihood of Attack** ({asset['likelihood_pct']:.0f}%) "
                    f"x **Financial Impact** ({format_inr_short(asset['impact_inr'])}) "
                    f"x **Asset Criticality Weight** ({asset['criticality_weight']:.2f})\n\n"
                    f"A higher value in any of these three factors raises the "
                    f"Expected Annual Loss proportionally."
                )

    if "highest risk" in q or "biggest risk" in q:
        top = df.iloc[0]
        return (f"Your highest financial risk is **{top['asset_name']}**, "
                f"with an Expected Annual Loss of "
                f"**{format_inr_short(top['expected_annual_loss_inr'])}**.")
    elif "total risk" in q or "total exposure" in q:
        return (f"Your total enterprise cyber risk exposure is "
                f"**{format_inr_short(total_risk)}** per year.")
    elif "budget" in q or "invest" in q:
        return ("Check the **Investment Optimization** section below to see "
                "budget recommendations and risk reduction analysis.")
    else:
        return ("This prototype currently answers questions about highest risk, "
                "total exposure, and budget recommendations. (A full version would "
                "use an LLM to answer any natural-language question over this data.)")


with st.container(border=True):
    st.subheader("Ask the Platform")

    # Reduce gap between buttons and chat history
    st.markdown(
        "<style>div[data-testid='stChatInput'] {margin-top: 0;} "
        ".stColumns + div[data-testid='stChatMessage'] {margin-top: -0.5rem;}</style>",
        unsafe_allow_html=True,
    )

    # Example question buttons
    eq1, eq2, eq3 = st.columns(3, gap="small")
    if eq1.button("What's our highest risk?", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": "What's our highest risk?"})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer("What's our highest risk?")})
    if eq2.button("What's our total exposure?", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": "What's our total exposure?"})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer("What's our total exposure?")})
    if eq3.button("Budget recommendation?", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": "Budget recommendation?"})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer("Budget recommendation?")})

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(placeholder="Ask anything about your cyber risk..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer(prompt)})
        st.rerun()

st.divider()

# --- Top summary ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Enterprise Risk (Expected Annual Loss)", format_inr_short(total_risk))
col2.metric("Assets Monitored", len(df))
col3.metric("High-Criticality Assets", int((df["criticality_weight"] > 0.7).sum()))

st.divider()

# --- Top risky assets ---
st.subheader("Top 5 Riskiest Assets")
st.dataframe(top_risky_assets(df), use_container_width=True, hide_index=True)

st.divider()

# --- Investment Optimization ---
st.subheader("Investment Optimization")

slider_col, custom_col = st.columns([3, 1])
with slider_col:
    slider_budget = st.select_slider(
        "Security Budget",
        options=list(range(0, 2_00_00_001, 5_00_000)),
        value=1_00_00_000,
        format_func=format_inr_short,
    )
with custom_col:
    custom_budget = st.number_input(
        "Add Custom Budget (₹)",
        min_value=0,
        max_value=50_00_00_000,
        value=0,
        step=1_00_000,
        help="Enter an exact budget amount. When set above 0, this overrides the slider.",
    )

budget = custom_budget if custom_budget > 0 else slider_budget

selected = optimize_budget(df, budget)
reduced = selected["risk_reduction_inr"].sum() if len(selected) else 0

c1, c2 = st.columns(2)
c1.metric("Recommended Assets to Remediate", len(selected))
c2.metric("Total Risk Reduced", format_inr_short(reduced))

if len(selected):
    st.write("Recommended remediation plan for this budget:")
    display_df = selected[["asset_name", "asset_type", "remediation_cost_inr", "risk_reduction_inr"]].copy()
    display_df["remediation_cost_inr"] = display_df["remediation_cost_inr"].map(format_inr_short)
    display_df["risk_reduction_inr"] = display_df["risk_reduction_inr"].map(format_inr_short)
    st.dataframe(
        display_df.rename(columns={
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
st.subheader("AI Risk Drivers (ML Layer)")
model, importance, mae = train_likelihood_model(df)
st.write("Which factors contribute most to attack likelihood, according to the model:")
st.bar_chart(importance.head(8))


st.divider()
st.caption("Prototype built for SIH26105 — Team demo. Data shown is synthetic, generated to "
           "illustrate the platform's logic, not real company data.")
