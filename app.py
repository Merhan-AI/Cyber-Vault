"""
app.py
------
Streamlit dashboard for the SIH26105 Cyber Risk Quantification prototype.

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

from generate_data import generate_dataset
from risk_engine import calculate_risk, total_enterprise_risk, top_risky_assets, format_inr, format_inr_short
from ml_layer import train_likelihood_model
from optimizer import build_remediation_options, optimize_budget, risk_reduction_curve, explain_budget_allocation
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


# =====================================================================
# SIDEBAR — Data Source + AI Config
# =====================================================================
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

        with st.sidebar.expander("Preview processed data"):
            st.dataframe(df.head(10))

        df = calculate_risk(df)
        df = build_remediation_options(df)
    else:
        st.sidebar.warning("Please upload a file to continue, or switch to Demo Data.")
        st.stop()
else:
    df = load_data()

# --- AI Chatbot config in sidebar ---
st.sidebar.divider()
st.sidebar.header("🤖 AI Chatbot Settings")
gemini_api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Enter your Google Gemini API key for AI-powered Q&A. Get one free at https://aistudio.google.com/apikey",
)

total_risk = total_enterprise_risk(df)

# =====================================================================
# HEADER
# =====================================================================
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


# =====================================================================
# AI CHATBOT with Gemini Integration
# =====================================================================
def build_data_context(df, budget, selected, reduced, total_risk, model_importance, model_mae):
    """Build a rich context string from the live data for the AI model."""
    top5 = df.nlargest(5, "expected_annual_loss_inr")
    top5_text = "\n".join([
        f"  - {r['asset_name']} ({r['asset_type']}): EAL={format_inr_short(r['expected_annual_loss_inr'])}, "
        f"Likelihood={r['likelihood_pct']:.1f}%, Impact={format_inr_short(r['potential_financial_impact_inr'])}, "
        f"Criticality={r['criticality_weight']:.2f}, Vulns={r['vulnerability_count']}"
        for _, r in top5.iterrows()
    ])

    bottom5 = df.nsmallest(5, "expected_annual_loss_inr")
    bottom5_text = "\n".join([
        f"  - {r['asset_name']}: EAL={format_inr_short(r['expected_annual_loss_inr'])}"
        for _, r in bottom5.iterrows()
    ])

    type_risk = df.groupby("asset_type")["expected_annual_loss_inr"].sum().sort_values(ascending=False)
    type_risk_text = "\n".join([f"  - {t}: {format_inr_short(v)}" for t, v in type_risk.items()])

    importance_text = "\n".join([f"  - {feat}: {imp:.4f}" for feat, imp in model_importance.head(5).items()])

    sel_text = ""
    if len(selected) > 0:
        sel_rows = selected.head(5)
        sel_text = "\n".join([
            f"  - {r['asset_name']}: Cost={format_inr_short(r['remediation_cost_inr'])}, "
            f"Risk Reduced={format_inr_short(r['risk_reduction_inr'])}"
            for _, r in sel_rows.iterrows()
        ])

    context = f"""You are a cybersecurity risk analyst AI embedded in the Cyber Risk Quantification Platform.
You answer questions about the organization's cyber risk posture using ONLY the data provided below.
Be concise, specific, and always reference numbers from the data. Use ₹ values when discussing finances.
If the user asks about something not in the data, say so honestly.

=== LIVE DATA SUMMARY ===
Total Assets Monitored: {len(df)}
Total Enterprise Risk (Expected Annual Loss): {format_inr_short(total_risk)}
High-Criticality Assets (weight > 0.7): {int((df['criticality_weight'] > 0.7).sum())}
Average Likelihood of Attack: {df['likelihood_pct'].mean():.1f}%
Average Criticality Weight: {df['criticality_weight'].mean():.2f}

Risk Formula: EAL = Likelihood × Financial Impact × Criticality Weight
(Based on FAIR — Factor Analysis of Information Risk framework)

Top 5 Riskiest Assets:
{top5_text}

Lowest 5 Risk Assets:
{bottom5_text}

Risk by Asset Type:
{type_risk_text}

ML Model (RandomForest) — Top Risk Drivers (feature importance):
{importance_text}
ML Model MAE: {model_mae:.4f}

Current Budget: {format_inr_short(budget)}
Assets Selected for Remediation: {len(selected)}
Total Risk Reduced by Remediation: {format_inr_short(reduced)}
Top Remediation Recommendations:
{sel_text if sel_text else "  (None selected at current budget)"}

=== ALL ASSETS DATA ===
{df[['asset_name', 'asset_type', 'vulnerability_count', 'likelihood_pct', 'potential_financial_impact_inr', 'criticality_weight', 'expected_annual_loss_inr']].to_string(index=False)}
"""
    return context


def get_gemini_response(question, context, api_key):
    """Call Google Gemini API with the data context."""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{context}\n\n=== USER QUESTION ===\n{question}",
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI model error: {str(e)}\n\nFalling back to basic mode."


def get_answer(question, df, budget, selected, reduced, total_risk, model_importance, model_mae, api_key=None):
    """Return an answer: use Gemini if API key available, otherwise keyword fallback."""

    # Try Gemini first
    if api_key:
        context = build_data_context(df, budget, selected, reduced, total_risk, model_importance, model_mae)
        response = get_gemini_response(question, context, api_key)
        if not response.startswith("⚠️ AI model error"):
            return response
        # Fall through to keyword matching if Gemini fails

    # Keyword fallback
    q = question.lower()

    if any(kw in q for kw in ("why", "explain", "how do you know", "how is that")):
        prev_msgs = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
        if prev_msgs:
            last = prev_msgs[-1]["content"]
            match = df[df["asset_name"].apply(lambda n: n in last)]
            if len(match):
                asset = match.iloc[0]
                return (
                    f"**{asset['asset_name']}** has an Expected Annual Loss of "
                    f"**{format_inr_short(asset['expected_annual_loss_inr'])}**, calculated as:\n\n"
                    f"> **Likelihood** ({asset['likelihood_pct']:.0f}%) "
                    f"× **Financial Impact** ({format_inr_short(asset['potential_financial_impact_inr'])}) "
                    f"× **Criticality Weight** ({asset['criticality_weight']:.2f})\n\n"
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
        if api_key:
            return "I couldn't generate a response. Please try rephrasing your question."
        return ("💡 **Tip:** Enter a Google Gemini API key in the sidebar for AI-powered answers "
                "to any question about your data.\n\n"
                "Without an API key, I can answer: *'What's our highest risk?'*, "
                "*'What's our total exposure?'*, *'Budget recommendation?'*")


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# We need budget/selected/model data for the chatbot, so compute them early
# (also used later in the dashboard)
# Default budget for initial chatbot context
_default_budget = 1_00_00_000

# Train ML model (cached)
model, importance, mae = train_likelihood_model(df)

with st.container(border=True):
    st.subheader("💬 Ask the Platform")

    if gemini_api_key:
        st.caption("🟢 AI-powered mode (Gemini)")
    else:
        st.caption("🔵 Basic mode — add a Gemini API key in the sidebar for AI answers")

    st.markdown(
        "<style>div[data-testid='stChatInput'] {margin-top: 0;} "
        ".stColumns + div[data-testid='stChatMessage'] {margin-top: -0.5rem;}</style>",
        unsafe_allow_html=True,
    )

    eq1, eq2, eq3 = st.columns(3, gap="small")

    # Pre-compute default optimization for chatbot context
    _sel_default = optimize_budget(df, _default_budget)
    _red_default = _sel_default["risk_reduction_inr"].sum() if len(_sel_default) else 0

    if eq1.button("What's our highest risk?", use_container_width=True):
        q = "What's our highest risk?"
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
            q, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae, gemini_api_key
        )})
    if eq2.button("What's our total exposure?", use_container_width=True):
        q = "What's our total exposure?"
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
            q, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae, gemini_api_key
        )})
    if eq3.button("Budget recommendation?", use_container_width=True):
        q = "Budget recommendation?"
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
            q, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae, gemini_api_key
        )})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(placeholder="Ask anything about your cyber risk..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
            prompt, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae, gemini_api_key
        )})
        st.rerun()

st.divider()

# =====================================================================
# TOP SUMMARY METRICS
# =====================================================================
col1, col2, col3 = st.columns(3)
col1.metric("Total Enterprise Risk (Expected Annual Loss)", format_inr_short(total_risk))
col2.metric("Assets Monitored", len(df))
col3.metric("High-Criticality Assets", int((df["criticality_weight"] > 0.7).sum()))

st.divider()

# =====================================================================
# TOP RISKY ASSETS
# =====================================================================
st.subheader("🔥 Top 5 Riskiest Assets")
st.dataframe(top_risky_assets(df), use_container_width=True, hide_index=True)

st.divider()

# =====================================================================
# CHARTS — Risk Analytics
# =====================================================================
st.subheader("📊 Risk Analytics")

chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
    "Risk Distribution", "By Asset Type", "Vulnerability vs Likelihood", "Risk Composition"
])

with chart_tab1:
    # Risk Distribution — horizontal bar chart of all assets by EAL
    st.write("**Expected Annual Loss by Asset** (sorted highest → lowest)")
    chart_df = df[["asset_name", "expected_annual_loss_inr"]].copy()
    chart_df = chart_df.sort_values("expected_annual_loss_inr", ascending=True).tail(20)
    st.bar_chart(chart_df.set_index("asset_name"), horizontal=True)

with chart_tab2:
    # Risk by Asset Type — grouped bar
    st.write("**Total Risk Exposure by Asset Type**")
    type_agg = df.groupby("asset_type").agg(
        total_eal=("expected_annual_loss_inr", "sum"),
        count=("asset_id", "count"),
        avg_likelihood=("likelihood_pct", "mean"),
    ).sort_values("total_eal", ascending=False).reset_index()

    col_a, col_b = st.columns(2)
    with col_a:
        st.bar_chart(type_agg.set_index("asset_type")["total_eal"])
    with col_b:
        st.write("**Breakdown Table**")
        display_type = type_agg.copy()
        display_type["total_eal"] = display_type["total_eal"].map(format_inr_short)
        display_type["avg_likelihood"] = display_type["avg_likelihood"].map(lambda x: f"{x:.1f}%")
        display_type.columns = ["Asset Type", "Total EAL", "Count", "Avg Likelihood"]
        st.dataframe(display_type, use_container_width=True, hide_index=True)

with chart_tab3:
    # Vulnerability Count vs Likelihood scatter
    st.write("**Vulnerability Count vs. Attack Likelihood** (bubble size = EAL)")
    scatter_df = df[["asset_name", "vulnerability_count", "likelihood_pct",
                     "expected_annual_loss_inr", "criticality_weight"]].copy()
    st.scatter_chart(
        scatter_df,
        x="vulnerability_count",
        y="likelihood_pct",
        size="expected_annual_loss_inr",
        color="criticality_weight",
    )

with chart_tab4:
    # Risk Composition — stacked breakdown showing contribution of each factor
    st.write("**Risk Factor Contribution** (top 10 assets)")
    top10 = df.nlargest(10, "expected_annual_loss_inr").copy()
    top10["Likelihood Score"] = top10["likelihood"] * top10["potential_financial_impact_inr"]
    top10["Criticality Multiplier"] = (
        top10["expected_annual_loss_inr"] - top10["Likelihood Score"]
    ).clip(lower=0)
    composition = top10[["asset_name", "Likelihood Score", "Criticality Multiplier"]].set_index("asset_name")
    st.bar_chart(composition)

st.divider()

# =====================================================================
# INVESTMENT OPTIMIZATION
# =====================================================================
st.subheader("💰 Investment Optimization")

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

# =====================================================================
# AI RISK DRIVERS (ML Layer)
# =====================================================================
st.subheader("🧠 AI Risk Drivers (ML Layer)")
st.write("Which factors contribute most to attack likelihood, according to the model:")
st.bar_chart(importance.head(8))

st.divider()

# =====================================================================
# XAI — EXPLAINABILITY & TRANSPARENCY LOGS
# =====================================================================
st.subheader("🔍 XAI — Explainability & Transparency Logs")
st.write(
    "This section provides full transparency into **how every number on this dashboard was calculated**. "
    "Use this to explain the platform's logic to judges, auditors, or leadership."
)

xai_tab1, xai_tab2, xai_tab3, xai_tab4 = st.tabs([
    "📐 Risk Formula", "📋 Per-Asset Breakdown", "💰 Optimizer Rationale", "🧠 ML Model Card"
])

with xai_tab1:
    st.markdown("""
### Risk Quantification Formula (FAIR-Aligned)

```
Expected Annual Loss (EAL) = Likelihood × Financial Impact × Criticality Weight
```

| Variable | Description | Range | Source |
|---|---|---|---|
| **Likelihood** | Annual probability of a breach/attack on this asset | 0.01 – 0.95 | Vulnerability scans, threat intelligence, SIEM telemetry |
| **Financial Impact** (₹) | Direct cost if this asset is breached (forensics, fines, downtime) | ₹5L – ₹1.5Cr | Business Impact Assessment (BIA) |
| **Criticality Weight** | How important this asset is to business operations | 0.1 – 1.0 | Asset owner classification |

**Why multiply?** This follows the actuarial expected-value model used in insurance and the
[FAIR framework](https://www.fairinstitute.org/). Multiplying probability × loss × importance
produces a single rupee figure that leadership can compare across assets and use to justify budgets.

**Total Enterprise Risk** is simply the sum of all individual asset EALs:
""")
    st.code(f"Total Enterprise Risk = Σ(EAL) = {format_inr(total_risk)}", language="text")

with xai_tab2:
    st.write("**Step-by-step EAL calculation for every asset:**")
    breakdown = df[["asset_name", "asset_type", "likelihood", "likelihood_pct",
                     "potential_financial_impact_inr", "criticality_weight",
                     "expected_annual_loss_inr"]].copy()
    breakdown["calculation"] = breakdown.apply(
        lambda r: (
            f"{r['likelihood']:.4f} × {format_inr_short(r['potential_financial_impact_inr'])} "
            f"× {r['criticality_weight']:.2f} = {format_inr_short(r['expected_annual_loss_inr'])}"
        ), axis=1
    )
    st.dataframe(
        breakdown[["asset_name", "asset_type", "likelihood_pct",
                    "potential_financial_impact_inr", "criticality_weight",
                    "expected_annual_loss_inr", "calculation"]]
        .rename(columns={
            "asset_name": "Asset",
            "asset_type": "Type",
            "likelihood_pct": "Likelihood %",
            "potential_financial_impact_inr": "Impact (₹)",
            "criticality_weight": "Criticality",
            "expected_annual_loss_inr": "EAL (₹)",
            "calculation": "Calculation",
        }),
        use_container_width=True,
        hide_index=True,
    )

with xai_tab3:
    st.markdown(f"""
### Investment Optimizer Logic

**Algorithm:** Greedy Knapsack Optimization
- All assets are ranked by **risk reduction per rupee spent** (ROI = risk_reduction ÷ remediation_cost)
- Assets are selected top-down until the budget is exhausted
- This maximizes total risk reduction for any given budget

**Current Budget:** {format_inr_short(budget)}
**Assets Selected:** {len(selected)}
**Total Risk Reduced:** {format_inr_short(reduced)}
**Budget Remaining:** {format_inr_short(max(0, budget - (selected['remediation_cost_inr'].sum() if len(selected) else 0)))}
""")

    if len(selected):
        st.write("**Why each asset was selected:**")
        explanations = explain_budget_allocation(selected.head(10), budget=budget)
        for i, exp in enumerate(explanations):
            st.info(f"**#{i+1}** {exp}")
    else:
        st.warning("No assets selected at this budget level.")

with xai_tab4:
    st.markdown(f"""
### ML Model Card — Attack Likelihood Predictor

| Property | Value |
|---|---|
| **Algorithm** | RandomForest Regressor (200 trees) |
| **Target Variable** | `likelihood` (probability of attack, 0-1) |
| **Features Used** | `vulnerability_count`, `criticality_weight`, `asset_type` (one-hot encoded) |
| **Train/Test Split** | 75% / 25% |
| **Mean Absolute Error** | {mae:.4f} |
| **Purpose** | Rank which factors drive risk — the "AI Decision Support" component |

**Feature Importance Ranking** (which inputs matter most to the model):
""")
    imp_df = importance.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    imp_df["Importance %"] = (imp_df["Importance"] * 100).round(2)
    st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.warning(
        "⚠️ **Important Note:** This model is trained on the current dataset snapshot. "
        "In production, it would be continuously retrained on real telemetry from "
        "SIEM, EDR, and vulnerability scanners for validated predictions."
    )


st.divider()
st.caption("Prototype built for SIH26105 — Team demo. Data shown is synthetic, generated to "
           "illustrate the platform's logic, not real company data.")
