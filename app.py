"""
app.py
------
Streamlit dashboard for the SIH26105 Cyber Risk Quantification platform.
Styled with the Obsidian Telemetry Design System (Stitch v4.2).

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from generate_data import generate_dataset
from risk_engine import (
    calculate_risk,
    total_enterprise_risk,
    top_risky_assets,
    format_inr,
    format_inr_short,
    calculate_var,
    format_var_summary,
)
from ml_layer import train_likelihood_model, explain_asset
from optimizer import (
    build_remediation_options,
    optimize_budget,
    risk_reduction_curve,
    explain_budget_allocation,
)
from data_ingestion import ingest_user_data
from ai_config import ask_ai, get_status, is_ai_configured

st.set_page_config(
    page_title="Cyber-Vault | Actuarial Risk Command Center",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# 1. OBSIDIAN TELEMETRY CYBER DESIGN SYSTEM (CSS)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');

    /* Global resets */
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0d1728 0%, #080c14 70%, #05080e 100%) !important;
        color: #dfe2ee !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 95% !important;
    }

    /* Headlines */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.02em;
        color: #f8fafc !important;
    }

    /* Monospaced numbers */
    .mono-num, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: 'tnum' 1, 'zero' 1;
    }

    /* Top Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background: rgba(13, 19, 31, 0.7);
        padding: 0.4rem 0.6rem;
        border-radius: 8px;
        border: 1px solid #1e293b;
        backdrop-filter: blur(12px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        border-radius: 6px;
        padding: 0 1.2rem;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00f2fe;
        background: rgba(0, 242, 254, 0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.12) !important;
        border: 1px solid rgba(0, 242, 254, 0.35) !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
    }

    /* Containers & Cards */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(13, 19, 31, 0.85) !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: border-color 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        border-color: rgba(0, 242, 254, 0.3) !important;
    }

    /* Custom HUD Telemetry Banner */
    .hud-banner {
        background: linear-gradient(135deg, rgba(13, 22, 38, 0.95) 0%, rgba(8, 12, 20, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-left: 5px solid #00f2fe;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(0, 242, 254, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .hud-title-wrap {
        display: flex;
        flex-direction: column;
    }
    .hud-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #00f2fe;
        padding: 3px 10px;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
        width: fit-content;
        margin-bottom: 6px;
    }
    .pulse-pip {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; box-shadow: 0 0 4px #10b981; }
        50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 10px #10b981; }
        100% { transform: scale(0.95); opacity: 0.8; box-shadow: 0 0 4px #10b981; }
    }
    .hud-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hud-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 4px 0 0 0;
    }

    /* Metric HUD Cards */
    .metric-card {
        background: rgba(19, 27, 43, 0.7);
        border: 1px solid rgba(30, 41, 59, 0.8);
        border-radius: 8px;
        padding: 1.2rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.08);
    }
    .metric-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f2fe, transparent);
        opacity: 0.5;
    }
    .metric-card-cyan::after { background: linear-gradient(90deg, transparent, #00f2fe, transparent); }
    .metric-card-amber::after { background: linear-gradient(90deg, transparent, #f59e0b, transparent); }
    .metric-card-emerald::after { background: linear-gradient(90deg, transparent, #10b981, transparent); }
    .metric-card-violet::after { background: linear-gradient(90deg, transparent, #818cf8, transparent); }

    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .metric-footer {
        font-size: 0.78rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Comparison Box */
    .compare-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .compare-card-legacy {
        background: rgba(30, 20, 25, 0.6);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1.2rem;
    }
    .compare-card-modern {
        background: rgba(16, 32, 28, 0.6);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1.2rem;
    }
    
    /* Tag Pills */
    .tag-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 500;
    }
    .tag-cyan { background: rgba(0, 242, 254, 0.12); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.3); }
    .tag-emerald { background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .tag-amber { background: rgba(245, 158, 11, 0.12); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-crimson { background: rgba(239, 68, 68, 0.12); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Custom Gateway */
    .gateway-hero {
        background: linear-gradient(180deg, rgba(13, 24, 44, 0.8) 0%, rgba(8, 12, 20, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 3.5rem 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
    }
    .gateway-feature-strip {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }
    
    /* Button overrides */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #00f2fe 0%, #0284c7 100%) !important;
        color: #080c14 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.6) !important;
        transform: translateY(-1px) !important;
    }
    .stButton button[kind="secondary"] {
        background: rgba(19, 27, 43, 0.8) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        color: #00f2fe !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
    }
    .stButton button[kind="secondary"]:hover {
        border-color: #00f2fe !important;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.25) !important;
    }
    
    /* Inputs */
    input, textarea {
        background-color: #070a10 !important;
        border: 1px solid #1e293b !important;
        color: #f8fafc !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    input:focus, textarea:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 0 1px #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)

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


if "data_source" not in st.session_state:
    st.session_state.data_source = None
if "custom_df" not in st.session_state:
    st.session_state.custom_df = None


# =====================================================================
# APP ROUTING: 1. THE UPLOAD GATEWAY (FIRST IMPRESSION)
# =====================================================================
if st.session_state.data_source is None:
    st.markdown("""
        <div class="gateway-hero">
            <div class="hud-badge"><span class="pulse-pip"></span> SIH26105 AI RISK PLATFORM</div>
            <h1 style="font-size: 3.2rem; margin-top: 0.5rem; margin-bottom: 0.5rem; letter-spacing: -0.04em;">Cyber-Vault</h1>
            <p style="font-size: 1.25rem; color: #94a3b8; max-width: 750px; margin: 0 auto; line-height: 1.6;">
                Translating Technical Vulnerabilities into Rupee-Denominated Financial Exposure (<span style="color:#00f2fe; font-family:'JetBrains Mono';">EAL & 95% VaR</span>) and Optimizing Security Capital ROI.
            </p>
            <div class="gateway-feature-strip">
                <span class="tag-pill tag-cyan">Open FAIR™ Actuarial Framework</span>
                <span class="tag-pill tag-emerald">Greedy Knapsack ROI Optimizer</span>
                <span class="tag-pill tag-amber">XAI & SHAP Risk Drivers</span>
                <span class="tag-pill tag-crimson">Autonomous Cyber Actuary</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("""
                <div style="margin-bottom:12px;">
                    <h3 style="margin:0; font-size:1.3rem;">Upload Infrastructure Telemetry</h3>
                </div>
                <p style="color:#94a3b8; font-size:0.92rem; margin-bottom:1rem;">
                    Upload custom asset inventories (CSV / XLSX) containing vulnerability, criticality, or impact telemetry.
                </p>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Drop telemetry file here", type=["csv", "xlsx"], label_visibility="collapsed")
            if uploaded_file:
                with st.spinner("Executing actuarial risk engine and training XAI models..."):
                    parsed_df, msgs = ingest_user_data(uploaded_file)
                    if len(parsed_df) == 0:
                        st.error("Could not process the uploaded file.")
                        for m in msgs:
                            st.error(m)
                    else:
                        parsed_df = calculate_risk(parsed_df)
                        parsed_df = build_remediation_options(parsed_df)
                        st.session_state.custom_df = parsed_df
                        st.session_state.data_source = "custom"
                        st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("""
                <div style="margin-bottom:12px;">
                    <h3 style="margin:0; font-size:1.3rem;">Launch Live Demo Command Center</h3>
                </div>
                <p style="color:#94a3b8; font-size:0.92rem; margin-bottom:1.5rem;">
                    Instant access to pre-calibrated enterprise dataset with 50 synthetic assets, full XAI attribution, and live ROI simulations.
                </p>
            """, unsafe_allow_html=True)
            if st.button("Initialize Command Center", type="primary", use_container_width=True):
                st.session_state.data_source = "demo"
                st.rerun()

    st.stop()

# =====================================================================
# ACTIVE PLATFORM STATE
# =====================================================================
if st.session_state.data_source == "custom":
    df = st.session_state.custom_df
else:
    df = load_data()

_ai_status = get_status()
total_risk = total_enterprise_risk(df)
_var_95 = calculate_var(df)
model, importance, mae = train_likelihood_model(df)
_default_budget = 1_00_00_000

# Top Navigation & HUD Header Bar
st.markdown(f"""
    <div class="hud-banner">
        <div class="hud-title-wrap">
            <div class="hud-badge"><span class="pulse-pip"></span> ACTUARIAL ENGINE v4.2 LIVE &bull; {'CUSTOM TELEMETRY' if st.session_state.data_source == 'custom' else 'SYNTHETIC ENTERPRISE TELEMETRY'}</div>
            <div class="hud-title">Cyber-Vault <span style="font-size:1.1rem; font-weight:500; color:#00f2fe; margin-left:8px; font-family:'JetBrains Mono';">COMMAND CENTER</span></div>
            <p class="hud-subtitle">Quantitative Cyber Risk Financial Intelligence & Capital Investment Optimization</p>
        </div>
        <div style="display:flex; gap:12px; align-items:center;">
            <span class="tag-pill tag-cyan">FAIR™ Quantitative</span>
            <span class="tag-pill tag-emerald">Monte Carlo VaR 95%</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Quick disconnect header row
col_ctrl1, col_ctrl2 = st.columns([5, 1])
with col_ctrl2:
    if st.button("Switch Dataset", use_container_width=True):
        st.session_state.data_source = None
        st.session_state.custom_df = None
        st.rerun()

# KPI Metric Strip
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.markdown(f"""
        <div class="metric-card metric-card-cyan">
            <div class="metric-label">Total Enterprise Exposure</div>
            <div class="metric-val">{format_inr_short(total_risk)}</div>
            <div class="metric-footer"><span style="color:#00f2fe;">Expected Annual Loss (EAL)</span></div>
        </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
        <div class="metric-card metric-card-amber">
            <div class="metric-label">95% Value at Risk (VaR)</div>
            <div class="metric-val">{format_inr_short(_var_95)}</div>
            <div class="metric-footer"><span style="color:#f59e0b;">1-in-20 Year Extreme Tail Risk</span></div>
        </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
        <div class="metric-card metric-card-emerald">
            <div class="metric-label">Monitored Assets</div>
            <div class="metric-val">{len(df)}</div>
            <div class="metric-footer"><span style="color:#10b981;">100% Telemetry Synchronized</span></div>
        </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    high_crit_count = int((df["criticality_weight"] > 0.7).sum())
    st.markdown(f"""
        <div class="metric-card metric-card-violet">
            <div class="metric-label">High-Criticality Assets</div>
            <div class="metric-val">{high_crit_count}</div>
            <div class="metric-footer"><span style="color:#818cf8;">Crown Jewels (Weight &gt; 0.70)</span></div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN NAVIGATION TABS ---
tab_exec, tab_analytics, tab_roi, tab_chat, tab_xai = st.tabs([
    "Executive Overview",
    "Risk Analytics & Heatmaps",
    "ROI & Capital Allocator",
    "Autonomous AI Actuary",
    "Explainability (XAI) & Audit",
])

# =====================================================================
# 1. EXECUTIVE OVERVIEW
# =====================================================================
with tab_exec:
    st.markdown("""
        <div class="compare-grid">
            <div class="compare-card-legacy">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <strong style="color:#ef4444; font-family:'Space Grotesk'; font-size:1.05rem;">Legacy Ordinal Matrix</strong>
                    <span class="tag-pill tag-crimson">Subjective</span>
                </div>
                <div style="font-size:1.4rem; font-weight:700; color:#ffffff; font-family:'Space Grotesk';">Risk Score: "HIGH"</div>
                <p style="color:#94a3b8; font-size:0.88rem; margin-top:6px; margin-bottom:0;">
                    Vague qualitative labels create ambiguity. Unclear to CFO/Board which assets warrant capital expenditure.
                </p>
            </div>
            <div class="compare-card-modern">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <strong style="color:#10b981; font-family:'Space Grotesk'; font-size:1.05rem;">Cyber-Vault FAIR™ Actuarial Platform</strong>
                    <span class="tag-pill tag-emerald">Empirical (₹)</span>
                </div>
                <div style="font-size:1.4rem; font-weight:700; color:#00f2fe; font-family:'JetBrains Mono';">Expected Loss: """ + format_inr_short(total_risk) + """</div>
                <p style="color:#94a3b8; font-size:0.88rem; margin-top:6px; margin-bottom:0;">
                    Continuous actuarial calculation: Likelihood &times; Financial Impact &times; Criticality Weight. Actionable and board-ready.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style="margin:0;">Actuarial VaR Confidence Analysis</h4>
                <span class="tag-pill tag-amber">Confidence Level: 95.0%</span>
            </div>
            <p style="color:#94a3b8; font-size:0.92rem; margin-top:6px;">
                {format_var_summary(_var_95, total_risk)}
            </p>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Top 5 Riskiest Enterprise Assets")
        _top_raw = top_risky_assets(df).copy()
        _top_display = _top_raw.rename(columns={
            "asset_name": "Asset Name",
            "asset_type": "Asset Type",
            "expected_annual_loss_inr": "Expected Annual Loss (₹)",
            "likelihood_pct": "Likelihood (%)",
            "criticality_weight": "Criticality Weight",
        })
        _top_display["Expected Annual Loss (₹)"] = _top_display["Expected Annual Loss (₹)"].apply(lambda x: f"{int(round(x)):,}")
        _top_display["Likelihood (%)"] = _top_display["Likelihood (%)"].apply(lambda x: f"{x:.2f}%")
        _top_display["Criticality Weight"] = _top_display["Criticality Weight"].apply(lambda x: f"{x:.2f}")
        st.dataframe(_top_display, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### AI Explainability (SHAP Vectors) for Top Assets")
        _top5 = df.nlargest(5, "expected_annual_loss_inr")
        _shap_features = df[["vulnerability_count", "criticality_weight", "asset_type"]].copy()
        _shap_features = pd.get_dummies(_shap_features, columns=["asset_type"], drop_first=True)
        for _, _row in _top5.iterrows():
            _asset_idx = _row.name
            with st.expander(f"Why this score? — {_row['asset_name']} (EAL: {format_inr_short(_row['expected_annual_loss_inr'])})"):
                _explanation = explain_asset(model, _shap_features, _asset_idx)
                st.markdown(_explanation)

# =====================================================================
# 2. RISK ANALYTICS
# =====================================================================
with tab_analytics:
    with st.container(border=True):
        st.markdown("### Enterprise Risk Telemetry & Distribution")
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
            "Loss Distribution by Asset",
            "Loss Exposure by Asset Type",
            "Vulnerability vs. Attack Likelihood",
            "Actuarial Risk Composition",
        ])

        with chart_tab1:
            st.caption("Top 20 Assets ranked by Expected Annual Loss (₹)")
            chart_df = df[["asset_name", "expected_annual_loss_inr"]].rename(
                columns={"asset_name": "Asset", "expected_annual_loss_inr": "Expected Annual Loss (₹)"}
            ).sort_values("Expected Annual Loss (₹)", ascending=True).tail(20)
            
            c_loss = alt.Chart(chart_df).mark_bar(color="#00f2fe", cornerRadiusEnd=4).encode(
                x=alt.X("Expected Annual Loss (₹):Q", title="Expected Annual Loss (₹)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#00f2fe")),
                y=alt.Y("Asset:N", title="Enterprise Asset", sort=None, axis=alt.Axis(labelColor="#dfe2ee", titleColor="#94a3b8")),
                tooltip=["Asset:N", "Expected Annual Loss (₹):Q"]
            ).properties(height=450)
            st.altair_chart(c_loss, use_container_width=True)

        with chart_tab2:
            st.caption("Aggregated risk exposure across infrastructure tiers")
            type_agg = df.groupby("asset_type").agg(
                total_eal=("expected_annual_loss_inr", "sum"),
                count=("asset_id", "count"),
                avg_likelihood=("likelihood_pct", "mean"),
            ).sort_values("total_eal", ascending=False).reset_index()

            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                c_type = alt.Chart(type_agg).mark_bar(color="#818cf8", cornerRadiusEnd=4).encode(
                    x=alt.X("asset_type:N", title="Asset Type", axis=alt.Axis(labelColor="#dfe2ee", titleColor="#818cf8", labelAngle=-20)),
                    y=alt.Y("total_eal:Q", title="Total EAL (₹)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#818cf8")),
                    tooltip=["asset_type:N", "total_eal:Q", "count:Q"]
                ).properties(height=350)
                st.altair_chart(c_type, use_container_width=True)
            with col_b:
                display_type = type_agg.copy()
                display_type["total_eal"] = display_type["total_eal"].apply(lambda x: f"{int(round(x)):,}")
                display_type["avg_likelihood"] = display_type["avg_likelihood"].map(lambda x: f"{x:.1f}%")
                display_type.columns = ["Asset Type", "Total EAL (₹)", "Count", "Avg Likelihood"]
                st.dataframe(display_type, use_container_width=True, hide_index=True)

        with chart_tab3:
            st.caption("Bubble size corresponds to Expected Annual Loss (₹), color indicates business criticality")
            scatter_df = df[["asset_name", "vulnerability_count", "likelihood_pct",
                             "expected_annual_loss_inr", "criticality_weight"]].copy()
            c_scat = alt.Chart(scatter_df).mark_circle().encode(
                x=alt.X("vulnerability_count:Q", title="Vulnerability Count (CVEs)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#00f2fe")),
                y=alt.Y("likelihood_pct:Q", title="Breach Likelihood (%)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#00f2fe")),
                size=alt.Size("expected_annual_loss_inr:Q", title="EAL Exposure (₹)", scale=alt.Scale(range=[60, 600])),
                color=alt.Color("criticality_weight:Q", title="Criticality Weight", scale=alt.Scale(scheme="plasma")),
                tooltip=["asset_name:N", "vulnerability_count:Q", "likelihood_pct:Q", "expected_annual_loss_inr:Q", "criticality_weight:Q"]
            ).properties(height=400)
            st.altair_chart(c_scat, use_container_width=True)

        with chart_tab4:
            st.caption("Decomposition of technical likelihood risk vs financial impact weight (Top 10)")
            top10 = df.nlargest(10, "expected_annual_loss_inr").copy()
            top10["Likelihood Score"] = top10["likelihood"] * top10["potential_financial_impact_inr"]
            top10["Criticality Multiplier"] = (
                top10["expected_annual_loss_inr"] - top10["Likelihood Score"]
            ).clip(lower=0)
            composition = top10.rename(columns={"asset_name": "Asset"})[["Asset", "Likelihood Score", "Criticality Multiplier"]].set_index("Asset")
            st.bar_chart(composition)

# =====================================================================
# 3. ROI & CAPITAL OPTIMIZER
# =====================================================================
with tab_roi:
    with st.container(border=True):
        st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0;">Security Budget & ROI Optimization</h3>
                <span class="tag-pill tag-emerald">Greedy Knapsack Algorithm</span>
            </div>
            <p style="color:#94a3b8; font-size:0.92rem;">
                Adjust available cybersecurity capital to maximize risk reduction across infrastructure crown jewels.
            </p>
        """, unsafe_allow_html=True)

        slider_col, custom_col = st.columns([3, 1])
        with slider_col:
            slider_budget = st.select_slider(
                "Select Security Budget (₹)",
                options=list(range(0, 2_00_00_001, 5_00_000)),
                value=1_00_00_000,
                format_func=format_inr,
            )
        with custom_col:
            custom_budget = st.number_input(
                "Exact Budget Override (₹)",
                min_value=0,
                max_value=50_00_00_000,
                value=0,
                step=1_00_000,
                help="Set a custom budget amount in rupees.",
            )

        budget = custom_budget if custom_budget > 0 else slider_budget
        selected = optimize_budget(df, budget)
        reduced = selected["risk_reduction_inr"].sum() if len(selected) else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
                <div class="metric-card metric-card-cyan">
                    <div class="metric-label">Allocated Capital</div>
                    <div class="metric-val">{format_inr_short(budget)}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card metric-card-emerald">
                    <div class="metric-label">Total Risk Reduced</div>
                    <div class="metric-val">{format_inr_short(reduced)}</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="metric-card metric-card-violet">
                    <div class="metric-label">Assets Remediated</div>
                    <div class="metric-val">{len(selected)}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if len(selected):
            st.markdown("#### Recommended Remediation Action Plan")
            display_df = selected[["asset_name", "asset_type", "remediation_cost_inr", "risk_reduction_inr"]].copy()
            display_df["remediation_cost_inr"] = display_df["remediation_cost_inr"].apply(lambda x: f"{int(round(x)):,}")
            display_df["risk_reduction_inr"] = display_df["risk_reduction_inr"].apply(lambda x: f"{int(round(x)):,}")
            st.dataframe(
                display_df.rename(columns={
                    "asset_name": "Target Asset",
                    "asset_type": "Infrastructure Type",
                    "remediation_cost_inr": "Estimated Remediation Cost (₹)",
                    "risk_reduction_inr": "Quantified Risk Reduction (₹)",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Diminishing Returns Curve & Optimal Inflection")
        st.caption("Evaluates risk reduction yield across varying budget tiers. The red dot indicates current budget.")

        curve = risk_reduction_curve(df, max_budget=2_00_00_000)
        curve["budget_fmt"] = curve["budget_inr"].map(format_inr_short)
        curve["reduction_fmt"] = curve["risk_reduction_inr"].map(format_inr_short)

        tick_expr = (
            "datum.value >= 10000000 ? '₹' + round(datum.value / 10000000 * 10) / 10 + 'Cr' : "
            "(datum.value >= 100000 ? '₹' + round(datum.value / 100000 * 10) / 10 + 'L' : '₹' + datum.value)"
        )

        base_line = alt.Chart(curve).mark_line(color="#00f2fe", strokeWidth=3).encode(
            x=alt.X(
                "budget_inr:Q",
                title="Security Budget (₹)",
                axis=alt.Axis(labelExpr=tick_expr, labelColor="#94a3b8", titleColor="#00f2fe"),
            ),
            y=alt.Y(
                "risk_reduction_inr:Q",
                title="Total Risk Reduced (₹)",
                axis=alt.Axis(labelExpr=tick_expr, labelColor="#94a3b8", titleColor="#10b981"),
            ),
            tooltip=[
                alt.Tooltip("budget_fmt:N", title="Budget"),
                alt.Tooltip("reduction_fmt:N", title="Risk Reduced"),
            ],
        )

        current_pt_df = pd.DataFrame([{
            "budget_inr": float(budget),
            "risk_reduction_inr": float(reduced),
            "budget_fmt": format_inr_short(budget),
            "reduction_fmt": format_inr_short(reduced),
        }])

        current_marker = alt.Chart(current_pt_df).mark_circle(size=180, color="#ef4444").encode(
            x="budget_inr:Q",
            y="risk_reduction_inr:Q",
            tooltip=[
                alt.Tooltip("budget_fmt:N", title="Selected Budget"),
                alt.Tooltip("reduction_fmt:N", title="Total Risk Reduced"),
            ],
        )

        st.altair_chart(base_line + current_marker, use_container_width=True)

# =====================================================================
# 4. AUTONOMOUS AI ACTUARY
# =====================================================================
def build_data_context(df, budget, selected, reduced, total_risk, model_importance, model_mae):
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


def get_answer(question, df, budget, selected, reduced, total_risk, model_importance, model_mae):
    ai_ready = is_ai_configured()
    if ai_ready:
        context = build_data_context(df, budget, selected, reduced, total_risk, model_importance, model_mae)
        response = ask_ai(question, context)
        if not response.startswith("[Error]") and not response.startswith("Error") and not response.startswith("⚠️"):
            return response

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
                    f"A higher value in any of these three factors raises the Expected Annual Loss proportionally."
                )

    if "highest risk" in q or "biggest risk" in q:
        top = df.iloc[0]
        return (f"Your highest financial risk is **{top['asset_name']}**, "
                f"with an Expected Annual Loss of **{format_inr_short(top['expected_annual_loss_inr'])}**.")
    elif "total risk" in q or "total exposure" in q:
        return f"Your total enterprise cyber risk exposure is **{format_inr_short(total_risk)}** per year."
    elif "budget" in q or "invest" in q:
        return "Check the **ROI & Capital Allocator** tab to see greedy knapsack budget recommendations and risk reduction curves."
    else:
        if ai_ready:
            return "I couldn't generate a response. Please try rephrasing your question."
        return (
            "**Tip:** Configure an AI provider in `.env` for AI-powered answers "
            "to any question about your data.\n\n"
            "Without an API key, I can answer: *'What's our highest risk?'*, "
            "*'What's our total exposure?'*, *'Budget recommendation?'*"
        )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with tab_chat:
    with st.container(border=True):
        st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0;">Autonomous AI Cyber Actuary</h3>
                <span class="tag-pill tag-cyan">FAIR™ Actuarial Reasoning</span>
            </div>
        """, unsafe_allow_html=True)

        if _ai_status["configured"]:
            st.caption(f"Status: AI Online ({_ai_status['provider'].capitalize()} / {_ai_status['model']})")
        else:
            st.caption("Status: Fallback Deterministic Mode (Configure API key in `.env` to enable full LLM)")

        _sel_default = optimize_budget(df, _default_budget)
        _red_default = _sel_default["risk_reduction_inr"].sum() if len(_sel_default) else 0

        st.write("**Quick Query Prompts:**")
        eq1, eq2, eq3 = st.columns(3, gap="small")
        if eq1.button("What's our highest risk asset?", use_container_width=True):
            q = "What's our highest risk?"
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
                q, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae
            )})
        if eq2.button("What's our total financial exposure?", use_container_width=True):
            q = "What's our total exposure?"
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
                q, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae
            )})
        if eq3.button("Budget ROI allocation advice?", use_container_width=True):
            q = "Budget recommendation?"
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
                q, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae
            )})

        st.markdown("---")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(placeholder="Ask anything about your cyber risk posture, CVEs, or budget..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": get_answer(
                prompt, df, _default_budget, _sel_default, _red_default, total_risk, importance, mae
            )})
            st.rerun()

# =====================================================================
# 5. EXPLAINABILITY (XAI) & AUDIT
# =====================================================================
with tab_xai:
    _exp_budget = 1_00_00_000
    _exp_selected = optimize_budget(df, _exp_budget)
    _exp_reduced = _exp_selected["risk_reduction_inr"].sum() if len(_exp_selected) else 0

    with st.container(border=True):
        st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0;">Explainability (XAI) & Audit Transparency</h3>
                <span class="tag-pill tag-violet">Open FAIR™ & Random Forest ML</span>
            </div>
            <p style="color:#94a3b8; font-size:0.92rem;">
                Empirical transparency into how every rupee, probability score, and optimization recommendation was calculated.
            </p>
        """, unsafe_allow_html=True)

        xai_tab1, xai_tab2, xai_tab3, xai_tab4, xai_tab5 = st.tabs([
            "Risk Drivers (Feature Importance)",
            "FAIR Risk Formula",
            "Per-Asset Calculation Ledger",
            "Optimizer Knapsack Logic",
            "ML Model Card",
        ])

        with xai_tab1:
            st.caption("Relative weight of input features in predicting attack likelihood")
            vuln_imp = float(importance.get("vulnerability_count", 0.0))
            crit_imp = float(importance.get("criticality_weight", 0.0))
            asset_type_imp = float(importance[importance.index.str.startswith("asset_type_")].sum())

            grouped_drivers = pd.DataFrame([
                {"Risk Factor": "Vulnerability Count (CVEs)", "Relative Importance": vuln_imp},
                {"Risk Factor": "Criticality Weight", "Relative Importance": crit_imp},
                {"Risk Factor": "Asset Type Classification", "Relative Importance": asset_type_imp},
            ]).sort_values("Relative Importance", ascending=False)
            grouped_drivers["Importance Pct"] = (grouped_drivers["Relative Importance"] * 100).round(2).astype(str) + "%"

            drivers_chart = alt.Chart(grouped_drivers).mark_bar(color="#00f2fe", cornerRadiusEnd=4).encode(
                x=alt.X("Risk Factor:N", title="Risk Factor", sort=alt.SortField("Relative Importance", order="descending"), axis=alt.Axis(labelColor="#dfe2ee", titleColor="#00f2fe")),
                y=alt.Y("Relative Importance:Q", title="Relative Importance", axis=alt.Axis(labelColor="#94a3b8", titleColor="#00f2fe")),
                tooltip=[alt.Tooltip("Risk Factor:N"), alt.Tooltip("Importance Pct:N", title="Relative Importance")],
            ).properties(height=320)
            st.altair_chart(drivers_chart, use_container_width=True)

        with xai_tab2:
            st.markdown("""
### Risk Quantification Formula (FAIR-Aligned)

```
Expected Annual Loss (EAL) = Likelihood × Financial Impact × Criticality Weight
```

| Variable | Description | Range | Source |
|---|---|---|---|
| **Likelihood** | Annual probability of breach on this asset | 0.01 – 0.95 | Vulnerability scans, SIEM & threat feeds |
| **Financial Impact** (₹) | Direct loss if breached (forensics, downtime, fines) | ₹5L – ₹1.5Cr | Business Impact Assessment (BIA) |
| **Criticality Weight** | Operational importance to enterprise | 0.1 – 1.0 | Asset owner classification |

```text
Total Enterprise Risk = Σ(EAL) = """ + format_inr(total_risk) + """
```
            """)

        with xai_tab3:
            st.caption("Step-by-step mathematical derivation for every monitored asset:")
            breakdown = df[["asset_name", "asset_type", "likelihood", "likelihood_pct",
                             "potential_financial_impact_inr", "criticality_weight",
                             "expected_annual_loss_inr"]].copy()
            breakdown["calculation"] = breakdown.apply(
                lambda r: (
                    f"{r['likelihood']:.4f} × {format_inr_short(r['potential_financial_impact_inr'])} "
                    f"× {r['criticality_weight']:.2f} = {format_inr_short(r['expected_annual_loss_inr'])}"
                ), axis=1
            )
            ledger_table = breakdown[["asset_name", "asset_type", "likelihood_pct",
                        "potential_financial_impact_inr", "criticality_weight",
                        "expected_annual_loss_inr", "calculation"]].copy()
            ledger_table["potential_financial_impact_inr"] = ledger_table["potential_financial_impact_inr"].apply(lambda x: f"{int(round(x)):,}")
            ledger_table["expected_annual_loss_inr"] = ledger_table["expected_annual_loss_inr"].apply(lambda x: f"{int(round(x)):,}")
            ledger_table["likelihood_pct"] = ledger_table["likelihood_pct"].apply(lambda x: f"{x:.2f}%")
            ledger_table["criticality_weight"] = ledger_table["criticality_weight"].apply(lambda x: f"{x:.2f}")
            st.dataframe(
                ledger_table.rename(columns={
                    "asset_name": "Asset",
                    "asset_type": "Type",
                    "likelihood_pct": "Likelihood %",
                    "potential_financial_impact_inr": "Impact (₹)",
                    "criticality_weight": "Criticality",
                    "expected_annual_loss_inr": "EAL (₹)",
                    "calculation": "Mathematical Derivation",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with xai_tab4:
            st.markdown(f"""
### Investment Optimizer Logic (Greedy Knapsack)

- Assets ranked by **risk reduction per rupee spent** (ROI = `risk_reduction ÷ remediation_cost`).
- Selected top-down until the capital budget is fully deployed.
- **Baseline Budget:** {format_inr_short(_exp_budget)} | **Assets Selected:** {len(_exp_selected)} | **Total Reduced:** {format_inr_short(_exp_reduced)}
""")
            if len(_exp_selected):
                st.write("**Remediation Allocation Rationale:**")
                explanations = explain_budget_allocation(_exp_selected.head(10), budget=_exp_budget)
                for i, exp in enumerate(explanations):
                    st.info(f"**#{i+1}** {exp}")

        with xai_tab5:
            st.markdown(f"""
### ML Model Card — Attack Likelihood Predictor

| Property | Value |
|---|---|
| **Algorithm** | RandomForest Regressor (500 trees) |
| **Target Variable** | `likelihood` (Attack probability 0.0 - 1.0) |
| **Features Used** | `vulnerability_count`, `criticality_weight`, `asset_type` |
| **Mean Absolute Error** | {mae:.4f} |
""")
            imp_df = importance.reset_index()
            imp_df.columns = ["Feature", "Importance"]
            imp_df["Feature"] = imp_df["Feature"].apply(lambda f: f.replace("_", " ").title())
            imp_df["Importance %"] = (imp_df["Importance"] * 100).round(2)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Cyber-Vault | SIH26105 AI Cyber Risk Quantification & Investment Optimization Command Center.")
