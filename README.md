# Cyber Risk Quantification Platform (Prototype)

Prototype for **SIH26105 — AI-Powered Continuous Cyber Risk Quantification
and Investment Optimization Platform**.

Converts vague "Low/Medium/High" cyber risk labels into a concrete
financial number (₹), and recommends where to invest a limited security
budget for maximum risk reduction.

## What's inside

| File | Purpose |
|---|---|
| `generate_data.py` | Creates synthetic asset data (simulates SIEM/EDR/vulnerability scanner output) |
| `risk_engine.py` | Core formula: `EAL = Likelihood x Financial Impact x Criticality` |
| `ml_layer.py` | RandomForest model that ranks which factors drive risk the most |
| `optimizer.py` | Greedy budget allocator — recommends best ROI remediations |
| `app.py` | Streamlit dashboard tying everything together |

## How to run

```bash
pip install -r requirements.txt
python generate_data.py      # optional — app.py will auto-generate data if missing
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## How to explain this in the demo

1. **Data Sources** → `generate_data.py` simulates what would normally come
   from SIEM, EDR, vulnerability scanners, and asset inventories.
2. **Risk Quantification Engine** → `risk_engine.py` turns that raw data
   into a rupee-denominated Expected Annual Loss per asset.
3. **AI/ML Layer** → `ml_layer.py` shows which factors (vulnerability count,
   criticality, asset type) drive the risk score — the "AI Decision
   Support" part of the problem statement.
4. **Investment Optimization** → `optimizer.py` answers "given ₹1 crore,
   what should we fix first?"
5. **Dashboard** → `app.py` is the executive-facing view: total risk,
   top risky assets, a live budget slider, and a simple Q&A box.

## Important note for judges/demo

The dataset is **synthetic**, generated to demonstrate the platform's
logic end-to-end. The ML model is trained on this simulated snapshot, so
treat its predictions as an illustration of the AI workflow — not a
validated real-world security model. In a production deployment, this
would be trained continuously on real telemetry from actual security
tools (SIEM, EDR, vulnerability scanners) across the organization.
