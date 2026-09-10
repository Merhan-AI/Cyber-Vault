"""
optimizer.py
------------
Investment Optimization Module for SIH26105.

Given a fixed security budget, recommends which assets to remediate
first to maximize total risk reduction (a knapsack-style problem,
solved here with a simple, explainable greedy algorithm ranked by
"risk reduction per rupee spent").
"""

import pandas as pd
import numpy as np

RANDOM_SEED = 7


def build_remediation_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each asset, simulate a remediation option: a cost to fix it, and
    the risk reduction achieved (assume remediation removes 60-90% of
    that asset's current EAL).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    df = df.copy()

    # Remediation cost loosely scales with criticality + impact
    df["remediation_cost_inr"] = (
        (df["potential_financial_impact_inr"] * 0.02)
        * rng.uniform(0.5, 1.5, size=len(df))
    ).round(0)

    reduction_fraction = rng.uniform(0.6, 0.9, size=len(df))
    df["risk_reduction_inr"] = (df["expected_annual_loss_inr"] * reduction_fraction).round(0)

    df["reduction_per_rupee"] = df["risk_reduction_inr"] / df["remediation_cost_inr"].replace(0, 1)
    return df


def optimize_budget(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """
    Greedy allocation: sort assets by risk-reduction-per-rupee (best ROI
    first) and keep adding remediations until the budget runs out.
    """
    ranked = df.sort_values("reduction_per_rupee", ascending=False).copy()

    selected_rows = []
    remaining_budget = budget
    for _, row in ranked.iterrows():
        if row["remediation_cost_inr"] <= remaining_budget:
            selected_rows.append(row)
            remaining_budget -= row["remediation_cost_inr"]

    return pd.DataFrame(selected_rows)


def risk_reduction_curve(df: pd.DataFrame, max_budget: float, steps: int = 20) -> pd.DataFrame:
    """Compute total risk reduction achieved at increasing budget levels (for the chart)."""
    budgets = np.linspace(0, max_budget, steps)
    results = []
    for b in budgets:
        selected = optimize_budget(df, b)
        results.append({
            "budget_inr": b,
            "risk_reduction_inr": selected["risk_reduction_inr"].sum() if len(selected) else 0,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    from risk_engine import calculate_risk, format_inr

    df = pd.read_csv("data/assets.csv")
    df = calculate_risk(df)
    df = build_remediation_options(df)

    budget = 1_00_00_000  # ₹1 crore
    selected = optimize_budget(df, budget)

    print(f"Budget: {format_inr(budget)}")
    print(f"Assets remediated: {len(selected)}")
    print(f"Total risk reduced: {format_inr(selected['risk_reduction_inr'].sum())}\n")
    print(selected[["asset_name", "remediation_cost_inr", "risk_reduction_inr"]].to_string(index=False))
