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


def optimize_budget(df: pd.DataFrame, budget: float, include_explanation: bool = False) -> pd.DataFrame:
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

    res = pd.DataFrame(selected_rows)
    if include_explanation and len(res):
        res["explanation"] = explain_budget_allocation(res, budget=budget)
    return res


def explain_asset_choice(row: pd.Series, rank: int = 1) -> str:
    """Explain in plain English why an individual asset was prioritized."""
    name = row.get("asset_name", "Asset")
    cost = row.get("remediation_cost_inr", 0)
    reduction = row.get("risk_reduction_inr", 0)
    ratio = row.get("reduction_per_rupee", 0)
    if (ratio == 0 or pd.isna(ratio)) and cost > 0:
        ratio = reduction / cost

    try:
        from risk_engine import format_inr
        cost_str = format_inr(cost)
        red_str = format_inr(reduction)
    except ImportError:
        cost_str = f"₹{int(round(cost)):,}"
        red_str = f"₹{int(round(reduction)):,}"

    if rank == 1:
        return (
            f"'{name}' was prioritized #1 because it offers the highest risk reduction per rupee spent "
            f"across all assets. Remediating it costs {cost_str} and eliminates {red_str} in expected annual loss, "
            f"yielding the top return of ₹{ratio:.2f} of risk mitigated for every ₹1 invested."
        )
    else:
        return (
            f"'{name}' was prioritized #{rank} because it offers high risk reduction per rupee spent "
            f"(₹{ratio:.2f} risk reduced per ₹1 spent), eliminating {red_str} in expected annual loss for a "
            f"remediation cost of {cost_str}."
        )


def explain_budget_allocation(
    selected: pd.DataFrame,
    budget: float = None,
    as_text: bool = False,
) -> list[str] | str:
    """
    Explains in plain English why specific assets were chosen for a given budget.

    Parameters:
        selected (pd.DataFrame): DataFrame of selected assets (from optimize_budget),
                                 or full asset candidates DataFrame if budget is provided.
        budget (float, optional): Total security budget allocated.
        as_text (bool, optional): If True, returns a formatted plain-English text report.
                                  If False (default), returns a list of individual asset
                                  plain-English explanations.

    Returns:
        list[str] or str: Plain-English explanation(s).
    """
    try:
        from risk_engine import format_inr
    except ImportError:
        def format_inr(amount: float) -> str:
            return f"₹{int(round(amount)):,}"

    # If full candidate dataframe was passed with a budget, run optimization first
    if budget is not None and len(selected) > 0 and "remediation_cost_inr" in selected.columns:
        if selected["remediation_cost_inr"].sum() > budget:
            selected = optimize_budget(selected, budget)

    if selected is None or len(selected) == 0:
        msg = (
            f"No assets could be remediated within the allocated budget of {format_inr(budget)} "
            "because all remediation costs exceed the available funds."
            if budget is not None
            else "No assets were selected for remediation."
        )
        return msg if as_text else [msg]

    explanations = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        explanations.append(explain_asset_choice(row, rank=rank))

    if as_text:
        total_cost = selected["remediation_cost_inr"].sum()
        total_reduction = selected["risk_reduction_inr"].sum()
        remaining = (budget - total_cost) if budget is not None else 0
        budget_str = f" of {format_inr(budget)}" if budget is not None else ""

        header = [
            "=== Security Budget Allocation Rationale ===",
            "Allocation Strategy: Greedy Knapsack Optimization (maximizing risk reduction per rupee spent).",
            f"Summary: With a total budget{budget_str}, {len(selected)} asset(s) were selected for remediation.",
            f"Total Investment: {format_inr(total_cost)}" + (f" (Remaining Unallocated: {format_inr(remaining)})" if budget is not None else ""),
            f"Total Risk Reduced: {format_inr(total_reduction)} in Expected Annual Loss.",
            "\nDetailed Asset Prioritization Reasons:",
        ]
        bullet_points = [f"{i+1}. {exp}" for i, exp in enumerate(explanations)]
        return "\n".join(header + bullet_points)

    return explanations


# Aliases for flexible API usage
explain_selection = explain_budget_allocation
explain_asset_selection = explain_budget_allocation


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

    print("\n" + "=" * 60)
    print("PLAIN-ENGLISH EXPLANATIONS (Top 5 Selected Assets):")
    print("=" * 60)
    for exp in explain_selection(selected.head(5), budget):
        print(f"• {exp}\n")

    print("=" * 60)
    print("EXECUTIVE SUMMARY (as_text=True):")
    print("=" * 60)
    print(explain_selection(selected.head(3), budget, as_text=True))

