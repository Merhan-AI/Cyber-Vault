"""
risk_engine.py
--------------
Core risk quantification logic for SIH26105.

Formula:
    Expected Annual Loss (EAL) = Likelihood x Financial Impact x Criticality Weight

This converts vague "Low/Medium/High" risk labels into a concrete rupee
figure that business leaders can act on.
"""

import pandas as pd


def calculate_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add an Expected Annual Loss (EAL) column to the asset dataframe."""
    df = df.copy()
    df["expected_annual_loss_inr"] = (
        df["likelihood"]
        * df["potential_financial_impact_inr"]
        * df["criticality_weight"]
    ).round(0)
    return df.sort_values("expected_annual_loss_inr", ascending=False).reset_index(drop=True)


def total_enterprise_risk(df: pd.DataFrame) -> float:
    """Sum of EAL across all assets = total organizational financial exposure."""
    return float(df["expected_annual_loss_inr"].sum())


def top_risky_assets(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.nlargest(n, "expected_annual_loss_inr")[
        ["asset_name", "asset_type", "expected_annual_loss_inr", "likelihood_pct", "criticality_weight"]
    ]


def format_inr(amount: float) -> str:
    """Format a number as an Indian Rupee string, e.g. 4000000 -> '₹40,00,000'."""
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"₹{','.join(parts)},{last3}"


if __name__ == "__main__":
    df = pd.read_csv("data/assets.csv")
    df = calculate_risk(df)

    total = total_enterprise_risk(df)
    print(f"Total Enterprise Risk (Expected Annual Loss): {format_inr(total)}\n")

    print("Top 5 Riskiest Assets:")
    print(top_risky_assets(df).to_string(index=False))

    df.to_csv("data/assets_with_risk.csv", index=False)
    print("\nSaved -> data/assets_with_risk.csv")
