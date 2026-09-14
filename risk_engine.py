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
import numpy as np


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


def format_inr_short(amount: float) -> str:
    """Compact Indian Rupee format: ₹45,000 / ₹3.13L / ₹1.2Cr."""
    amount = abs(float(amount))
    if amount >= 1_00_00_000:
        val = amount / 1_00_00_000
        return f"₹{val:.1f}Cr" if val != int(val) else f"₹{int(val)}Cr"
    elif amount >= 1_00_000:
        val = amount / 1_00_000
        return f"₹{val:.2f}L" if val != round(val, 1) else f"₹{val:.1f}L"
    else:
        return format_inr(amount)


def calculate_var(df: pd.DataFrame, confidence: float = 0.95, n_trials: int = 5000) -> float:
    """
    Compute the Value at Risk (VaR) for the asset portfolio using Monte Carlo
    simulation.

    For each trial, every asset either "breaches" (with probability = its
    likelihood, contributing impact × criticality) or doesn't (contributing 0).
    The VaR is the loss value at the chosen confidence percentile across all
    trials — i.e. there is only a (1 - confidence) chance that the real loss
    in a given year exceeds this number.
    """
    rng = np.random.default_rng(42)

    likelihoods = df["likelihood"].values
    losses = (df["potential_financial_impact_inr"] * df["criticality_weight"]).values

    # Monte Carlo: each row of `coin` is one trial, each column is one asset
    coin = rng.random((n_trials, len(df)))          # uniform [0, 1)
    breached = coin < likelihoods                   # True where the asset is hit
    trial_totals = (breached * losses).sum(axis=1)  # total loss per trial

    return float(np.percentile(trial_totals, confidence * 100))


def format_var_summary(var_95: float, eal_total: float) -> str:
    """
    Return a plain-English sentence comparing the expected annual loss with
    the 95th-percentile Value at Risk.
    """
    return (
        f"While the expected annual loss is {format_inr_short(eal_total)}, "
        f"there is a 5% chance losses could exceed {format_inr_short(var_95)} "
        f"in a bad year."
    )


if __name__ == "__main__":
    df = pd.read_csv("data/assets.csv")
    df = calculate_risk(df)

    total = total_enterprise_risk(df)
    print(f"Total Enterprise Risk (Expected Annual Loss): {format_inr(total)}\n")

    print("Top 5 Riskiest Assets:")
    print(top_risky_assets(df).to_string(index=False))

    var_95 = calculate_var(df)
    print(f"\nValue at Risk (95th percentile): {format_inr(var_95)}")
    print(format_var_summary(var_95, total))

    df.to_csv("data/assets_with_risk.csv", index=False)
    print("\nSaved -> data/assets_with_risk.csv")
