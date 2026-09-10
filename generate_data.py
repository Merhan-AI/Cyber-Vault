"""
generate_data.py
-----------------
Generates a synthetic dataset of enterprise IT/security assets for the
Cyber Risk Quantification prototype (SIH26105).

In a real deployment, this data would come from SIEM, EDR, vulnerability
scanners, IAM, and asset inventory tools. For this prototype we simulate
that data so the rest of the pipeline (risk engine, ML layer, optimizer,
dashboard) can be demonstrated end-to-end without real company data.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_ASSETS = 40

ASSET_TYPES = [
    "Database", "Web Application", "Payment System", "Email Server",
    "Internal Tool", "Cloud Storage", "API Gateway", "Employee Endpoint",
]

ASSET_NAME_POOL = [
    "Customer Database", "Payment Gateway", "HR System", "Email Server",
    "Internal Wiki", "Employee Laptop Fleet", "CRM Platform", "Billing API",
    "Cloud File Storage", "VPN Gateway", "Analytics Dashboard", "Backup Server",
    "Marketing Website", "Partner API", "Inventory System", "Support Portal",
    "Mobile App Backend", "Authentication Service", "Data Warehouse",
    "Vendor Management Tool",
]


def generate_dataset(num_assets: int = NUM_ASSETS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(num_assets):
        base_name = rng.choice(ASSET_NAME_POOL)
        asset_name = f"{base_name} #{i + 1}" if i >= len(ASSET_NAME_POOL) else base_name
        asset_type = rng.choice(ASSET_TYPES)

        vulnerability_count = int(rng.integers(0, 25))

        # Likelihood of attack (0-1), loosely driven by vulnerability count
        base_likelihood = 0.05 + (vulnerability_count / 25) * 0.5
        likelihood = float(np.clip(base_likelihood + rng.normal(0, 0.05), 0.01, 0.95))

        # Financial impact if breached (in INR)
        potential_financial_impact = int(rng.integers(5, 150)) * 100000  # 5L to 1.5Cr

        # Business criticality weight (0.1 = low importance, 1.0 = critical)
        criticality_weight = round(float(rng.uniform(0.1, 1.0)), 2)

        rows.append({
            "asset_id": f"A{i + 1:03d}",
            "asset_name": asset_name,
            "asset_type": asset_type,
            "vulnerability_count": vulnerability_count,
            "likelihood_pct": round(likelihood * 100, 2),
            "likelihood": round(likelihood, 4),
            "potential_financial_impact_inr": potential_financial_impact,
            "criticality_weight": criticality_weight,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/assets.csv", index=False)
    print(f"Generated {len(df)} synthetic assets -> data/assets.csv")
    print(df.head())
