"""
ml_layer.py
-----------
AI Decision Support layer for SIH26105.

Trains a simple, explainable RandomForest model to predict attack
likelihood from asset characteristics, and reports feature importance
so the platform can tell stakeholders WHICH factors drive risk the most
(the "Risk Driver Ranking" feature from the problem statement).

NOTE (important for the demo/judges):
This model is trained on synthetic data for a single organization
snapshot. It demonstrates the AI workflow end-to-end, but its
predictions are illustrative, not a validated real-world security
model. In production this would be trained continuously on real
telemetry from many assets and updated over time.
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

RANDOM_SEED = 42


def train_likelihood_model(df: pd.DataFrame):
    features = df[["vulnerability_count", "criticality_weight", "asset_type"]].copy()
    features = pd.get_dummies(features, columns=["asset_type"], drop_first=True)
    target = df["likelihood"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    importance = pd.Series(model.feature_importances_, index=features.columns)
    importance = importance.sort_values(ascending=False)

    return model, importance, mae


def explain_asset(model, features_df: pd.DataFrame, asset_index: int) -> str:
    """
    Generate a plain-English explanation of the factors that influenced the
    likelihood prediction for a single asset.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained Random Forest model.
    features_df : pd.DataFrame
        One-hot encoded feature matrix (same columns used for training).
    asset_index : int
        Row index of the asset to explain.

    Returns
    -------
    str
        Human-readable sentence describing the two most influential features
        (by absolute SHAP value) and whether they increased or decreased the
        risk score, together with the magnitude of their contribution.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)  # shape: (n_samples, n_features)

    # Extract SHAP values for the requested asset
    asset_shap = shap_values[asset_index]

    # Identify the two features with the largest absolute impact
    top2_idx = np.argsort(np.abs(asset_shap))[-2:][::-1]
    top_features = [(features_df.columns[i], asset_shap[i]) for i in top2_idx]

    # Build explanation sentence
    parts = []
    for feat, val in top_features:
        direction = "increased" if val > 0 else "decreased"
        parts.append(f"{feat.replace('_', ' ')} {direction} the risk by {abs(val):.3f}")
    explanation = " and ".join(parts)
    return f"The prediction for this asset was mainly driven by: {explanation}."


if __name__ == "__main__":
    df = pd.read_csv("data/assets.csv")
    model, importance, mae = train_likelihood_model(df)

    print(f"Model trained. Mean Absolute Error on held-out data: {mae:.4f}\n")
    print("Top Risk Drivers (feature importance):")
    print(importance.head(8).to_string())

    # ----------------------------------------------------------------
    # Demonstrate SHAP-based explainability for the first asset
    # ----------------------------------------------------------------
    features = df[["vulnerability_count", "criticality_weight", "asset_type"]].copy()
    features = pd.get_dummies(features, columns=["asset_type"], drop_first=True)
    explanation_0 = explain_asset(model, features, asset_index=0)
    explanation_5 = explain_asset(model, features, asset_index=5)
    explanation_10 = explain_asset(model, features, asset_index=10)
    print("\nSHAP Explainability Comparison:")
    print(f"  Asset  0: {explanation_0}")
    print(f"  Asset  5: {explanation_5}")
    print(f"  Asset 10: {explanation_10}")
