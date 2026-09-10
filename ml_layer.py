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

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    importance = pd.Series(model.feature_importances_, index=features.columns)
    importance = importance.sort_values(ascending=False)

    return model, importance, mae


if __name__ == "__main__":
    df = pd.read_csv("data/assets.csv")
    model, importance, mae = train_likelihood_model(df)

    print(f"Model trained. Mean Absolute Error on held-out data: {mae:.4f}\n")
    print("Top Risk Drivers (feature importance):")
    print(importance.head(8).to_string())
