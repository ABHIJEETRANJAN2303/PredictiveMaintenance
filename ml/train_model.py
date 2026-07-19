from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "ml" / "healthy_features_windowed.csv"

MODEL_FILE = BASE_DIR / "ml" / "isolation_forest.pkl"

SCALER_FILE = BASE_DIR / "ml" / "scaler.pkl"

FEATURE_COLUMNS_FILE = BASE_DIR / "ml" / "feature_columns.pkl"


def main():

    print("\nLoading feature dataset...\n")

    df = pd.read_csv(FEATURE_FILE)

    X = df.drop(columns=["file", "window"])

    print(f"Training Samples : {len(X)}")
    print(f"Feature Count    : {X.shape[1]}")

    # ----------------------------
    # Feature Scaling
    # ----------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ----------------------------
    # Isolation Forest
    # ----------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        bootstrap=False,
        max_samples="auto",
        n_jobs=-1
    )

    model.fit(X_scaled)

    # ----------------------------
    # Save
    # ----------------------------

    joblib.dump(model, MODEL_FILE)

    joblib.dump(scaler, SCALER_FILE)

    joblib.dump(list(X.columns), FEATURE_COLUMNS_FILE)

    print("\n==============================")
    print("Training Complete")
    print("==============================")

    print(f"\nModel Saved           : {MODEL_FILE}")
    print(f"Scaler Saved          : {SCALER_FILE}")
    print(f"Feature Columns Saved : {FEATURE_COLUMNS_FILE}")


if __name__ == "__main__":
    main()