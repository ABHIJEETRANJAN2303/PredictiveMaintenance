from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "ml" / "healthy_features_windowed.csv"

MODEL_FILE = BASE_DIR / "ml" / "isolation_forest.pkl"
SCALER_FILE = BASE_DIR / "ml" / "scaler.pkl"
FEATURE_COLUMNS_FILE = BASE_DIR / "ml" / "feature_columns.pkl"

THRESHOLD_FILE = BASE_DIR / "ml" / "threshold.pkl"


def main():

    print("\nLoading healthy dataset...")

    df = pd.read_csv(FEATURE_FILE)

    X = df.drop(columns=["file", "window"])

    scaler = joblib.load(SCALER_FILE)
    model = joblib.load(MODEL_FILE)

    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)

    X = X[feature_columns]

    X_scaled = scaler.transform(X)

    scores = model.decision_function(X_scaled)

    mean = np.mean(scores)
    std = np.std(scores)

    threshold = mean - 3 * std

    print("\n======================================")
    print(f"Mean Score      : {mean:.6f}")
    print(f"Std Deviation   : {std:.6f}")
    print(f"Threshold       : {threshold:.6f}")
    print("======================================")

    joblib.dump(threshold, THRESHOLD_FILE)

    print(f"\nThreshold saved to:\n{THRESHOLD_FILE}")


if __name__ == "__main__":
    main()