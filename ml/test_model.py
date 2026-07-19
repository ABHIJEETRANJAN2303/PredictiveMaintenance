from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "ml" / "healthy_features.csv"

MODEL_FILE = BASE_DIR / "ml" / "isolation_forest.pkl"

SCALER_FILE = BASE_DIR / "ml" / "scaler.pkl"


def main():

    print("\nLoading model...\n")

    model = joblib.load(MODEL_FILE)

    scaler = joblib.load(SCALER_FILE)

    df = pd.read_csv(FEATURE_FILE)

    files = df["file"]

    X = df.drop(columns=["file"])

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)

    score = model.decision_function(X_scaled)

    result = pd.DataFrame({
        "file": files,
        "prediction": prediction,
        "score": score
    })

    print(result)

    print("\n==============================")

    normal = (prediction == 1).sum()

    anomaly = (prediction == -1).sum()

    print(f"Normal  : {normal}")

    print(f"Anomaly : {anomaly}")

    print("==============================")

    print("\nLegend")

    print(" 1  -> Normal")

    print("-1 -> Anomaly")


if __name__ == "__main__":
    main()