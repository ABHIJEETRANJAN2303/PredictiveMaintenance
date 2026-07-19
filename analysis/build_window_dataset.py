from pathlib import Path

import pandas as pd

from analysis.features import extract_features
from analysis.preprocessing import preprocess

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "healthy"

OUTPUT_DIR = BASE_DIR / "ml"

OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_SECONDS = 2


def main():

    dataset = []

    files = sorted(DATA_DIR.glob("*.csv"))

    print(f"\nFound {len(files)} recordings.\n")

    for file in files:

        print(f"Processing {file.name}")

        df = pd.read_csv(file)

        total_time = df["timestamp"].iloc[-1]

        start = 0

        window_number = 1

        while start + WINDOW_SECONDS <= total_time:

            window = df[
                (df["timestamp"] >= start)
                &
                (df["timestamp"] < start + WINDOW_SECONDS)
            ].copy()

            if len(window) < 100:
                start += WINDOW_SECONDS
                continue

            window = preprocess(window)

            features = extract_features(window)

            features["file"] = file.name
            features["window"] = window_number

            dataset.append(features)

            start += WINDOW_SECONDS
            window_number += 1

    dataset = pd.DataFrame(dataset)

    cols = ["file", "window"] + [
        c for c in dataset.columns
        if c not in ["file", "window"]
    ]

    dataset = dataset[cols]

    output = OUTPUT_DIR / "healthy_features_windowed.csv"

    dataset.to_csv(output, index=False)

    print("\n===============================")
    print("Dataset Created Successfully")
    print("===============================")

    print(dataset.head())

    print(f"\nSaved to:\n{output}")


if __name__ == "__main__":
    main()