from pathlib import Path

import pandas as pd

from analysis.features import extract_features
from analysis.preprocessing import preprocess

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "healthy"
OUTPUT_DIR = BASE_DIR / "ml"

OUTPUT_DIR.mkdir(exist_ok=True)





def main():

    all_features = []

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print(f"\nFound {len(csv_files)} recordings.\n")

    for csv_file in csv_files:

        print(f"Processing {csv_file.name}")

        df = pd.read_csv(csv_file)
        df = preprocess(df)

        features = extract_features(df)

        features["file"] = csv_file.name

        all_features.append(features)

    dataset = pd.DataFrame(all_features)

    columns = ["file"] + [c for c in dataset.columns if c != "file"]

    dataset = dataset[columns]

    output_file = OUTPUT_DIR / "healthy_features.csv"

    dataset.to_csv(output_file, index=False)

    print("\n====================================")
    print("Dataset Created Successfully!")
    print("====================================")
    print(dataset.head())
    print()
    print(f"Saved to:\n{output_file}")


if __name__ == "__main__":
    main()