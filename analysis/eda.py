from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "ml" / "healthy_features.csv"


def main():

    df = pd.read_csv(FEATURE_FILE)

    print("\n==============================")
    print("Dataset Shape")
    print("==============================")
    print(df.shape)

    print("\n==============================")
    print("Missing Values")
    print("==============================")
    print(df.isnull().sum())

    print("\n==============================")
    print("Summary Statistics")
    print("==============================")
    print(df.describe())

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:

        plt.figure(figsize=(6,4))

        plt.hist(df[column], bins=10)

        plt.title(column)

        plt.xlabel(column)

        plt.ylabel("Count")

        plt.grid(True)

        plt.tight_layout()

    plt.show()

    correlation = df[numeric_columns].corr()

    plt.figure(figsize=(14,10))

    plt.imshow(correlation)

    plt.colorbar()

    plt.xticks(
        range(len(numeric_columns)),
        numeric_columns,
        rotation=90
    )

    plt.yticks(
        range(len(numeric_columns)),
        numeric_columns
    )

    plt.title("Feature Correlation Matrix")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()