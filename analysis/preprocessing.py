import pandas as pd

from analysis.loader import load_latest_csv


def preprocess(df, remove_mean=True):
    """
    Preprocess any dataframe containing:
    timestamp, ax, ay, az
    """

    processed = df.copy()

    if remove_mean:

        for axis in ["ax", "ay", "az"]:
            processed[axis] = processed[axis] - processed[axis].mean()

    return processed


if __name__ == "__main__":

    df = load_latest_csv()

    processed = preprocess(df)

    print(processed.head())