import numpy as np
import pandas as pd

from analysis.preprocessing import preprocess
from analysis.statistics import compute_statistics
from analysis.fft import compute_fft
from analysis.loader import load_latest_csv


def extract_features(df):

    time = df["timestamp"].to_numpy()

    dt = np.mean(np.diff(time))

    sampling_frequency = 1 / dt

    features = {}

    # ------------------------------------
    # Statistical Features (Per Axis)
    # ------------------------------------

    for axis in ["ax", "ay", "az"]:

        stats = compute_statistics(df[axis])

        for key, value in stats.items():
            features[f"{axis}_{key}"] = value

    # ------------------------------------
    # Resultant Acceleration
    # ------------------------------------

    vibration = np.sqrt(
        df["ax"] ** 2 +
        df["ay"] ** 2 +
        df["az"] ** 2
    )

    # ------------------------------------
    # FFT Features
    # ------------------------------------

    fft = compute_fft(vibration, sampling_frequency)

    features["dominant_frequency"] = fft["dominant_frequency"]
    features["peak_magnitude"] = fft["peak_magnitude"]

    return features


if __name__ == "__main__":

    df = load_latest_csv()

    df = preprocess(df)

    feature_vector = extract_features(df)

    feature_table = pd.DataFrame([feature_vector])

    print(feature_table)