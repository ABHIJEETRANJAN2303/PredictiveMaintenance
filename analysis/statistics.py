import numpy as np
from analysis.preprocessing import preprocess


def compute_statistics(values):
    """
    Compute statistical features for one signal.
    """

    values = np.array(values)

    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "variance": np.var(values),
        "rms": np.sqrt(np.mean(values ** 2)),
        "minimum": np.min(values),
        "maximum": np.max(values),
        "peak_to_peak": np.ptp(values),
    }


if __name__ == "__main__":

    df = preprocess()

    for axis in ["ax", "ay", "az"]:

        stats = compute_statistics(df[axis])

        print("\n" + "=" * 50)
        print(axis.upper())
        print("=" * 50)

        for key, value in stats.items():
            print(f"{key:15}: {value:.2f}")