from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "data" / "healthy"


def analyze_file(file):

    df = pd.read_csv(file)

    samples = len(df)

    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]

    fs = samples / duration if duration > 0 else 0

    missing = df.isna().sum().sum()

    rms = np.sqrt(
        np.mean(
            df["ax"]**2 +
            df["ay"]**2 +
            df["az"]**2
        )
    )

    return {
        "file": file.name,
        "samples": samples,
        "duration": duration,
        "sampling_rate": fs,
        "missing": missing,
        "rms": rms,
    }


def main():

    files = sorted(DATASET.glob("*.csv"))

    print("=" * 90)

    print(f"Total recordings : {len(files)}")

    print("=" * 90)

    for file in files:

        result = analyze_file(file)

        print(
            f"{result['file']:12s}"
            f" Samples={result['samples']:5d}"
            f" Duration={result['duration']:6.2f}s"
            f" Fs={result['sampling_rate']:7.2f}Hz"
            f" Missing={result['missing']:2d}"
            f" RMS={result['rms']:.4f} g"
        )


if __name__ == "__main__":
    main()