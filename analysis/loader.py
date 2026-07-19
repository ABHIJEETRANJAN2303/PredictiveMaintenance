from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_latest_csv():
    """
    Load the newest CSV file from any dataset folder.
    """

    csv_files = list(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found inside data/")

    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)

    print(f"\nLoaded: {latest_file.relative_to(DATA_DIR)}\n")

    return pd.read_csv(latest_file)


if __name__ == "__main__":

    df = load_latest_csv()

    print(df.head())