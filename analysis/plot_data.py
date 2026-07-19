from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Automatically pick the newest CSV
latest_file = max(DATA_DIR.glob("*.csv"), key=lambda f: f.stat().st_mtime)

print("Reading:", latest_file)

df = pd.read_csv(latest_file)

print(df.head())

plt.figure(figsize=(14,6))

plt.plot(df["timestamp"], df["ax"], label="Accel X")
plt.plot(df["timestamp"], df["ay"], label="Accel Y")
plt.plot(df["timestamp"], df["az"], label="Accel Z")

plt.title("Raw Accelerometer Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Raw Sensor Value")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()