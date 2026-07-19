import sys
import time
from pathlib import Path
from collections import deque

import joblib
import pandas as pd
import serial

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from analysis.preprocessing import preprocess
from analysis.features import extract_features

# -------------------------------------------------
# Configuration
# -------------------------------------------------

SERIAL_PORT = "/dev/cu.usbserial-110"
BAUD_RATE = 115200

WINDOW_SECONDS = 2
TARGET_SAMPLING_RATE = 300
BUFFER_SIZE = WINDOW_SECONDS * TARGET_SAMPLING_RATE

# -------------------------------------------------
# Load ML Objects
# -------------------------------------------------

model = joblib.load(BASE_DIR / "ml" / "isolation_forest.pkl")
scaler = joblib.load(BASE_DIR / "ml" / "scaler.pkl")
feature_columns = joblib.load(BASE_DIR / "ml" / "feature_columns.pkl")
threshold = joblib.load(BASE_DIR / "ml" / "threshold.pkl")

# -------------------------------------------------
# Serial Connection
# -------------------------------------------------

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

# -------------------------------------------------
# Rolling Buffer
# -------------------------------------------------

buffer = deque(maxlen=BUFFER_SIZE)
last_prediction_time = 0


def get_prediction():
    """
    Returns:
        {
            status,
            score,
            threshold,
            samples
        }

    Returns None until the rolling buffer is full.
    """

    global last_prediction_time

    while True:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        values = line.split(",")

        if len(values) != 3:
            continue

        try:
            ax = float(values[0])
            ay = float(values[1])
            az = float(values[2])
        except ValueError:
            continue

        buffer.append([
            time.time(),
            ax,
            ay,
            az
        ])

        if len(buffer) < BUFFER_SIZE:
            return None

        if time.time() - last_prediction_time < 1:
            return None

        last_prediction_time = time.time()

        df = pd.DataFrame(
            list(buffer),
            columns=[
                "timestamp",
                "ax",
                "ay",
                "az"
            ]
        )

        df["timestamp"] = (
            df["timestamp"]
            - df["timestamp"].iloc[0]
        )

        df = preprocess(df)

        feature_vector = extract_features(df)

        feature_df = pd.DataFrame([feature_vector])

        feature_df = feature_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        X = scaler.transform(feature_df)

        score = model.decision_function(X)[0]

        status = (
            "NORMAL"
            if score >= threshold
            else "ANOMALY"
        )

        return {
            "status": status,
            "score": score,
            "threshold": threshold,
            "samples": len(df),

            # For dashboard visualization
            "data": df.copy()
}