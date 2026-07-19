import sys
import time
from pathlib import Path
from collections import deque

import joblib
import pandas as pd
import serial

# -------------------------------------------------
# Allow importing from project root
# -------------------------------------------------

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

MODEL_PATH = BASE_DIR / "ml" / "isolation_forest.pkl"
SCALER_PATH = BASE_DIR / "ml" / "scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "ml" / "feature_columns.pkl"
THRESHOLD_PATH = BASE_DIR / "ml" / "threshold.pkl"

# -------------------------------------------------
# Load ML Objects
# -------------------------------------------------

print("\nLoading ML model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
threshold = joblib.load(THRESHOLD_PATH)

print("Model Loaded Successfully.")
print(f"Healthy Threshold : {threshold:.6f}")

# -------------------------------------------------
# Connect ESP32
# -------------------------------------------------

print(f"\nConnecting to {SERIAL_PORT}...")

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("ESP32 Connected.")
print("\nWaiting for 2-second buffer to fill...\n")

# -------------------------------------------------
# Rolling Buffer
# -------------------------------------------------

buffer = deque(maxlen=BUFFER_SIZE)

last_prediction_time = time.time()

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

while True:

    try:

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

        timestamp = time.time()

        buffer.append([
            timestamp,
            ax,
            ay,
            az
        ])

        # Wait until the rolling buffer is full
        if len(buffer) < BUFFER_SIZE:
            continue

        # Predict once every second
        if time.time() - last_prediction_time < 1:
            continue

        last_prediction_time = time.time()

        # -----------------------------------------
        # Build DataFrame
        # -----------------------------------------

        df = pd.DataFrame(
            list(buffer),
            columns=[
                "timestamp",
                "ax",
                "ay",
                "az"
            ]
        )

        # Convert timestamps to relative time
        df["timestamp"] = (
            df["timestamp"]
            - df["timestamp"].iloc[0]
        )

        # -----------------------------------------
        # Preprocess
        # -----------------------------------------

        df = preprocess(df)

        # -----------------------------------------
        # Feature Extraction
        # -----------------------------------------

        feature_vector = extract_features(df)

        feature_df = pd.DataFrame([feature_vector])

        # Ensure same feature order used during training
        feature_df = feature_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # -----------------------------------------
        # Scale Features
        # -----------------------------------------

        X = scaler.transform(feature_df)

        # -----------------------------------------
        # Compute Anomaly Score
        # -----------------------------------------

        anomaly_score = model.decision_function(X)[0]

        status = (
            "🟢 NORMAL"
            if anomaly_score >= threshold
            else "🔴 ANOMALY"
        )

        # -----------------------------------------
        # Console Output
        # -----------------------------------------

        print("\n================================================")
        print(f"Time            : {time.strftime('%H:%M:%S')}")
        print(f"Status          : {status}")
        print(f"Anomaly Score   : {anomaly_score:.6f}")
        print(f"Threshold       : {threshold:.6f}")
        print(f"Samples         : {len(df)}")
        print("================================================")

    except KeyboardInterrupt:

        print("\nStopping Live Prediction...")

        ser.close()

        break

    except Exception as e:

        print(f"\nError : {e}")