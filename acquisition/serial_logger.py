import serial
import csv
import time
from pathlib import Path

PORT = "/dev/cu.usbserial-110"
BAUD = 115200

# Find the project root regardless of where the script is run from
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

filename = DATA_DIR / f"healthy_{int(time.time())}.csv"

print(f"Saving to: {filename}")

ser = serial.Serial(PORT, BAUD, timeout=1)

with open(filename, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["timestamp", "ax", "ay", "az"])

    start = time.time()

    try:
        while True:

            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 3:
                continue

            timestamp = time.time() - start

            writer.writerow([timestamp, *parts])

            print(f"{timestamp:.3f}s -> {parts}")

    except KeyboardInterrupt:
        print("\nRecording stopped.")

ser.close()

print(f"\nCSV saved at:\n{filename}")