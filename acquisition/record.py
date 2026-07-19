import csv
import serial
import time
from pathlib import Path

PORT = "/dev/cu.usbserial-110"
BAUD = 115200


def get_next_filename(folder: Path):

    folder.mkdir(parents=True, exist_ok=True)

    existing = sorted(folder.glob("run_*.csv"))

    if not existing:
        return folder / "run_001.csv"

    last = int(existing[-1].stem.split("_")[1])

    return folder / f"run_{last+1:03}.csv"


def main():

    label = input("Enter label (healthy / imbalance / loose): ").strip().lower()

    duration = float(input("Recording duration (seconds): "))

    project_root = Path(__file__).resolve().parent.parent

    save_folder = project_root / "data" / label

    filename = get_next_filename(save_folder)

    print("\nConnecting to ESP32...")

    ser = serial.Serial(PORT, BAUD, timeout=1)

    time.sleep(2)

    print("Recording...\n")

    start = time.time()

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["timestamp", "ax", "ay", "az"])

        while time.time() - start < duration:

            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 3:
                continue

            timestamp = time.time() - start

            writer.writerow([timestamp] + parts)

    ser.close()

    print("\nRecording Complete!")
    print(f"Saved to:\n{filename}")


if __name__ == "__main__":
    main()