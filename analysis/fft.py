import numpy as np
import matplotlib.pyplot as plt

from analysis.preprocessing import preprocess
from analysis.loader import load_latest_csv


def compute_fft(signal, sampling_frequency):

    signal = np.asarray(signal)

    N = len(signal)

    # Apply Hanning window
    window = np.hanning(N)
    signal = signal * window

    fft = np.fft.rfft(signal)

    frequency = np.fft.rfftfreq(N, d=1 / sampling_frequency)

    magnitude = np.abs(fft) / (N / 2)

    # Remove DC
    frequency = frequency[1:]
    magnitude = magnitude[1:]

    # Ignore frequencies below 5 Hz
    valid = frequency >= 5

    frequency = frequency[valid]
    magnitude = magnitude[valid]

    dominant_index = np.argmax(magnitude)

    return {
        "frequency": frequency,
        "magnitude": magnitude,
        "dominant_frequency": frequency[dominant_index],
        "peak_magnitude": magnitude[dominant_index],
    }


if __name__ == "__main__":

    df = load_latest_csv()

    df = preprocess(df)

    time = df["timestamp"].to_numpy()

    dt = np.mean(np.diff(time))

    fs = 1 / dt

    vibration = np.sqrt(
        df["ax"] ** 2 +
        df["ay"] ** 2 +
        df["az"] ** 2
    )

    result = compute_fft(vibration, fs)

    print(f"Sampling Frequency : {fs:.2f} Hz")
    print(f"Dominant Frequency : {result['dominant_frequency']:.2f} Hz")
    print(f"Peak Magnitude     : {result['peak_magnitude']:.4f}")

    plt.figure(figsize=(14, 6))

    plt.plot(result["frequency"], result["magnitude"])

    plt.scatter(
        result["dominant_frequency"],
        result["peak_magnitude"],
        color="red",
        s=80,
        label=f"{result['dominant_frequency']:.2f} Hz",
    )

    plt.title("FFT Spectrum (Resultant Acceleration)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()