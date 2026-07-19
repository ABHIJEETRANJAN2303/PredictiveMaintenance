import matplotlib.pyplot as plt

from preprocessing import preprocess

# Load preprocessed data
df = preprocess(remove_mean=True)

# Show only first 500 samples
df_small = df.iloc[:500]

plt.figure(figsize=(14, 6))

plt.plot(df_small["timestamp"], df_small["ax"], label="Accel X")
plt.plot(df_small["timestamp"], df_small["ay"], label="Accel Y")
plt.plot(df_small["timestamp"], df_small["az"], label="Accel Z")

plt.title("Acceleration Signal (First 500 Samples)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()