# ⚙️ Real-Time Predictive Maintenance using ESP32 & Machine Learning

A real-time predictive maintenance system that monitors machine vibration using an **ESP32** and **MPU6050** sensor, extracts vibration features, and detects abnormal machine behavior using an **Isolation Forest** anomaly detection model. The system provides a live **Streamlit dashboard** for monitoring machine health.

---

## 📌 Project Overview

Industrial machines often develop faults gradually. Small changes in vibration patterns can indicate problems long before complete failure occurs.

This project learns the vibration characteristics of a **healthy machine** and continuously compares incoming vibration data against that learned behavior.

Instead of training on faulty data, the model is trained **only on healthy operating conditions**. Any significant deviation from normal behavior is classified as an anomaly.

---

## Features

- Real-time vibration acquisition using ESP32
- MPU6050 3-axis accelerometer
- Signal preprocessing
- Statistical feature extraction
- FFT-based frequency analysis
- Isolation Forest anomaly detection
- Automatic threshold calibration
- Live Streamlit dashboard
- Real-time machine health monitoring
- Live vibration visualization
- Live anomaly score history

---

# Hardware Used

- ESP32 Development Board
- MPU6050 Accelerometer & Gyroscope
- Table Fan (Test Machine)
- USB Connection

---

# Software Stack

- Python 3
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Plotly
- PySerial
- Joblib

---

# Project Structure

```
PredictiveMaintenance/

├── acquisition/
│   ├── record.py
│   ├── serial_logger.py
│   └── live_plot.py
│
├── analysis/
│   ├── preprocessing.py
│   ├── statistics.py
│   ├── fft.py
│   ├── features.py
│   ├── build_dataset.py
│   ├── build_window_dataset.py
│   ├── check_dataset.py
│   └── eda.py
│
├── dashboard/
│   └── app.py
│
├── ml/
│   ├── train_model.py
│   ├── predict.py
│   ├── live_predict.py
│   ├── test_model.py
│   └── calibrate_threshold.py
│
├── firmware/
│
├── data/
│
└── README.md
```

---

# System Pipeline

```
MPU6050 Sensor
        │
        ▼
ESP32
        │
        ▼
Serial Communication
        │
        ▼
Rolling 2 Second Buffer
        │
        ▼
Preprocessing
        │
        ▼
Feature Extraction
        │
        ▼
Isolation Forest
        │
        ▼
Threshold Calibration
        │
        ▼
Live Dashboard
```

---

# Feature Extraction

The following statistical features are computed for each axis (X, Y, Z):

- Mean
- Standard Deviation
- Variance
- RMS
- Minimum
- Maximum
- Peak-to-Peak

Additionally, FFT is performed on the resultant vibration signal to obtain:

- Dominant Frequency
- Peak Magnitude

These features form the machine learning feature vector.

---

# Machine Learning Model

This project uses an **Isolation Forest**.

Why Isolation Forest?

- Requires only healthy data
- Suitable for anomaly detection
- Fast inference
- Robust to high-dimensional feature vectors

The model is trained only on healthy vibration recordings.

During deployment:

Healthy vibration
→ High anomaly score
→ NORMAL

Abnormal vibration
→ Low anomaly score
→ ANOMALY

---

# Dashboard Features

The Streamlit dashboard displays:

- Machine Health
- Current Anomaly Score
- Decision Threshold
- Live Vibration Plot
- Anomaly Score History
- Monitoring Statistics

The dashboard updates continuously in real time.

---

# Data Collection

Healthy vibration data was collected from a table fan.

Dataset characteristics:

- Sampling Rate ≈ 300 Hz
- Recording Duration = 10 seconds
- Window Size = 2 seconds
- Number of Healthy Recordings = 19
- Number of Training Windows = 95

---

# Running the Project

## 1. Clone Repository

```bash
git clone https://github.com/ABHIJEETRANJAN2303/PredictiveMaintenance.git

cd PredictiveMaintenance
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Connect ESP32

Upload the firmware to the ESP32 and connect it using USB.

Update the serial port in:

```
ml/predict.py
```

if necessary.

---

## 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Wait approximately 2 seconds for the rolling buffer to fill.

The dashboard will automatically begin monitoring the machine.

---

# Example Workflow

Machine Running Normally

```
Machine Health

🟢 NORMAL
```

Introduce vibration disturbance

```
Machine Health

🔴 ANOMALY
```

Remove disturbance

```
Machine Health

🟢 NORMAL
```

---

# Future Improvements

- Automatic serial port detection
- Fault type classification
- MQTT/IoT integration
- Cloud monitoring
- Historical database
- Email/SMS alerts
- Model retraining interface

---


# Author

**Abhijeet Ranjan**

GitHub:

https://github.com/ABHIJEETRANJAN2303/PredictiveMaintenance

---

# License

This project is released under the MIT License.