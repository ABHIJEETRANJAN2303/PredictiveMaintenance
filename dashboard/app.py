import sys
import time
from pathlib import Path
from collections import deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from ml.predict import get_prediction

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("⚙️ System Information")

st.sidebar.markdown("### Sensor")
st.sidebar.write("MPU6050")

st.sidebar.markdown("### Sampling Rate")
st.sidebar.write("300 Hz")

st.sidebar.markdown("### Window Size")
st.sidebar.write("2 Seconds")

st.sidebar.markdown("### Features")
st.sidebar.write("23")

st.sidebar.markdown("### Model")
st.sidebar.write("Isolation Forest")

st.sidebar.markdown("### Threshold")
st.sidebar.write("-0.090320")

st.sidebar.success("Monitoring")

# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("⚙️ Predictive Maintenance Dashboard")

st.caption(
    "Real-Time Predictive Maintenance using ESP32 + MPU6050 + Isolation Forest"
)

placeholder = st.empty()

score_history = deque(maxlen=100)

start_time = time.time()

prediction_count = 0

fault_count = 0

# --------------------------------------------------
# Live Dashboard
# --------------------------------------------------

while True:

    prediction = get_prediction()

    if prediction is None:
        continue

    prediction_count += 1

    if prediction["status"] == "ANOMALY":
        fault_count += 1

    score_history.append(prediction["score"])

    df = prediction["data"]

    uptime = int(time.time() - start_time)

    hours = uptime // 3600
    minutes = (uptime % 3600) // 60
    seconds = uptime % 60

    uptime_text = f"{hours:02}:{minutes:02}:{seconds:02}"

    with placeholder.container():

        st.markdown("---")

        c1, c2, c3 = st.columns([1.4,1,1])

        with c1:

            st.subheader("Machine Health")

            if prediction["status"] == "NORMAL":
                st.success("🟢 NORMAL")
            else:
                st.error("🔴 ANOMALY")

        with c2:

            st.metric(
                "Anomaly Score",
                f"{prediction['score']:.6f}"
            )

        with c3:

            st.metric(
                "Threshold",
                f"{prediction['threshold']:.6f}"
            )

        st.markdown("---")

        a, b, c = st.columns(3)

        a.metric(
            "Monitoring Time",
            uptime_text
        )

        b.metric(
            "Predictions",
            prediction_count
        )

        c.metric(
            "Faults Detected",
            fault_count
        )

        st.markdown("---")

        st.subheader("Live Vibration")

        vibration = df[
            ["ax", "ay", "az"]
        ].reset_index(drop=True)

        st.line_chart(
            vibration,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("Anomaly Score History")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=list(score_history),
                mode="lines",
                line=dict(width=3),
                name="Score"
            )
        )

        fig.add_hline(
            y=prediction["threshold"],
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold"
        )

        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            showlegend=False,
            xaxis_title="Prediction",
            yaxis_title="Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    time.sleep(0.1)