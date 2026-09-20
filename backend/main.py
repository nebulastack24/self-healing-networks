from pathlib import Path
import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "network_metrics.csv"

MODEL_FILE = PROJECT_ROOT / "models" / "congestion_model.pkl"
ENCODER_FILE = PROJECT_ROOT / "models" / "label_encoder.pkl"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Self-Healing Campus Network API",
    description="Backend API for network monitoring, congestion prediction and adaptive QoS management.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD ML MODEL
# ============================================================

model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_network_data():

    df = pd.read_csv(DATA_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")

    return df


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Self-Healing Campus Network API is running",
        "status": "online"
    }


# ============================================================
# NETWORK STATUS
# ============================================================

@app.get("/api/status")
def get_status():

    df = load_network_data()

    latest = df.iloc[-1]

    return {
        "timestamp": str(latest["timestamp"]),
        "connected_devices": int(latest["connected_devices"]),
        "upload_mbps": float(latest["upload_mbps"]),
        "download_mbps": float(latest["download_mbps"]),
        "latency_ms": float(latest["latency_ms"]),
        "packet_loss_percent": float(latest["packet_loss_percent"]),
        "bandwidth_usage_percent": float(
            latest["bandwidth_usage_percent"]
        ),
        "throughput_mbps": float(latest["throughput_mbps"])
    }


# ============================================================
# NETWORK METRICS
# ============================================================

@app.get("/api/metrics")
def get_metrics():

    df = load_network_data()

    # Return latest 50 measurements

    recent = df.tail(50)

    recent = recent.copy()

    recent["timestamp"] = recent["timestamp"].astype(str)

    return recent.to_dict(orient="records")


# ============================================================
# ML CONGESTION PREDICTION
# ============================================================

@app.get("/api/prediction")
def get_prediction():

    df = load_network_data()

    latest = df.iloc[-1]

    timestamp = latest["timestamp"]

    input_data = pd.DataFrame([{
        "connected_devices": latest["connected_devices"],
        "upload_mbps": latest["upload_mbps"],
        "download_mbps": latest["download_mbps"],
        "latency_ms": latest["latency_ms"],
        "packet_loss_percent": latest["packet_loss_percent"],
        "bandwidth_usage_percent": latest["bandwidth_usage_percent"],
        "throughput_mbps": latest["throughput_mbps"],
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "day_of_week": timestamp.dayofweek
    }])

    prediction = model.predict(input_data)[0]

    congestion_level = label_encoder.inverse_transform(
        [prediction]
    )[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_result = {}

    for label, probability in zip(
        label_encoder.classes_,
        probabilities
    ):

        probability_result[label] = round(
            float(probability) * 100,
            2
        )

    return {
        "timestamp": str(timestamp),
        "congestion_level": congestion_level,
        "probabilities": probability_result
    }


# ============================================================
# RECOMMENDATION
# ============================================================

@app.get("/api/recommendation")
def get_recommendation():

    df = load_network_data()

    latest = df.iloc[-1]

    timestamp = latest["timestamp"]

    input_data = pd.DataFrame([{
        "connected_devices": latest["connected_devices"],
        "upload_mbps": latest["upload_mbps"],
        "download_mbps": latest["download_mbps"],
        "latency_ms": latest["latency_ms"],
        "packet_loss_percent": latest["packet_loss_percent"],
        "bandwidth_usage_percent": latest["bandwidth_usage_percent"],
        "throughput_mbps": latest["throughput_mbps"],
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "day_of_week": timestamp.dayofweek
    }])

    prediction = model.predict(input_data)[0]

    congestion_level = label_encoder.inverse_transform(
        [prediction]
    )[0]


    # --------------------------------------------
    # Recommendation rules
    # --------------------------------------------

    if congestion_level == "Normal":

        severity = "LOW"

        recommendation = (
            "Maintain current bandwidth allocation "
            "and continue normal monitoring."
        )

        actions = [
            "Maintain current bandwidth allocation",
            "Continue regular network monitoring",
            "No immediate traffic restriction required"
        ]


    elif congestion_level == "Moderate":

        severity = "MEDIUM"

        recommendation = (
            "Prioritize important network traffic "
            "and increase monitoring frequency."
        )

        actions = [
            "Prioritize academic and critical traffic",
            "Monitor network metrics more frequently",
            "Limit unnecessary background traffic",
            "Prepare for possible bandwidth reallocation"
        ]


    elif congestion_level == "High":

        severity = "HIGH"

        recommendation = (
            "Initiate congestion mitigation by "
            "prioritizing critical traffic and "
            "reducing background traffic."
        )

        actions = [
            "Prioritize critical academic traffic",
            "Prioritize real-time communication traffic",
            "Reduce or defer large background downloads",
            "Recommend bandwidth reallocation",
            "Trigger immediate network re-monitoring"
        ]


    else:

        severity = "UNKNOWN"

        recommendation = "Use default network management policy."

        actions = [
            "Continue monitoring"
        ]


    return {
        "timestamp": str(timestamp),
        "congestion_level": congestion_level,
        "severity": severity,
        "recommendation": recommendation,
        "actions": actions
    }


# ============================================================
# ADAPTIVE QoS POLICY
# ============================================================

@app.get("/api/adaptive-policy")
def get_adaptive_policy():

    df = load_network_data()

    latest = df.iloc[-1]

    timestamp = latest["timestamp"]

    input_data = pd.DataFrame([{
        "connected_devices": latest["connected_devices"],
        "upload_mbps": latest["upload_mbps"],
        "download_mbps": latest["download_mbps"],
        "latency_ms": latest["latency_ms"],
        "packet_loss_percent": latest["packet_loss_percent"],
        "bandwidth_usage_percent": latest["bandwidth_usage_percent"],
        "throughput_mbps": latest["throughput_mbps"],
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "day_of_week": timestamp.dayofweek
    }])

    prediction = model.predict(input_data)[0]

    congestion_level = label_encoder.inverse_transform(
        [prediction]
    )[0]


    # --------------------------------------------
    # Adaptive policy
    # --------------------------------------------

    if congestion_level == "Normal":

        policy = "NORMAL"

        allocation = {
            "critical_academic": 40,
            "general_web": 40,
            "background": 20
        }

        priority = {
            "critical_academic": "HIGH",
            "general_web": "MEDIUM",
            "background": "LOW"
        }


    elif congestion_level == "Moderate":

        policy = "MODERATE_CONGESTION"

        allocation = {
            "critical_academic": 50,
            "general_web": 35,
            "background": 15
        }

        priority = {
            "critical_academic": "VERY HIGH",
            "general_web": "MEDIUM",
            "background": "LOW"
        }


    elif congestion_level == "High":

        policy = "HIGH_CONGESTION"

        allocation = {
            "critical_academic": 60,
            "general_web": 30,
            "background": 10
        }

        priority = {
            "critical_academic": "VERY HIGH",
            "general_web": "MEDIUM",
            "background": "VERY LOW"
        }


    else:

        policy = "DEFAULT"

        allocation = {
            "critical_academic": 40,
            "general_web": 40,
            "background": 20
        }

        priority = {
            "critical_academic": "HIGH",
            "general_web": "MEDIUM",
            "background": "LOW"
        }


    return {
        "timestamp": str(timestamp),
        "congestion_level": congestion_level,
        "policy": policy,
        "bandwidth_allocation_percent": allocation,
        "qos_priority": priority
    }