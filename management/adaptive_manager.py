from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "network_metrics.csv"

MODEL_FILE = PROJECT_ROOT / "models" / "congestion_model.pkl"
ENCODER_FILE = PROJECT_ROOT / "models" / "label_encoder.pkl"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "adaptive_management"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = OUTPUT_DIR / "adaptive_history.csv"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)


# ============================================================
# LOAD NETWORK DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by timestamp
df = df.sort_values("timestamp")

# Get latest network measurement
latest = df.iloc[-1]


# ============================================================
# CREATE TIME FEATURES
# ============================================================

hour = latest["timestamp"].hour
minute = latest["timestamp"].minute
day_of_week = latest["timestamp"].dayofweek


# ============================================================
# MODEL FEATURES
# ============================================================

features = [
    "connected_devices",
    "upload_mbps",
    "download_mbps",
    "latency_ms",
    "packet_loss_percent",
    "bandwidth_usage_percent",
    "throughput_mbps",
]

input_data = pd.DataFrame([[
    latest["connected_devices"],
    latest["upload_mbps"],
    latest["download_mbps"],
    latest["latency_ms"],
    latest["packet_loss_percent"],
    latest["bandwidth_usage_percent"],
    latest["throughput_mbps"],
]], columns=features)


# Add time features
input_data["hour"] = hour
input_data["minute"] = minute
input_data["day_of_week"] = day_of_week


# ============================================================
# PREDICT CONGESTION
# ============================================================

prediction = model.predict(input_data)[0]

congestion_level = label_encoder.inverse_transform([prediction])[0]


# ============================================================
# ADAPTIVE BANDWIDTH / QoS POLICY
# ============================================================

if congestion_level == "Normal":

    policy = "NORMAL"

    critical_bandwidth = 40
    general_bandwidth = 40
    background_bandwidth = 20

    critical_priority = "HIGH"
    general_priority = "MEDIUM"
    background_priority = "LOW"

    action = "Maintain current bandwidth allocation."


elif congestion_level == "Moderate":

    policy = "MODERATE_CONGESTION"

    critical_bandwidth = 50
    general_bandwidth = 35
    background_bandwidth = 15

    critical_priority = "VERY HIGH"
    general_priority = "MEDIUM"
    background_priority = "LOW"

    action = "Increase priority for critical traffic and reduce background traffic."


elif congestion_level == "High":

    policy = "HIGH_CONGESTION"

    critical_bandwidth = 60
    general_bandwidth = 30
    background_bandwidth = 10

    critical_priority = "VERY HIGH"
    general_priority = "MEDIUM"
    background_priority = "VERY LOW"

    action = "Strongly prioritize critical traffic and restrict background traffic."


else:

    policy = "UNKNOWN"

    critical_bandwidth = 40
    general_bandwidth = 40
    background_bandwidth = 20

    critical_priority = "HIGH"
    general_priority = "MEDIUM"
    background_priority = "LOW"

    action = "Use default bandwidth allocation."


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("ADAPTIVE BANDWIDTH & QoS MANAGEMENT")
print("=" * 60)

print("\nLatest Network Status")
print("-" * 60)

print(f"Timestamp             : {latest['timestamp']}")
print(f"Connected Devices     : {latest['connected_devices']}")
print(f"Upload                : {latest['upload_mbps']:.2f} Mbps")
print(f"Download              : {latest['download_mbps']:.2f} Mbps")
print(f"Latency               : {latest['latency_ms']:.2f} ms")
print(f"Packet Loss           : {latest['packet_loss_percent']:.2f}%")
print(f"Bandwidth Usage       : {latest['bandwidth_usage_percent']:.2f}%")
print(f"Throughput            : {latest['throughput_mbps']:.2f} Mbps")

print("\nML Prediction")
print("-" * 60)
print(f"Congestion Level      : {congestion_level}")

print("\nAdaptive Policy")
print("-" * 60)

print(f"Policy                : {policy}")
print(f"Action                : {action}")

print("\nBandwidth Allocation")
print("-" * 60)

print(f"Critical / Academic   : {critical_bandwidth}%")
print(f"General Web           : {general_bandwidth}%")
print(f"Background Traffic    : {background_bandwidth}%")

print("\nQoS Priority")
print("-" * 60)

print(f"Critical / Academic   : {critical_priority}")
print(f"General Web           : {general_priority}")
print(f"Background Traffic    : {background_priority}")


# ============================================================
# SAVE ADAPTATION HISTORY
# ============================================================

result = {
    "timestamp": datetime.now(),
    "network_timestamp": latest["timestamp"],
    "congestion_level": congestion_level,
    "policy": policy,
    "critical_bandwidth_percent": critical_bandwidth,
    "general_bandwidth_percent": general_bandwidth,
    "background_bandwidth_percent": background_bandwidth,
    "critical_priority": critical_priority,
    "general_priority": general_priority,
    "background_priority": background_priority,
    "action": action
}

result_df = pd.DataFrame([result])

if HISTORY_FILE.exists():

    result_df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=False,
        index=False
    )

else:

    result_df.to_csv(
        HISTORY_FILE,
        index=False
    )


print("\n" + "=" * 60)
print("Adaptive management decision saved.")
print(f"File: {HISTORY_FILE}")
print("=" * 60)