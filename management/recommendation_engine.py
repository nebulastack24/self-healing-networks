from pathlib import Path

import pandas as pd
import joblib


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "congestion_model.pkl"
)

ENCODER_FILE = (
    PROJECT_ROOT
    / "models"
    / "label_encoder.pkl"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "network_metrics.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "recommendations"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "connected_devices",
    "upload_mbps",
    "download_mbps",
    "latency_ms",
    "packet_loss_percent",
    "bandwidth_usage_percent",
    "throughput_mbps",
    "hour",
    "minute",
    "day_of_week"
]


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

print("=" * 70)
print("STEP 9 - NETWORK RECOMMENDATION ENGINE")
print("=" * 70)


# ---------------------------------------------------------
# CHECK FILES
# ---------------------------------------------------------

if not MODEL_FILE.exists():

    print("\nERROR: congestion_model.pkl not found.")

    print(
        f"Expected:\n{MODEL_FILE}"
    )

    raise SystemExit


if not ENCODER_FILE.exists():

    print("\nERROR: label_encoder.pkl not found.")

    raise SystemExit


if not DATA_FILE.exists():

    print("\nERROR: network_metrics.csv not found.")

    raise SystemExit


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(
    MODEL_FILE
)

label_encoder = joblib.load(
    ENCODER_FILE
)

print("Model loaded successfully.")


# ---------------------------------------------------------
# LOAD NETWORK DATA
# ---------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Network records loaded: {len(df)}"
)


# ---------------------------------------------------------
# TIMESTAMP
# ---------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp"]
)


# ---------------------------------------------------------
# TIME FEATURES
# ---------------------------------------------------------

df["hour"] = (
    df["timestamp"].dt.hour
)

df["minute"] = (
    df["timestamp"].dt.minute
)

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)


# ---------------------------------------------------------
# LATEST NETWORK MEASUREMENT
# ---------------------------------------------------------

latest = df.iloc[-1]


# ---------------------------------------------------------
# CREATE MODEL INPUT
# ---------------------------------------------------------

input_data = pd.DataFrame(
    [[
        latest["connected_devices"],
        latest["upload_mbps"],
        latest["download_mbps"],
        latest["latency_ms"],
        latest["packet_loss_percent"],
        latest["bandwidth_usage_percent"],
        latest["throughput_mbps"],
        latest["hour"],
        latest["minute"],
        latest["day_of_week"]
    ]],
    columns=FEATURE_COLUMNS
)


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

prediction_encoded = model.predict(
    input_data
)[0]


prediction = label_encoder.inverse_transform(
    [prediction_encoded]
)[0]


# ---------------------------------------------------------
# PREDICTION PROBABILITIES
# ---------------------------------------------------------

probabilities = model.predict_proba(
    input_data
)[0]


class_probabilities = dict(
    zip(
        label_encoder.classes_,
        probabilities
    )
)


# ---------------------------------------------------------
# NETWORK VALUES
# ---------------------------------------------------------

latency = float(
    latest["latency_ms"]
)

packet_loss = float(
    latest["packet_loss_percent"]
)

bandwidth_usage = float(
    latest["bandwidth_usage_percent"]
)

connected_devices = int(
    latest["connected_devices"]
)

throughput = float(
    latest["throughput_mbps"]
)


# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------

if prediction == "Normal":

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


elif prediction == "Moderate":

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


elif prediction == "High":

    severity = "HIGH"

    recommendation = (
        "Initiate congestion mitigation by prioritizing "
        "critical traffic and reducing background traffic."
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

    recommendation = (
        "Unknown congestion state. Continue monitoring "
        "without automatic network changes."
    )

    actions = [
        "Continue monitoring",
        "Do not apply automatic network changes"
    ]


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT NETWORK STATUS")
print("=" * 70)

print(
    f"\nTimestamp           : {latest['timestamp']}"
)

print(
    f"Connected Devices   : {connected_devices}"
)

print(
    f"Latency             : {latency:.2f} ms"
)

print(
    f"Packet Loss         : {packet_loss:.2f}%"
)

print(
    f"Bandwidth Usage     : {bandwidth_usage:.2f}%"
)

print(
    f"Throughput          : {throughput:.2f} Mbps"
)


print("\n" + "=" * 70)
print("ML PREDICTION")
print("=" * 70)

print(
    f"\nCongestion Level: {prediction}"
)

print(
    f"Severity        : {severity}"
)


print("\nPrediction Probabilities:")

for class_name, probability in class_probabilities.items():

    print(
        f"  {class_name:10s}: "
        f"{probability * 100:.2f}%"
    )


print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print(
    f"\n{recommendation}"
)


print("\nRecommended Actions:")

for number, action in enumerate(
    actions,
    start=1
):

    print(
        f"  {number}. {action}"
    )


# ---------------------------------------------------------
# SAVE RECOMMENDATION
# ---------------------------------------------------------

recommendation_record = pd.DataFrame([
    {
        "timestamp":
            latest["timestamp"],

        "connected_devices":
            connected_devices,

        "latency_ms":
            latency,

        "packet_loss_percent":
            packet_loss,

        "bandwidth_usage_percent":
            bandwidth_usage,

        "throughput_mbps":
            throughput,

        "congestion_level":
            prediction,

        "severity":
            severity,

        "recommendation":
            recommendation,

        "recommended_actions":
            " | ".join(actions)
    }
])


output_file = (
    RESULT_DIR
    / "recommendation_history.csv"
)


if output_file.exists():

    recommendation_record.to_csv(
        output_file,
        mode="a",
        header=False,
        index=False
    )

else:

    recommendation_record.to_csv(
        output_file,
        index=False
    )


print("\nRecommendation saved to:")

print(output_file)


print("\n" + "=" * 70)
print("STEP 9 COMPLETED")
print("=" * 70)