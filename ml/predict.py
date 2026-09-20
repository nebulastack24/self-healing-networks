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


# ---------------------------------------------------------
# FEATURES REQUIRED BY MODEL
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
# CHECK FILES
# ---------------------------------------------------------

print("=" * 70)
print("STEP 8 - REAL-TIME CONGESTION PREDICTION")
print("=" * 70)


if not MODEL_FILE.exists():

    print("\nERROR: Trained model not found.")

    print(
        f"Expected file:\n{MODEL_FILE}"
    )

    print(
        "\nRun train_model.py and evaluate_models.py first."
    )

    raise SystemExit


if not ENCODER_FILE.exists():

    print("\nERROR: Label encoder not found.")

    raise SystemExit


if not DATA_FILE.exists():

    print("\nERROR: Network metrics file not found.")

    print(
        f"Expected file:\n{DATA_FILE}"
    )

    raise SystemExit


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("\nLoading trained Random Forest model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")


# ---------------------------------------------------------
# LOAD LABEL ENCODER
# ---------------------------------------------------------

label_encoder = joblib.load(
    ENCODER_FILE
)


# ---------------------------------------------------------
# LOAD NETWORK DATA
# ---------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print(
    f"\nNetwork data loaded: {len(df)} records"
)


# ---------------------------------------------------------
# CHECK REQUIRED RAW COLUMNS
# ---------------------------------------------------------

raw_columns = [
    "timestamp",
    "connected_devices",
    "upload_mbps",
    "download_mbps",
    "latency_ms",
    "packet_loss_percent",
    "bandwidth_usage_percent",
    "throughput_mbps"
]


missing_columns = [
    column
    for column in raw_columns
    if column not in df.columns
]


if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise SystemExit


# ---------------------------------------------------------
# CONVERT TIMESTAMP
# ---------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# Remove invalid timestamps

df = df.dropna(
    subset=["timestamp"]
)


# ---------------------------------------------------------
# CREATE TIME FEATURES
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
# GET LATEST NETWORK MEASUREMENT
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
# DISPLAY INPUT
# ---------------------------------------------------------

print("\nLatest Network Metrics")
print("-" * 70)

print(
    f"Timestamp              : {latest['timestamp']}"
)

print(
    f"Connected Devices      : {latest['connected_devices']}"
)

print(
    f"Upload                 : {latest['upload_mbps']:.2f} Mbps"
)

print(
    f"Download               : {latest['download_mbps']:.2f} Mbps"
)

print(
    f"Latency                : {latest['latency_ms']:.2f} ms"
)

print(
    f"Packet Loss            : {latest['packet_loss_percent']:.2f}%"
)

print(
    f"Bandwidth Usage        : {latest['bandwidth_usage_percent']:.2f}%"
)

print(
    f"Throughput             : {latest['throughput_mbps']:.2f} Mbps"
)


# ---------------------------------------------------------
# PREDICT CONGESTION
# ---------------------------------------------------------

prediction_encoded = model.predict(
    input_data
)[0]


prediction = label_encoder.inverse_transform(
    [prediction_encoded]
)[0]


# ---------------------------------------------------------
# PREDICTION PROBABILITY
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
# DISPLAY PREDICTION
# ---------------------------------------------------------

print("\n" + "=" * 70)

print("CONGESTION PREDICTION")

print("=" * 70)

print(
    f"\nPredicted Congestion Level: {prediction}"
)


print("\nPrediction Probabilities:")

for class_name, probability in class_probabilities.items():

    print(
        f"{class_name:10s}: "
        f"{probability * 100:.2f}%"
    )


print("\n" + "=" * 70)
print("STEP 8 COMPLETED")
print("=" * 70)