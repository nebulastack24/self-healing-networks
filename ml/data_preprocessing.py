from pathlib import Path

import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "network_metrics.csv"
)

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed_network_metrics.csv"
)

ML_FILE = (
    PROJECT_ROOT
    / "data"
    / "ml_dataset.csv"
)


# =========================================================
# START
# =========================================================

print("=" * 70)
print("STEP 3 - DATA CLEANING AND FEATURE ENGINEERING")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal rows    : {len(df)}")
print(f"Original columns : {len(df.columns)}")


# =========================================================
# REMOVE DUPLICATES
# =========================================================

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows found: {duplicate_count}")

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        f"Removed {duplicate_count} duplicate rows."
    )

else:

    print("No duplicates to remove.")


# =========================================================
# TIMESTAMP CONVERSION
# =========================================================

print("\nProcessing timestamp...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


invalid_timestamps = df["timestamp"].isna().sum()

print(
    f"Invalid timestamps: {invalid_timestamps}"
)


if invalid_timestamps > 0:

    df = df.dropna(
        subset=["timestamp"]
    )


# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\nMissing values:")

print(
    df.isnull().sum()
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("\nCreating time-based features...")


df["hour"] = (
    df["timestamp"].dt.hour
)


df["minute"] = (
    df["timestamp"].dt.minute
)


df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)


# =========================================================
# SORT BY TIMESTAMP
# =========================================================

df = df.sort_values(
    "timestamp"
)


# =========================================================
# SAVE PROCESSED DATA
# =========================================================

df.to_csv(
    PROCESSED_FILE,
    index=False
)


print(
    f"\nProcessed dataset saved to:"
)

print(
    PROCESSED_FILE
)


# =========================================================
# CREATE ML DATASET
# =========================================================

features = [
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


target = "congestion_level"


# Check that all required columns exist

missing_features = [
    column
    for column in features
    if column not in df.columns
]


if missing_features:

    print("\nERROR: Missing features:")

    for column in missing_features:
        print(
            " -",
            column
        )

    raise SystemExit


if target not in df.columns:

    print(
        "\nERROR: congestion_level is missing."
    )

    raise SystemExit


# =========================================================
# CREATE FINAL ML DATASET
# =========================================================

ml_df = df[
    features + [target]
].copy()


ml_df.to_csv(
    ML_FILE,
    index=False
)


# =========================================================
# DISPLAY RESULT
# =========================================================

print(
    "\nML dataset saved to:"
)

print(
    ML_FILE
)


print("\nML Features:")

for feature in features:

    print(
        "  ✓",
        feature
    )


print(
    "\nTarget:"
)

print(
    "  ✓",
    target
)


print(
    "\nFinal ML dataset shape:"
)

print(
    ml_df.shape
)


print(
    "\nCongestion class distribution:"
)

print(
    ml_df["congestion_level"]
    .value_counts()
)


# =========================================================
# END
# =========================================================

print("\n" + "=" * 70)

print(
    "STEP 3 COMPLETED SUCCESSFULLY"
)

print("=" * 70)