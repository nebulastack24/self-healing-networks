from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "network_metrics.csv"
PLOT_DIR = PROJECT_ROOT / "experiments" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 70)
print("STEP 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"\nDataset loaded: {df.shape[0]} rows")


# ---------------------------------------------------------
# CONVERT TIMESTAMP
# ---------------------------------------------------------

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.sort_values("timestamp")


# ---------------------------------------------------------
# 1. CONGESTION DISTRIBUTION
# ---------------------------------------------------------

if "congestion_level" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="congestion_level"
    )

    plt.title("Congestion Level Distribution")
    plt.xlabel("Congestion Level")
    plt.ylabel("Number of Records")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "01_congestion_distribution.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# 2. LATENCY VS TIME
# ---------------------------------------------------------

if "timestamp" in df.columns and "latency_ms" in df.columns:

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["timestamp"],
        df["latency_ms"]
    )

    plt.title("Latency vs Time")
    plt.xlabel("Time")
    plt.ylabel("Latency (ms)")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "02_latency_vs_time.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# 3. UPLOAD / DOWNLOAD VS TIME
# ---------------------------------------------------------

if "timestamp" in df.columns:

    if (
        "upload_mbps" in df.columns
        and "download_mbps" in df.columns
    ):

        plt.figure(figsize=(10, 5))

        plt.plot(
            df["timestamp"],
            df["upload_mbps"],
            label="Upload"
        )

        plt.plot(
            df["timestamp"],
            df["download_mbps"],
            label="Download"
        )

        plt.title("Upload and Download Rate vs Time")
        plt.xlabel("Time")
        plt.ylabel("Mbps")

        plt.legend()

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "03_upload_download_vs_time.png",
            dpi=300
        )

        plt.show()


# ---------------------------------------------------------
# 4. LATENCY BY CONGESTION
# ---------------------------------------------------------

if (
    "congestion_level" in df.columns
    and "latency_ms" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="congestion_level",
        y="latency_ms"
    )

    plt.title("Latency by Congestion Level")
    plt.xlabel("Congestion Level")
    plt.ylabel("Latency (ms)")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "04_latency_by_congestion.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# 5. PACKET LOSS BY CONGESTION
# ---------------------------------------------------------

if (
    "congestion_level" in df.columns
    and "packet_loss_percent" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="congestion_level",
        y="packet_loss_percent"
    )

    plt.title("Packet Loss by Congestion Level")
    plt.xlabel("Congestion Level")
    plt.ylabel("Packet Loss (%)")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "05_packet_loss_by_congestion.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# 6. BANDWIDTH USAGE BY CONGESTION
# ---------------------------------------------------------

if (
    "congestion_level" in df.columns
    and "bandwidth_usage_percent" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="congestion_level",
        y="bandwidth_usage_percent"
    )

    plt.title("Bandwidth Usage by Congestion Level")
    plt.xlabel("Congestion Level")
    plt.ylabel("Bandwidth Usage (%)")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "06_bandwidth_by_congestion.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# 7. DEVICES BY CONGESTION
# ---------------------------------------------------------

if (
    "congestion_level" in df.columns
    and "connected_devices" in df.columns
):

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="congestion_level",
        y="connected_devices"
    )

    plt.title("Connected Devices by Congestion Level")
    plt.xlabel("Congestion Level")
    plt.ylabel("Connected Devices")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "07_devices_by_congestion.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# 8. CORRELATION MATRIX
# ---------------------------------------------------------

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:

    plt.figure(figsize=(10, 8))

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Network Metrics Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "08_correlation_matrix.png",
        dpi=300
    )

    plt.show()


# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print(f"\nGraphs saved to:")
print(PLOT_DIR)