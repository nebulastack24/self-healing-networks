import psutil
import time
import csv
from pathlib import Path
from datetime import datetime
from ping3 import ping


# ============================================================
# CONFIGURATION
# ============================================================

PING_TARGET = "8.8.8.8"
PING_COUNT = 5
MONITOR_INTERVAL = 10


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

# Current file:
# project/monitoring/network_monitor.py

# Go two levels up to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project-level data directory
DATA_DIR = PROJECT_ROOT / "data"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# CSV file
CSV_FILE = DATA_DIR / "network_metrics.csv"


# ============================================================
# INITIALIZE CSV FILE
# ============================================================

if not CSV_FILE.exists():

    with open(CSV_FILE, mode="w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "upload_mbps",
            "download_mbps",
            "latency_ms",
            "packet_loss_percent"
        ])


# ============================================================
# GET NETWORK BYTE COUNTERS
# ============================================================

def get_network_counters():
    """
    Returns total bytes sent and received
    by all network interfaces.
    """

    stats = psutil.net_io_counters()

    bytes_sent = stats.bytes_sent
    bytes_received = stats.bytes_recv

    return bytes_sent, bytes_received


# ============================================================
# CALCULATE UPLOAD AND DOWNLOAD SPEED
# ============================================================

def calculate_network_speed(
    previous_sent,
    previous_received,
    current_sent,
    current_received,
    interval
):
    """
    Calculates upload and download rate in Mbps.
    """

    sent_difference = current_sent - previous_sent
    received_difference = current_received - previous_received

    # Convert bytes to bits
    sent_bits = sent_difference * 8
    received_bits = received_difference * 8

    # Convert to Mbps
    upload_mbps = (sent_bits / interval) / 1_000_000
    download_mbps = (received_bits / interval) / 1_000_000

    return upload_mbps, download_mbps


# ============================================================
# MEASURE LATENCY AND PACKET LOSS
# ============================================================

def measure_latency_and_packet_loss():
    """
    Sends multiple ICMP echo requests and calculates
    average latency and packet loss.
    """

    successful_pings = []
    failed_pings = 0

    for _ in range(PING_COUNT):

        try:

            result = ping(
                PING_TARGET,
                timeout=2,
                unit="ms"
            )

            if result is None or result is False:

                failed_pings += 1

            else:

                successful_pings.append(result)

        except Exception:

            failed_pings += 1

        time.sleep(0.2)

    # Calculate packet loss
    packet_loss = (
        failed_pings / PING_COUNT
    ) * 100

    # Calculate average latency
    if successful_pings:

        average_latency = (
            sum(successful_pings)
            / len(successful_pings)
        )

    else:

        average_latency = None

    return average_latency, packet_loss


# ============================================================
# SAVE DATA TO CSV
# ============================================================

def save_to_csv(
    timestamp,
    upload_mbps,
    download_mbps,
    latency_ms,
    packet_loss
):
    """
    Saves one monitoring record into CSV.
    """

    with open(
        CSV_FILE,
        mode="a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            round(upload_mbps, 4),
            round(download_mbps, 4),
            (
                round(latency_ms, 2)
                if latency_ms is not None
                else "N/A"
            ),
            round(packet_loss, 2)
        ])


# ============================================================
# MAIN MONITORING FUNCTION
# ============================================================

def start_monitoring():

    print("=" * 70)
    print("       SELF-HEALING CAMPUS NETWORK MONITOR")
    print("=" * 70)

    print(f"Ping Target        : {PING_TARGET}")
    print(f"Monitoring Interval: {MONITOR_INTERVAL} seconds")
    print(f"Ping Count         : {PING_COUNT}")
    print(f"CSV File           : {CSV_FILE}")

    print("\nStarting network monitoring...")
    print("Press CTRL + C to stop.\n")

    # Get initial network counters
    previous_sent, previous_received = (
        get_network_counters()
    )

    previous_time = time.time()

    while True:

        # Wait for monitoring interval
        time.sleep(MONITOR_INTERVAL)

        # Current time
        current_time = time.time()

        # Current network counters
        current_sent, current_received = (
            get_network_counters()
        )

        # Calculate actual elapsed time
        elapsed_time = (
            current_time - previous_time
        )

        # Calculate upload/download
        upload_mbps, download_mbps = (
            calculate_network_speed(
                previous_sent,
                previous_received,
                current_sent,
                current_received,
                elapsed_time
            )
        )

        # Measure latency and packet loss
        latency_ms, packet_loss = (
            measure_latency_and_packet_loss()
        )

        # Timestamp
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Display results
        print("-" * 70)

        print(
            f"Timestamp          : {timestamp}"
        )

        print(
            f"Upload Rate        : "
            f"{upload_mbps:.4f} Mbps"
        )

        print(
            f"Download Rate      : "
            f"{download_mbps:.4f} Mbps"
        )

        if latency_ms is not None:

            print(
                f"Average Latency    : "
                f"{latency_ms:.2f} ms"
            )

        else:

            print(
                "Average Latency    : N/A"
            )

        print(
            f"Packet Loss        : "
            f"{packet_loss:.2f}%"
        )

        # Save data
        save_to_csv(
            timestamp,
            upload_mbps,
            download_mbps,
            latency_ms,
            packet_loss
        )

        # Update previous values
        previous_sent = current_sent
        previous_received = current_received
        previous_time = current_time


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        start_monitoring()

    except KeyboardInterrupt:

        print("\n")
        print("=" * 70)
        print("Network monitoring stopped.")
        print(f"Data saved to: {CSV_FILE}")
        print("=" * 70)