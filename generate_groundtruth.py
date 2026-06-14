#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

OBD_FILE = DATA_DIR / "obd_timeseries.csv"
OUTPUT_FILE = DATA_DIR / "ground_truth_windows.csv"

WINDOW_SECONDS = 60


def main():

    obd_df = pd.read_csv(
        OBD_FILE,
        parse_dates=["timestamp"],
    )

    # ---------------------------
    # BASE LABEL (IMPORTANT FIX)
    # ---------------------------
    obd_df["actual_anomaly"] = (
        obd_df["fault_label"] != "healthy"
    ).astype(int)

    # ---------------------------
    # WINDOW CREATION
    # ---------------------------
    obd_df["window_start"] = obd_df["timestamp"].dt.floor(f"{WINDOW_SECONDS}s")

    obd_df["window_end"] = (
        obd_df["window_start"]
        + pd.Timedelta(seconds=WINDOW_SECONDS)
    )

    # ---------------------------
    # WINDOW-LEVEL AGGREGATION
    # ---------------------------
    ground_truth = (
        obd_df.groupby(
            ["vehicle_id", "window_start", "window_end"],
            as_index=False,
        )
        .agg(
            anomaly_count=("actual_anomaly", "sum"),
            total_count=("actual_anomaly", "count"),
            max_anomaly=("actual_anomaly", "max"),
        )
    )

    # ---------------------------
    # EXTRA FEATURES (YOUR UPGRADE)
    # ---------------------------
    ground_truth["anomaly_ratio"] = (
        ground_truth["anomaly_count"] / ground_truth["total_count"]
    )

    def severity(r):
        if r == 0:
            return "normal"
        elif r < 0.1:
            return "low"
        elif r < 0.4:
            return "medium"
        else:
            return "high"

    ground_truth["severity"] = ground_truth["anomaly_ratio"].apply(severity)

    # ---------------------------
    # 🔥 CRITICAL FIX FOR YOUR ERROR
    # ---------------------------
    ground_truth["actual_anomaly"] = ground_truth["max_anomaly"]

    # ---------------------------
    # SAVE FILE
    # ---------------------------
    ground_truth.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved ground truth to: {OUTPUT_FILE.resolve()}")

    print("\nSeverity distribution:")
    print(ground_truth["severity"].value_counts())


if __name__ == "__main__":
    main()