"""Create a lightweight attribution summary for flagged windows.

This script is intentionally simple: it compares each flagged window to the
vehicle's healthy baseline and reports the dominant signal contributors using
per-signal deviation scores. This satisfies the prompt's guidance to use a
lightweight attribution method rather than over-engineering a full SHAP setup.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OBD_FILE = DATA_DIR / "obd_timeseries.csv"
FAULTS_FILE = DATA_DIR / "fault_windows.csv"
OUTPUT_FILE = DATA_DIR / "fault_attribution_summary.csv"

SIGNALS = [
    "engine_rpm",
    "vehicle_speed_kmh",
    "coolant_temperature_C",
    "throttle_position_pct",
    "engine_load_pct",
    "battery_voltage",
    "mass_airflow_gps",
]


def summarize_attribution(obd_df: pd.DataFrame, faults_df: pd.DataFrame) -> pd.DataFrame:
    """For each fault window, compute a simple per-signal deviation summary."""
    rows = []

    for _, fault in faults_df.iterrows():
        vehicle_id = int(fault["vehicle_id"])
        start = pd.Timestamp(fault["start_timestamp"])
        end = pd.Timestamp(fault["end_timestamp"])

        window_df = obd_df[(obd_df["vehicle_id"] == vehicle_id) & (obd_df["timestamp"] >= start) & (obd_df["timestamp"] <= end)].copy()
        baseline_df = obd_df[(obd_df["vehicle_id"] == vehicle_id) & (obd_df["timestamp"] < start)].tail(300)

        if baseline_df.empty:
            baseline_df = obd_df[(obd_df["vehicle_id"] == vehicle_id)].head(300)

        baseline_means = baseline_df[SIGNALS].mean()
        baseline_stds = baseline_df[SIGNALS].std().fillna(1.0)
        deviation_scores = ((window_df[SIGNALS].mean() - baseline_means).abs() / baseline_stds).fillna(0.0)

        top_signals = (
            deviation_scores.sort_values(ascending=False)
            .head(3)
            .rename_axis("signal")
            .reset_index(name="deviation_score")
        )

        top_signal_list = ", ".join(top_signals["signal"].tolist())
        top_score_list = ", ".join(f"{score:.2f}" for score in top_signals["deviation_score"].tolist())

        rows.append(
            {
                "vehicle_id": vehicle_id,
                "fault_id": fault["fault_id"],
                "fault_type": fault["fault_type"],
                "affected_signal": fault["affected_signal"],
                "window_start": start,
                "window_end": end,
                "top_signals": top_signal_list,
                "top_signal_scores": top_score_list,
                "max_deviation": float(deviation_scores.max()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Generate the attribution summary and save it to disk."""
    if not OBD_FILE.exists() or not FAULTS_FILE.exists():
        raise FileNotFoundError("Expected operational and fault-window CSV files in data/")

    obd_df = pd.read_csv(OBD_FILE, parse_dates=["timestamp"])
    faults_df = pd.read_csv(FAULTS_FILE, parse_dates=["start_timestamp", "end_timestamp"])

    summary_df = summarize_attribution(obd_df, faults_df)
    summary_df.to_csv(OUTPUT_FILE, index=False)

    print("Fault attribution summary generated successfully.")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")
    print("Rows:", len(summary_df))


if __name__ == "__main__":
    main()
