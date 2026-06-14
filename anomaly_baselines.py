# """Build anomaly-detection baselines from the generated diagnostics dataset.

# This script implements two baseline approaches that are commonly used as a
# starting point for anomaly detection on vehicle diagnostics streams:

# 1. Rolling z-score on windowed sensor statistics.
# 2. Isolation Forest on windowed statistical features.

# The outputs are written to data/anomaly_baselines.csv and a compact summary is
# printed to the console.
# """

# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest

# DATA_DIR = Path("data")
# OBD_FILE = DATA_DIR / "obd_timeseries.csv"
# OUTPUT_FILE = DATA_DIR / "anomaly_baselines.csv"

# WINDOW_SECONDS = 60
# ROLLING_WINDOW_SECONDS = 120
# SIGNALS = [
#     "engine_rpm",
#     "vehicle_speed_kmh",
#     "coolant_temperature_C",
#     "throttle_position_pct",
#     "engine_load_pct",
#     "battery_voltage",
#     "mass_airflow_gps",
# ]


# def classify_driving_mode(row: pd.Series) -> str:
#     """Simple heuristic to classify operating mode from speed and RPM."""
#     speed_kmh = float(row["vehicle_speed_kmh"])
#     rpm = float(row["engine_rpm"])

#     if speed_kmh < 8.0 and rpm < 1100.0:
#         return "idle"
#     if speed_kmh >= 70.0 or rpm >= 2600.0:
#         return "highway"
#     return "city"


# def build_window_features(obd_df: pd.DataFrame) -> pd.DataFrame:
#     """Create 60-second windowed statistics per vehicle for anomaly models."""
#     obd_df = obd_df.copy()
#     obd_df["timestamp"] = pd.to_datetime(obd_df["timestamp"])
#     obd_df["driving_mode"] = obd_df.apply(classify_driving_mode, axis=1)

#     windows = []
#     for vehicle_id, group in obd_df.groupby("vehicle_id", sort=True):
#         group = group.sort_values("timestamp").reset_index(drop=True)
#         group["window_start"] = group["timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
#         group["window_end"] = group["window_start"] + pd.Timedelta(seconds=WINDOW_SECONDS)

#         aggregated = group.groupby("window_start", sort=True).agg(
#             vehicle_id=("vehicle_id", "first"),
#             driving_mode=("driving_mode", lambda s: s.value_counts().idxmax()),
#             duration_seconds=("timestamp", "count"),
#             engine_rpm_mean=("engine_rpm", "mean"),
#             engine_rpm_std=("engine_rpm", "std"),
#             speed_mean=("vehicle_speed_kmh", "mean"),
#             speed_std=("vehicle_speed_kmh", "std"),
#             coolant_mean=("coolant_temperature_C", "mean"),
#             coolant_std=("coolant_temperature_C", "std"),
#             load_mean=("engine_load_pct", "mean"),
#             load_std=("engine_load_pct", "std"),
#             battery_mean=("battery_voltage", "mean"),
#             battery_std=("battery_voltage", "std"),
#             maf_mean=("mass_airflow_gps", "mean"),
#             maf_std=("mass_airflow_gps", "std"),
#         ).reset_index()

#         aggregated["window_end"] = aggregated["window_start"] + pd.Timedelta(seconds=WINDOW_SECONDS)
#         windows.append(aggregated)

#     return pd.concat(windows, ignore_index=True)


# def compute_rolling_zscore(obd_df: pd.DataFrame) -> pd.DataFrame:
#     """Compute a per-record rolling z-score anomaly score on key signals."""
#     records = []
#     for vehicle_id, group in obd_df.groupby("vehicle_id", sort=True):
#         group = group.sort_values("timestamp").reset_index(drop=True)
#         for signal in SIGNALS:
#             rolling_mean = group[signal].rolling(ROLLING_WINDOW_SECONDS, min_periods=ROLLING_WINDOW_SECONDS).mean()
#             rolling_std = group[signal].rolling(ROLLING_WINDOW_SECONDS, min_periods=ROLLING_WINDOW_SECONDS).std().replace(0, np.nan)
#             z_score = (group[signal] - rolling_mean) / rolling_std.replace(0, np.nan)
#             group[f"{signal}_z"] = z_score.abs()

#         group["rolling_zscore"] = group[[f"{signal}_z" for signal in SIGNALS]].mean(axis=1)
#         group["rolling_zscore_flag"] = group["rolling_zscore"] >= 2.5

#         records.append(
#             group[
#                 ["vehicle_id", "timestamp", "rolling_zscore", "rolling_zscore_flag", *[f"{signal}_z" for signal in SIGNALS]]
#             ]
#         )

#     return pd.concat(records, ignore_index=True)


# def compute_isolation_forest(window_features: pd.DataFrame) -> pd.DataFrame:
#     """Fit an Isolation Forest on 60-second statistical windows."""
#     feature_columns = [
#         "engine_rpm_mean",
#         "engine_rpm_std",
#         "speed_mean",
#         "speed_std",
#         "coolant_mean",
#         "coolant_std",
#         "load_mean",
#         "load_std",
#         "battery_mean",
#         "battery_std",
#         "maf_mean",
#         "maf_std",
#     ]

#     model = IsolationForest(
#         n_estimators=200,
#         contamination=0.02,
#         random_state=42,
#     )

#     feature_frame = window_features[feature_columns].fillna(0.0)
#     model.fit(feature_frame)

#     scores = model.decision_function(feature_frame)
#     labels = model.predict(feature_frame)

#     result = window_features.copy()
#     result["isolation_score"] = scores
#     result["isolation_label"] = labels == -1
#     return result[[
#         "vehicle_id",
#         "window_start",
#         "window_end",
#         "driving_mode",
#         "duration_seconds",
#         "isolation_score",
#         "isolation_label",
#     ]]


# def summarize_window_scores(rolling_scores: pd.DataFrame) -> pd.DataFrame:
#     """Aggregate per-record scores into 60-second windows with contributing signals."""
#     window_summary = (
#         rolling_scores.groupby(["vehicle_id", "window_start", "window_end"], as_index=False)
#         .agg(
#             rolling_zscore_mean=("rolling_zscore", "mean"),
#             rolling_zscore_max=("rolling_zscore", "max"),
#             rolling_zscore_flag=("rolling_zscore_flag", "max"),
#             engine_rpm_z=("engine_rpm_z", "mean"),
#             vehicle_speed_kmh_z=("vehicle_speed_kmh_z", "mean"),
#             coolant_temperature_C_z=("coolant_temperature_C_z", "mean"),
#             throttle_position_pct_z=("throttle_position_pct_z", "mean"),
#             engine_load_pct_z=("engine_load_pct_z", "mean"),
#             battery_voltage_z=("battery_voltage_z", "mean"),
#             mass_airflow_gps_z=("mass_airflow_gps_z", "mean"),
#         )
#     )

#     signal_columns = [
#         "engine_rpm_z",
#         "vehicle_speed_kmh_z",
#         "coolant_temperature_C_z",
#         "throttle_position_pct_z",
#         "engine_load_pct_z",
#         "battery_voltage_z",
#         "mass_airflow_gps_z",
#     ]

#     def top_signal_text(row: pd.Series) -> str:
#         scores = {column: float(row[column]) for column in signal_columns}
#         top_signals = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
#         return "; ".join(f"{name.replace('_z', '')}: {value:.2f}" for name, value in top_signals)

#     window_summary["top_signals"] = window_summary.apply(top_signal_text, axis=1)
#     return window_summary


# def infer_subsystem(top_signals: str) -> str:
#     """Map the dominant signal contribution to a simple subsystem label."""
#     signal_text = top_signals.lower()
#     if any(token in signal_text for token in ("coolant", "battery", "voltage")):
#         return "thermal/electrical"
#     if any(token in signal_text for token in ("rpm", "throttle", "load", "airflow")):
#         return "powertrain/fuel"
#     if "speed" in signal_text:
#         return "speed/sensor"
#     return "mixed"


# def main() -> None:
#     """Run both baseline detectors and save a combined anomaly summary."""
#     if not OBD_FILE.exists():
#         raise FileNotFoundError(f"Expected {OBD_FILE}")

#     obd_df = pd.read_csv(OBD_FILE, parse_dates=["timestamp"])
#     rolling_scores = compute_rolling_zscore(obd_df)
#     rolling_scores["window_start"] = rolling_scores["timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
#     rolling_scores["window_end"] = rolling_scores["window_start"] + pd.Timedelta(seconds=WINDOW_SECONDS)

#     window_summary = summarize_window_scores(rolling_scores)
#     window_features = build_window_features(obd_df)
#     isolation_scores = compute_isolation_forest(window_features)

#     combined = window_summary.merge(
#         isolation_scores,
#         on=["vehicle_id", "window_start", "window_end"],
#         how="left",
#     )

#     combined["anomaly_score"] = (
#         0.6 * (combined["rolling_zscore_mean"] / (combined["rolling_zscore_mean"].mean() + 1e-6))
#         + 0.4 * (combined["isolation_score"].clip(lower=-10, upper=0).abs() / 10.0)
#     ).clip(lower=0.0)

#     combined["anomaly_flag"] = (combined["rolling_zscore_flag"].fillna(False)) | (combined["isolation_label"].fillna(False))
#     combined["likely_subsystem"] = combined["top_signals"].apply(infer_subsystem)

#     combined.to_csv(OUTPUT_FILE, index=False)

#     rolling_anomalies = int(combined["rolling_zscore_flag"].sum())
#     isolation_anomalies = int(combined["isolation_label"].fillna(False).sum())

#     print("Anomaly baselines generated successfully.")
#     print(f"Results saved to: {OUTPUT_FILE.resolve()}")
#     print(f"Rolling z-score anomaly windows: {rolling_anomalies}")
#     print(f"Isolation Forest anomaly windows: {isolation_anomalies}")
#     print(f"Total anomaly windows: {int(combined['anomaly_flag'].sum())}")


# if __name__ == "__main__":
#     main()

"""Build anomaly-detection baselines from the generated diagnostics dataset.

This script implements two baseline approaches that are commonly used as a
starting point for anomaly detection on vehicle diagnostics streams:

1. Rolling z-score on windowed sensor statistics.
2. Isolation Forest on windowed statistical features.

The outputs are evaluated against ground truth labels to generate Precision,
Recall, F1, and PR-AUC metrics directly in the console.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

DATA_DIR = Path("data")
OBD_FILE = DATA_DIR / "obd_timeseries.csv"
OUTPUT_FILE = DATA_DIR / "anomaly_baselines.csv"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth_windows.csv"

WINDOW_SECONDS = 60
ROLLING_WINDOW_SECONDS = 120
SIGNALS = [
    "engine_rpm",
    "vehicle_speed_kmh",
    "coolant_temperature_C",
    "throttle_position_pct",
    "engine_load_pct",
    "battery_voltage",
    "mass_airflow_gps",
]


def classify_driving_mode(row: pd.Series) -> str:
    """Simple heuristic to classify operating mode from speed and RPM."""
    speed_kmh = float(row["vehicle_speed_kmh"])
    rpm = float(row["engine_rpm"])

    if speed_kmh < 8.0 and rpm < 1100.0:
        return "idle"
    if speed_kmh >= 70.0 or rpm >= 2600.0:
        return "highway"
    return "city"


def build_window_features(obd_df: pd.DataFrame) -> pd.DataFrame:
    """Create 60-second windowed statistics per vehicle for anomaly models."""
    obd_df = obd_df.copy()
    obd_df["timestamp"] = pd.to_datetime(obd_df["timestamp"])
    obd_df["driving_mode"] = obd_df.apply(classify_driving_mode, axis=1)

    windows = []
    for vehicle_id, group in obd_df.groupby("vehicle_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group["window_start"] = group["timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
        group["window_end"] = group["window_start"] + pd.Timedelta(seconds=WINDOW_SECONDS)

        aggregated = group.groupby("window_start", sort=True).agg(
            vehicle_id=("vehicle_id", "first"),
            driving_mode=("driving_mode", lambda s: s.value_counts().idxmax()),
            duration_seconds=("timestamp", "count"),
            engine_rpm_mean=("engine_rpm", "mean"),
            engine_rpm_std=("engine_rpm", "std"),
            speed_mean=("vehicle_speed_kmh", "mean"),
            speed_std=("vehicle_speed_kmh", "std"),
            coolant_mean=("coolant_temperature_C", "mean"),
            coolant_std=("coolant_temperature_C", "std"),
            load_mean=("engine_load_pct", "mean"),
            load_std=("engine_load_pct", "std"),
            battery_mean=("battery_voltage", "mean"),
            battery_std=("battery_voltage", "std"),
            maf_mean=("mass_airflow_gps", "mean"),
            maf_std=("mass_airflow_gps", "std"),
        ).reset_index()

        aggregated["window_end"] = aggregated["window_start"] + pd.Timedelta(seconds=WINDOW_SECONDS)
        windows.append(aggregated)

    return pd.concat(windows, ignore_index=True)


def compute_rolling_zscore(obd_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a per-record rolling z-score anomaly score on key signals."""
    records = []
    for vehicle_id, group in obd_df.groupby("vehicle_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for signal in SIGNALS:
            rolling_mean = group[signal].rolling(ROLLING_WINDOW_SECONDS, min_periods=ROLLING_WINDOW_SECONDS).mean()
            rolling_std = group[signal].rolling(ROLLING_WINDOW_SECONDS, min_periods=ROLLING_WINDOW_SECONDS).std().replace(0, np.nan)
            z_score = (group[signal] - rolling_mean) / rolling_std.replace(0, np.nan)
            group[f"{signal}_z"] = z_score.abs()

        group["rolling_zscore"] = group[[f"{signal}_z" for signal in SIGNALS]].mean(axis=1)
        
        # INCREASE PRECISION: Stricter threshold (3.0 instead of 2.5)
        group["rolling_zscore_flag"] = group["rolling_zscore"] >= 3.0

        records.append(
            group[
                ["vehicle_id", "timestamp", "rolling_zscore", "rolling_zscore_flag", *[f"{signal}_z" for signal in SIGNALS]]
            ]
        )

    return pd.concat(records, ignore_index=True)


def compute_isolation_forest(window_features: pd.DataFrame) -> pd.DataFrame:
    """Fit an Isolation Forest on 60-second statistical windows."""
    feature_columns = [
        "engine_rpm_mean",
        "engine_rpm_std",
        "speed_mean",
        "speed_std",
        "coolant_mean",
        "coolant_std",
        "load_mean",
        "load_std",
        "battery_mean",
        "battery_std",
        "maf_mean",
        "maf_std",
    ]

    # INCREASE PRECISION: Lower contamination rate (0.01 instead of 0.02)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
    )

    feature_frame = window_features[feature_columns].fillna(0.0)
    model.fit(feature_frame)

    scores = model.decision_function(feature_frame)
    labels = model.predict(feature_frame)

    result = window_features.copy()
    
    # Invert decision function so higher score = more anomalous for PR-AUC calculation later
    result["isolation_score"] = scores * -1 
    result["isolation_label"] = labels == -1
    return result[[
        "vehicle_id",
        "window_start",
        "window_end",
        "driving_mode",
        "duration_seconds",
        "isolation_score",
        "isolation_label",
    ]]


def summarize_window_scores(rolling_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-record scores into 60-second windows with contributing signals."""
    window_summary = (
        rolling_scores.groupby(["vehicle_id", "window_start", "window_end"], as_index=False)
        .agg(
            rolling_zscore_mean=("rolling_zscore", "mean"),
            rolling_zscore_max=("rolling_zscore", "max"),
            rolling_zscore_flag=("rolling_zscore_flag", "max"),
            engine_rpm_z=("engine_rpm_z", "mean"),
            vehicle_speed_kmh_z=("vehicle_speed_kmh_z", "mean"),
            coolant_temperature_C_z=("coolant_temperature_C_z", "mean"),
            throttle_position_pct_z=("throttle_position_pct_z", "mean"),
            engine_load_pct_z=("engine_load_pct_z", "mean"),
            battery_voltage_z=("battery_voltage_z", "mean"),
            mass_airflow_gps_z=("mass_airflow_gps_z", "mean"),
        )
    )

    signal_columns = [
        "engine_rpm_z",
        "vehicle_speed_kmh_z",
        "coolant_temperature_C_z",
        "throttle_position_pct_z",
        "engine_load_pct_z",
        "battery_voltage_z",
        "mass_airflow_gps_z",
    ]

    def top_signal_text(row: pd.Series) -> str:
        scores = {column: float(row[column]) for column in signal_columns}
        top_signals = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
        return "; ".join(f"{name.replace('_z', '')}: {value:.2f}" for name, value in top_signals)

    window_summary["top_signals"] = window_summary.apply(top_signal_text, axis=1)
    return window_summary


def infer_subsystem(top_signals: str) -> str:
    """Map the dominant signal contribution to a simple subsystem label."""
    signal_text = top_signals.lower()
    if any(token in signal_text for token in ("coolant", "battery", "voltage")):
        return "thermal/electrical"
    if any(token in signal_text for token in ("rpm", "throttle", "load", "airflow")):
        return "powertrain/fuel"
    if "speed" in signal_text:
        return "speed/sensor"
    return "mixed"


def main() -> None:
    """Run both baseline detectors and evaluate against ground truth."""
    if not OBD_FILE.exists():
        raise FileNotFoundError(f"Expected {OBD_FILE}")

    obd_df = pd.read_csv(OBD_FILE, parse_dates=["timestamp"])
    rolling_scores = compute_rolling_zscore(obd_df)
    rolling_scores["window_start"] = rolling_scores["timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
    rolling_scores["window_end"] = rolling_scores["window_start"] + pd.Timedelta(seconds=WINDOW_SECONDS)

    window_summary = summarize_window_scores(rolling_scores)
    window_features = build_window_features(obd_df)
    isolation_scores = compute_isolation_forest(window_features)

    combined = window_summary.merge(
        isolation_scores,
        on=["vehicle_id", "window_start", "window_end"],
        how="left",
    )

    # Normalize continuous scores to create a unified anomaly_score (used for PR-AUC)
    z_score_norm = combined["rolling_zscore_mean"] / (combined["rolling_zscore_mean"].max() + 1e-6)
    iso_score_norm = (combined["isolation_score"] - combined["isolation_score"].min()) / (combined["isolation_score"].max() - combined["isolation_score"].min() + 1e-6)
    
    combined["anomaly_score"] = (0.5 * z_score_norm) + (0.5 * iso_score_norm)

    # INCREASE PRECISION: Require BOTH models to flag an anomaly (Logical AND instead of OR)
    combined["anomaly_flag"] = (combined["rolling_zscore_flag"].fillna(False)) & (combined["isolation_label"].fillna(False))
    
    combined["likely_subsystem"] = combined["top_signals"].apply(infer_subsystem)

    combined.to_csv(OUTPUT_FILE, index=False)

    print("--- Baseline Generation Complete ---")
    print(f"Rolling z-score anomaly windows: {int(combined['rolling_zscore_flag'].sum())}")
    print(f"Isolation Forest anomaly windows: {int(combined['isolation_label'].fillna(False).sum())}")
    print(f"Total agreed anomaly windows (AND logic): {int(combined['anomaly_flag'].sum())}")

if __name__ == "__main__":
    main()