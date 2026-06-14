#!/usr/bin/env python3
"""Generate synthetic vehicle diagnostics datasets for maintenance experiments.

This script creates:
- vehicle_metadata.csv
- obd_timeseries.csv
- fault_windows.csv
- dtc_logs.csv

It uses fixed random seeds for reproducibility and writes all outputs under the
project's data/ directory.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path("data")
VEHICLE_COUNT = 50
TRACK_SECONDS = 2 * 60 * 60  # 2 hours of 1 Hz data
MIN_FAULTS_PER_VEHICLE = 1
MAX_FAULTS_PER_VEHICLE = 3

MODEL_CHOICES = ["Sedan", "SUV", "Hatchback"]
MODEL_WEIGHTS = [0.45, 0.35, 0.20]
FUEL_CHOICES = ["Petrol", "Diesel", "Hybrid", "EV"]
FUEL_WEIGHTS = [0.45, 0.20, 0.20, 0.15]
MODE_CYCLE = ["idle", "city", "highway", "stop_and_go"]
FAULT_TYPES = ["sensor_drift", "spike", "stuck_at", "dropout", "noise"]


MODE_PROFILES: Dict[str, Dict[str, float]] = {
    "idle": {
        "speed_mean": 4.0,
        "speed_std": 2.5,
        "rpm_mean": 850.0,
        "rpm_std": 65.0,
        "throttle_mean": 8.0,
        "throttle_std": 3.0,
        "load_mean": 12.0,
        "load_std": 3.5,
        "maf_mean": 1.8,
        "maf_std": 0.8,
        "heat_bias": 0.0,
    },
    "city": {
        "speed_mean": 32.0,
        "speed_std": 7.5,
        "rpm_mean": 1800.0,
        "rpm_std": 150.0,
        "throttle_mean": 22.0,
        "throttle_std": 6.0,
        "load_mean": 35.0,
        "load_std": 6.0,
        "maf_mean": 9.5,
        "maf_std": 2.0,
        "heat_bias": 0.2,
    },
    "highway": {
        "speed_mean": 96.0,
        "speed_std": 12.0,
        "rpm_mean": 2600.0,
        "rpm_std": 220.0,
        "throttle_mean": 34.0,
        "throttle_std": 8.0,
        "load_mean": 52.0,
        "load_std": 8.0,
        "maf_mean": 28.0,
        "maf_std": 4.5,
        "heat_bias": 0.6,
    },
    "stop_and_go": {
        "speed_mean": 18.0,
        "speed_std": 8.5,
        "rpm_mean": 1350.0,
        "rpm_std": 180.0,
        "throttle_mean": 18.0,
        "throttle_std": 5.0,
        "load_mean": 26.0,
        "load_std": 4.5,
        "maf_mean": 7.0,
        "maf_std": 2.2,
        "heat_bias": 0.4,
    },
}


def set_seed(seed: int) -> None:
    """Configure deterministic random behavior for the generator."""
    random.seed(seed)
    np.random.seed(seed)


def generate_vehicle_metadata(vehicle_count: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate realistic vehicle metadata for the dataset."""
    rows = []
    for vehicle_id in range(1, vehicle_count + 1):
        model = rng.choice(MODEL_CHOICES, p=MODEL_WEIGHTS).item()
        mileage_km = max(1.0, int(rng.normal(62000.0, 32000.0)))
        age_years = max(0.5, min(12.0, float(rng.normal(4.2, 2.2))))
        fuel_type = rng.choice(FUEL_CHOICES, p=FUEL_WEIGHTS).item()

        rows.append(
            {
                "vehicle_id": vehicle_id,
                "model": model,
                "mileage_km": mileage_km,
                "age_years": round(age_years, 1),
                "fuel_type": fuel_type,
            }
        )

    return pd.DataFrame(rows)


def build_mode_schedule(duration_seconds: int, rng: np.random.Generator) -> List[Tuple[str, int]]:
    """Create a repeating sequence of driving modes for one vehicle."""
    schedule: List[Tuple[str, int]] = []
    remaining = duration_seconds
    current_mode_index = int(rng.integers(0, len(MODE_CYCLE)))

    while remaining > 0:
        segment_seconds = int(rng.integers(180, 900))
        segment_seconds = min(segment_seconds, remaining)
        schedule.append((MODE_CYCLE[current_mode_index], segment_seconds))
        remaining -= segment_seconds
        current_mode_index = (current_mode_index + 1) % len(MODE_CYCLE)

    return schedule


def simulate_vehicle_signals(vehicle_row: pd.Series, rng: np.random.Generator) -> Tuple[pd.DataFrame, List[dict]]:
    """Simulate a healthy multivariate time-series trace for one vehicle."""
    vehicle_id = int(vehicle_row["vehicle_id"])
    timestamps = pd.date_range("2026-06-14 00:00:00", periods=TRACK_SECONDS, freq="s", tz=None)

    mode_schedule = build_mode_schedule(TRACK_SECONDS, rng)
    segment_index = 0
    current_mode, segment_length = mode_schedule[0]
    segment_end = segment_length

    records = []
    fuel_level = float(rng.uniform(90.0, 98.0))

    for second, timestamp in enumerate(timestamps):
        if second >= segment_end:
            segment_index += 1
            current_mode, segment_length = mode_schedule[segment_index]
            segment_end += segment_length

        profile = MODE_PROFILES[current_mode]
        speed_base = profile["speed_mean"]
        rpm_base = profile["rpm_mean"]
        throttle_base = profile["throttle_mean"]
        load_base = profile["load_mean"]

        mode_noise = rng.normal(0.0, 1.2)
        throttle = max(0.0, min(100.0, rng.normal(throttle_base, profile["throttle_std"]) + 0.4 * mode_noise + 0.3 * (second % 120) / 40.0))
        speed = max(0.0, min(160.0, rng.normal(speed_base, profile["speed_std"]) + 0.20 * throttle + 0.8 * mode_noise))
        rpm = max(600.0, min(4500.0, rng.normal(rpm_base, profile["rpm_std"]) + 2.4 * speed + 18.0 * throttle + 12.0 * mode_noise))
        engine_load = max(0.0, min(100.0, rng.normal(load_base, profile["load_std"]) + 0.35 * throttle + 2.0 * mode_noise))
        maf = max(0.0, min(80.0, rng.normal(profile["maf_mean"], profile["maf_std"]) + 0.05 * speed + 0.12 * engine_load + 0.02 * rpm + 0.5 * mode_noise))

        coolant_temperature = 85.0 + 0.015 * second + profile["heat_bias"] * 4.0 + rng.normal(0.0, 0.45)
        battery_voltage = 12.4 + 0.02 * throttle + rng.normal(0.0, 0.08)
        fuel_level = max(0.0, min(100.0, fuel_level - 0.003 + rng.normal(0.0, 0.18)))

        # Add vehicle-specific offsets to create realistic variability.
        speed += 0.8 * math.sin((vehicle_id * 0.15) + second * 0.02)
        rpm += 15.0 * math.cos((vehicle_id * 0.07) + second * 0.01)
        battery_voltage += 0.02 * ((vehicle_id % 4) - 1.5)
        coolant_temperature += 0.5 * ((vehicle_id % 3) - 1)
        fuel_level += 0.3 * ((vehicle_id % 5) - 2)

        records.append(
            {
                "timestamp": timestamp,
                "vehicle_id": vehicle_id,
                "engine_rpm": round(rpm, 2),
                "vehicle_speed_kmh": round(speed, 2),
                "coolant_temperature_C": round(coolant_temperature, 2),
                "throttle_position_pct": round(throttle, 2),
                "engine_load_pct": round(engine_load, 2),
                "fuel_level_pct": round(fuel_level, 2),
                "battery_voltage": round(battery_voltage, 2),
                "mass_airflow_gps": round(maf, 2),
                "fault_label": "healthy",
            }
        )

    return pd.DataFrame(records), []


def _overlaps(candidate_start: int, candidate_end: int, existing_windows: List[Tuple[int, int]]) -> bool:
    """Return True when a fault window overlaps an existing one."""
    return any(not (candidate_end <= existing_start or candidate_start >= existing_end) for existing_start, existing_end in existing_windows)


def inject_faults(obd_df: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    """Inject synthetic fault windows and return the labeled data plus DTC events."""
    vehicle_id = int(obd_df["vehicle_id"].iloc[0])
    fault_windows: List[dict] = []
    dtc_events: List[dict] = []

    total_seconds = len(obd_df)
    fault_count = int(rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10]))
    existing_windows: List[Tuple[int, int]] = []

    for fault_index in range(fault_count):
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            attempts += 1
            start_idx = int(rng.integers(180, total_seconds - 300))
            duration_seconds = int(rng.integers(60, 241))
            end_idx = min(total_seconds, start_idx + duration_seconds)

            if _overlaps(start_idx, end_idx, existing_windows):
                continue

            fault_type = rng.choice(FAULT_TYPES).item()
            if fault_type == "sensor_drift":
                affected_signal = rng.choice(["coolant_temperature_C", "battery_voltage"]).item()
            elif fault_type == "spike":
                affected_signal = rng.choice(["engine_rpm", "mass_airflow_gps"]).item()
            elif fault_type == "stuck_at":
                affected_signal = rng.choice(["vehicle_speed_kmh", "throttle_position_pct"]).item()
            elif fault_type == "dropout":
                affected_signal = rng.choice(["battery_voltage", "mass_airflow_gps", "throttle_position_pct"]).item()
            else:
                affected_signal = "engine_load_pct"

            start_ts = obd_df.iloc[start_idx]["timestamp"]
            end_ts = obd_df.iloc[min(end_idx - 1, total_seconds - 1)]["timestamp"]

            window_slice = obd_df.iloc[start_idx:end_idx].copy()
            fault_windows.append(
                {
                    "vehicle_id": vehicle_id,
                    "fault_id": f"{vehicle_id:02d}-{fault_index + 1}",
                    "fault_type": fault_type,
                    "affected_signal": affected_signal,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                }
            )

            for row_index in range(start_idx, end_idx):
                current_value = float(obd_df.loc[row_index, affected_signal])
                relative = (row_index - start_idx) / max(1, end_idx - start_idx)

                if fault_type == "sensor_drift":
                    drift_strength = rng.uniform(0.8, 2.5)
                    if affected_signal == "coolant_temperature_C":
                        drift_strength *= 1.3
                    elif affected_signal == "battery_voltage":
                        drift_strength *= 0.18
                    if rng.random() < 0.5:
                        drift_strength *= -1.0
                    obd_df.loc[row_index, affected_signal] = current_value + drift_strength * relative

                elif fault_type == "spike":
                    spike_strength = rng.uniform(25.0, 120.0)
                    if affected_signal == "engine_rpm":
                        spike_strength = max(spike_strength, 80.0)
                    else:
                        spike_strength = max(spike_strength, 8.0)
                    if 0.35 <= relative <= 0.65:
                        obd_df.loc[row_index, affected_signal] = current_value + spike_strength
                    elif relative < 0.1 or relative > 0.9:
                        obd_df.loc[row_index, affected_signal] = current_value + 0.2 * spike_strength

                elif fault_type == "stuck_at":
                    stuck_value = current_value
                    if affected_signal == "vehicle_speed_kmh":
                        stuck_value = float(rng.uniform(8.0, 18.0))
                    else:
                        stuck_value = float(rng.uniform(18.0, 28.0))
                    obd_df.loc[row_index, affected_signal] = stuck_value

                elif fault_type == "dropout":
                    # Intermittent dropout: brief signal suppression to mimic missing or zeroed sensor values.
                    pulse = (row_index - start_idx) % 9
                    if pulse < 3:
                        if affected_signal in {"battery_voltage", "mass_airflow_gps"}:
                            obd_df.loc[row_index, affected_signal] = current_value * 0.08
                        else:
                            obd_df.loc[row_index, affected_signal] = max(0.0, current_value - 20.0)
                    else:
                        obd_df.loc[row_index, affected_signal] = current_value

                else:  # noise
                    noise_scale = rng.uniform(5.0, 12.0)
                    obd_df.loc[row_index, affected_signal] = current_value + rng.normal(0.0, noise_scale)

                obd_df.loc[row_index, "fault_label"] = fault_type

            # Create one DTC event per fault near the end of the window.
            dtc_timestamp = obd_df.iloc[min(end_idx - 1, total_seconds - 1)]["timestamp"]
            if fault_type == "sensor_drift":
                if affected_signal == "coolant_temperature_C":
                    dtc_code = "P0117"
                    description = "Engine coolant temperature sensor circuit issue"
                else:
                    dtc_code = "P0562"
                    description = "Battery voltage low / sensor drift"
            elif fault_type == "spike":
                dtc_code = "P0300"
                description = "Random/multiple cylinder misfire or RPM spike"
            elif fault_type == "stuck_at":
                dtc_code = "P0500"
                description = "Vehicle speed sensor or throttle position stuck"
            elif fault_type == "dropout":
                dtc_code = "P0102" if affected_signal == "mass_airflow_gps" else "P0562"
                description = "Intermittent signal dropout or voltage drop"
            else:
                dtc_code = "P0101"
                description = "Mass air flow / load sensor noise anomaly"

            dtc_events.append(
                {
                    "vehicle_id": vehicle_id,
                    "timestamp": dtc_timestamp,
                    "dtc_code": dtc_code,
                    "description": description,
                    "fault_id": fault_windows[-1]["fault_id"],
                }
            )

            existing_windows.append((start_idx, end_idx))
            placed = True

    return obd_df, fault_windows, dtc_events


def validate_fault_windows(fault_windows: List[dict]) -> None:
    """Ensure that windows do not overlap for the same vehicle."""
    by_vehicle: Dict[int, List[Tuple[int, int]]] = {}
    for fault in fault_windows:
        start = pd.Timestamp(fault["start_timestamp"]).value
        end = pd.Timestamp(fault["end_timestamp"]).value
        by_vehicle.setdefault(int(fault["vehicle_id"]), []).append((start, end))

    for vehicle_id, windows in by_vehicle.items():
        windows = sorted(windows)
        for left, right in zip(windows, windows[1:]):
            if left[1] >= right[0]:
                raise ValueError(f"Overlapping fault windows found for vehicle {vehicle_id}")


def main() -> None:
    """Generate all dataset artifacts and write them under the data/ directory."""
    set_seed(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    metadata = generate_vehicle_metadata(VEHICLE_COUNT, rng)

    all_obd_frames = []
    all_fault_windows = []
    all_dtc_events = []

    for _, vehicle_row in metadata.iterrows():
        healthy_df, _ = simulate_vehicle_signals(vehicle_row, rng)
        injected_df, fault_windows, dtc_events = inject_faults(healthy_df, rng)
        all_obd_frames.append(injected_df)
        all_fault_windows.extend(fault_windows)
        all_dtc_events.extend(dtc_events)

    obd_df = pd.concat(all_obd_frames, ignore_index=True)
    fault_windows_df = pd.DataFrame(all_fault_windows)
    dtc_df = pd.DataFrame(all_dtc_events)

    validate_fault_windows(all_fault_windows)

    metadata.to_csv(DATA_DIR / "vehicle_metadata.csv", index=False)
    obd_df.to_csv(DATA_DIR / "obd_timeseries.csv", index=False)
    fault_windows_df.to_csv(DATA_DIR / "fault_windows.csv", index=False)
    dtc_df.to_csv(DATA_DIR / "dtc_logs.csv", index=False)

    healthy_share = (obd_df["fault_label"] == "healthy").mean() * 100
    faults_by_type = obd_df["fault_label"].value_counts().to_dict()

    print("Synthetic vehicle diagnostics dataset generated successfully.")
    print(f"Number of vehicles: {len(metadata)}")
    print(f"Total records: {len(obd_df)}")
    print(f"Healthy share: {healthy_share:.2f}%")
    print("Fault counts by type:")
    for fault_type, count in sorted(faults_by_type.items()):
        print(f"  - {fault_type}: {count}")
    print(f"Number of DTC events: {len(dtc_df)}")
    print(f"Output directory: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
