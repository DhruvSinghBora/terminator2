"""Example script for loading the generated datasets into pandas."""

import pandas as pd


def main() -> None:
    metadata = pd.read_csv("data/vehicle_metadata.csv")
    obd = pd.read_csv("data/obd_timeseries.csv", parse_dates=["timestamp"])
    faults = pd.read_csv("data/fault_windows.csv", parse_dates=["start_timestamp", "end_timestamp"])
    dtc = pd.read_csv("data/dtc_logs.csv", parse_dates=["timestamp"])

    print("Vehicles:", metadata.shape[0])
    print("Operational records:", obd.shape[0])
    print("Fault windows:", faults.shape[0])
    print("DTC events:", dtc.shape[0])

    # Example: build a simple feature matrix for machine learning.
    feature_frame = obd[[
        "engine_rpm",
        "vehicle_speed_kmh",
        "coolant_temperature_C",
        "throttle_position_pct",
        "engine_load_pct",
        "fuel_level_pct",
        "battery_voltage",
        "mass_airflow_gps",
        "fault_label",
    ]].copy()

    print(feature_frame.head())


if __name__ == "__main__":
    main()
