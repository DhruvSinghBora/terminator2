"""Simple Streamlit app for exploring the synthetic vehicle diagnostics dataset.

This viewer keeps the interface intentionally lightweight:
- select a vehicle
- inspect its diagnostic signals over time
- show flagged fault windows and DTC events
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data")

DTC_FAMILY_MAP = {
    "P0117": "Thermal sensor / coolant",
    "P0562": "Electrical / battery",
    "P0300": "Powertrain / misfire",
    "P0500": "Speed / sensor",
    "P0101": "Airflow / load",
    "P0102": "Airflow / intake",
}


@st.cache_data(show_spinner=False)
def load_datasets():
    """Load the generated datasets into memory for the Streamlit UI."""
    metadata = pd.read_csv(DATA_DIR / "vehicle_metadata.csv")
    obd = pd.read_csv(DATA_DIR / "obd_timeseries.csv", parse_dates=["timestamp"])
    faults = pd.read_csv(DATA_DIR / "fault_windows.csv", parse_dates=["start_timestamp", "end_timestamp"])
    dtc = pd.read_csv(DATA_DIR / "dtc_logs.csv", parse_dates=["timestamp"])
    anomaly_windows = pd.read_csv(DATA_DIR / "anomaly_baselines.csv", parse_dates=["window_start", "window_end"])
    return metadata, obd, faults, dtc, anomaly_windows


def main() -> None:
    """Render the interactive diagnostic dashboard."""
    st.set_page_config(page_title="Vehicle Diagnostics Viewer", layout="wide")
    st.title("Vehicle Diagnostics Time-Series Viewer")
    st.caption("Simple, model-backed inspection of synthetic vehicle faults and OBD signals.")

    metadata, obd, faults, dtc, anomaly_windows = load_datasets()

    col1, col2, col3 = st.columns(3)
    col1.metric("Vehicles", int(metadata["vehicle_id"].nunique()))
    col2.metric("Operational records", int(len(obd)))
    col3.metric("Fault windows", int(len(faults)))

    vehicle_id = st.selectbox("Select vehicle", sorted(metadata["vehicle_id"].unique().tolist()))

    vehicle_df = obd[obd["vehicle_id"] == vehicle_id].sort_values("timestamp").reset_index(drop=True)
    vehicle_faults = faults[faults["vehicle_id"] == vehicle_id].sort_values("start_timestamp").reset_index(drop=True)
    vehicle_dtc = dtc[dtc["vehicle_id"] == vehicle_id].sort_values("timestamp").reset_index(drop=True)
    vehicle_anomalies = anomaly_windows[anomaly_windows["vehicle_id"] == vehicle_id].sort_values("window_start").reset_index(drop=True)

    st.subheader("Anomaly window summary")
    anomaly_table = vehicle_anomalies[["window_start", "window_end", "rolling_zscore_mean", "anomaly_score", "top_signals", "likely_subsystem", "anomaly_flag"]].copy()
    anomaly_table["anomaly_score"] = anomaly_table["anomaly_score"].round(3)
    anomaly_table["rolling_zscore_mean"] = anomaly_table["rolling_zscore_mean"].round(3)
    st.dataframe(anomaly_table, use_container_width=True)

    if not vehicle_anomalies.empty:
        anomaly_choice = st.selectbox("Inspect an anomaly window", vehicle_anomalies.index.tolist(), format_func=lambda idx: f"{vehicle_anomalies.loc[idx, 'window_start']} to {vehicle_anomalies.loc[idx, 'window_end']} (score {vehicle_anomalies.loc[idx, 'anomaly_score']:.3f})")
        selected_window = vehicle_anomalies.loc[anomaly_choice]
        start = selected_window["window_start"] - pd.Timedelta(minutes=2)
        end = selected_window["window_end"] + pd.Timedelta(minutes=2)
        window_df = vehicle_df[(vehicle_df["timestamp"] >= start) & (vehicle_df["timestamp"] <= end)].copy()

        st.write("Selected anomaly window score:", round(float(selected_window["anomaly_score"]), 3))
        st.write("Likely subsystem:", selected_window["likely_subsystem"])
        st.write("Top signal contributors:", selected_window["top_signals"])

        st.line_chart(window_df.set_index("timestamp")[["engine_rpm", "vehicle_speed_kmh", "coolant_temperature_C", "battery_voltage"]])

        dtc_families = []
        for _, event in vehicle_dtc.iterrows():
            family = DTC_FAMILY_MAP.get(str(event["dtc_code"]), "General diagnostics")
            dtc_families.append(f"{event['dtc_code']} — {family}")
        if dtc_families:
            st.caption("DTC family hints: " + " | ".join(dtc_families))
    else:
        st.info("No anomaly windows were flagged for this vehicle in the current baseline run.")

    st.subheader("Signal overview")
    signal_options = [
        "engine_rpm",
        "vehicle_speed_kmh",
        "coolant_temperature_C",
        "throttle_position_pct",
        "engine_load_pct",
        "battery_voltage",
        "mass_airflow_gps",
    ]
    selected_signals = st.multiselect("Choose signals to display", signal_options, default=signal_options[:4])

    if vehicle_df.empty:
        st.warning("No operational records found for this vehicle.")
        return

    line_chart = vehicle_df.set_index("timestamp")[[*selected_signals]].copy()
    st.line_chart(line_chart)

    st.subheader("Fault windows and DTC events")
    st.dataframe(vehicle_faults, use_container_width=True)
    st.dataframe(vehicle_dtc, use_container_width=True)

    st.subheader("Vehicle metadata")
    st.dataframe(metadata[metadata["vehicle_id"] == vehicle_id], use_container_width=True)

    st.caption("This viewer intentionally keeps the UI small and focused on the time-series evidence.")


if __name__ == "__main__":
    main()
