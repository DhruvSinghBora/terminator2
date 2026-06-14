# Synthetic Vehicle Predictive Maintenance Dataset

This project generates a synthetic vehicle diagnostics dataset for machine-learning experiments and fault-detection research.

## What is included

The generator writes the following CSV files to the data/ folder:

- vehicle_metadata.csv
- obd_timeseries.csv
- fault_windows.csv
- dtc_logs.csv

## Dataset structure

### 1. Vehicle metadata

vehicle_metadata.csv contains one row per vehicle with:

- vehicle_id
- model
- mileage_km
- age_years
- fuel_type

### 2. Operational behavior data

obd_timeseries.csv contains 1 Hz multivariate diagnostics for each vehicle. The signal set includes:

- timestamp
- vehicle_id
- engine_rpm
- vehicle_speed_kmh
- coolant_temperature_C
- throttle_position_pct
- engine_load_pct
- fuel_level_pct
- battery_voltage
- mass_airflow_gps
- fault_label

The fault_label column uses the following values:

- healthy
- sensor_drift
- spike
- stuck_at
- dropout
- noise

### 3. Fault windows

fault_windows.csv contains one row per injected fault with:

- vehicle_id
- fault_id
- fault_type
- affected_signal
- start_timestamp
- end_timestamp

### 4. DTC event logs

dtc_logs.csv maps each fault window to a diagnostic code, for example:

- sensor_drift -> P0117 / P0562
- spike -> P0300
- stuck_at -> P0500
- noise -> P0101

## Fault injection methodology

The generator creates healthy driving traces and injects synthetic faults into selected windows:

- sensor_drift: gradual increase or decrease over time on coolant temperature or battery voltage
- spike: short abnormal peaks on engine RPM or mass airflow
- stuck_at: frozen signal values on speed or throttle position
- dropout: intermittent signal suppression to mimic missing or zeroed sensor values
- noise: increased variance on engine load

Fault windows are injected into otherwise healthy traces using a fixed random seed, with 1-3 fault events per vehicle, 60-600 second durations, and non-overlapping windows for each vehicle. This follows the prompt's evaluation recipe for synthetic fault injection and keeps the healthy portion of the dataset above 90%.

To summarize which signals drove each flagged interval, run:

```bash
python attribution_summary.py
```

This writes a lightweight attribution report to [data/fault_attribution_summary.csv](data/fault_attribution_summary.csv) using per-signal deviation scores rather than a heavier SHAP pipeline.

## Example loading code

```python
import pandas as pd

metadata = pd.read_csv("data/vehicle_metadata.csv")
obd = pd.read_csv("data/obd_timeseries.csv", parse_dates=["timestamp"])
fault_windows = pd.read_csv("data/fault_windows.csv", parse_dates=["start_timestamp", "end_timestamp"])
dtc_logs = pd.read_csv("data/dtc_logs.csv", parse_dates=["timestamp"])

print(metadata.head())
print(obd.head())
print("Unique fault labels:", sorted(obd["fault_label"].unique().tolist()))
```

## Reproducibility

The dataset uses a fixed random seed so the generated samples are repeatable.

## Running the generator

```bash
python generate_dataset.py
```

## Driving-mode baseline analysis

A simple mode-aware baseline workflow is included in [mode_baselines.py](mode_baselines.py):

```bash
python mode_baselines.py
```

This script classifies each record as idle, city, or highway using the speed/RPM heuristic from the project prompt and writes a summary table to [data/mode_baselines.csv](data/mode_baselines.csv).

## Anomaly-detection baselines

The project also includes [anomaly_baselines.py](anomaly_baselines.py), which creates two baseline detectors for comparison:

- rolling z-score on windowed signal statistics
- Isolation Forest on 60-second aggregated windows

Run it with:

```bash
python anomaly_baselines.py
```

The output is written to [data/anomaly_baselines.csv](data/anomaly_baselines.csv).

## Simple viewer

A lightweight Streamlit viewer is included in [app.py](app.py). It lets you inspect the generated diagnostics data, click through vehicle records, and review fault windows and DTC events without a heavy dashboard.

Run it with:

```bash
streamlit run app.py
```
