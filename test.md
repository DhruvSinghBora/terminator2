# Vehicle Diagnostics Anomaly Detection

This project is a starter workspace for the anomaly-detection problem on vehicle diagnostics data, and it now uses Grafana as the primary dashboard layer.

## What is included
- A baseline anomaly-detection pipeline based on rolling statistics and Isolation Forest
- Synthetic sample data generation for quick demos
- A Prometheus-style metrics endpoint for Grafana dashboards
- Grafana provisioning files for a ready-to-import anomaly dashboard

## Quick start
1. Install dependencies:
   python -m pip install -r requirements.txt
2. Generate sample data:
   python data/generate_sample_data.py
3. Start the metrics endpoint:
   python metrics_server.py
4. Launch the full Grafana stack:
   docker compose up --build

Then open:
- Grafana: http://localhost:3000 (admin / admin)
- Prometheus: http://localhost:9090
- Metrics endpoint: http://localhost:8000/metrics

## Project goals
- Flag abnormal windows in multivariate vehicle diagnostics streams
- Surface the main signals contributing to each anomaly
- Provide a human-legible interface for workshop or fleet operators



## Problem 1 — Anomaly Detection on Vehicle Diagnostics Data

**Theme:** Time-series ML, unsupervised/semi-supervised learning
**Persona served:** Workshop diagnostics technician, fleet operations manager, OEM remote diagnostics team

### Problem Description
Modern vehicles continuously emit diagnostics data — OBD-II PIDs (RPM, coolant temperature, battery voltage, fuel trims), CAN bus signals, and DTC event streams. Today, most of this data is only inspected *after* a hard fault occurs. Subtle anomalies — sensor drift, intermittent electrical faults, abnormal thermal or vibration signatures, degrading battery health — go unnoticed until they become breakdowns, comebacks, or warranty claims. There is no scalable way for a workshop or fleet operator to answer: "Which of my vehicles is behaving abnormally right now, and in what way?"

### Inputs
- Multivariate time-series of vehicle diagnostics: OBD-II PID logs and/or CAN signal traces sampled at 1Hz–100Hz (public datasets: HCRL automotive CAN datasets, Kaggle OBD-II driving datasets, NASA battery degradation data)
- DTC event logs with timestamps (can be synthetic)
- Vehicle metadata: model, mileage, age (synthetic)
- Optional: labeled fault windows for a small validation subset (teams may inject synthetic faults — sensor drift, spikes, stuck-at values — into healthy data to create ground truth)

### Expected Outputs
- Anomaly score per time window per vehicle, with detection of anomalous windows in unseen data
- Contributing-signal attribution: which signals drove the anomaly (e.g., "coolant temp rising abnormally relative to RPM and speed")
- Optional mapping of anomaly patterns to likely DTC families or subsystem (thermal, electrical, fuel)
- A time-series viewer UI where a user can select a vehicle, scrub the timeline, and inspect flagged windows with highlighted signals

### Design & Implementation Hints
- Establish per-driving-mode baselines (idle / city / highway) — anomalies are mode-relative; a simple speed/RPM-based mode classifier is enough
- Baselines to beat: rolling z-score and Isolation Forest on windowed statistical features. Stronger: LSTM/Conv autoencoder reconstruction error, or Matrix Profile for discord discovery
- For attribution, per-signal reconstruction error or SHAP on the window features is sufficient — don't over-engineer
- Inject synthetic faults (gradual drift, intermittent dropout, stuck sensor) into clean data for evaluation; document the injection recipe
- Keep the UI simple: Plotly/Streamlit time-series scrubber beats a polished dashboard with no model behind it

### Success Criteria
- Detects ≥80% of injected fault windows at a false-positive rate the team can defend (precision/recall reported on the held-out injected-fault set)
- Attribution is human-legible: a judge can look at a flagged window and understand *why* it was flagged
- Handles at least 5–10 signals simultaneously (genuinely multivariate, not per-signal thresholds)
- Live demo: load an unseen trace, system flags anomalies in seconds

### Stretch Goals
- Anomaly-to-DTC-family classification
- Fleet-level ranking view ("top 10 vehicles by anomaly burden this week")