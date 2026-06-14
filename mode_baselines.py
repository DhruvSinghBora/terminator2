"""Build per-driving-mode baselines from the generated vehicle diagnostics data.

The baseline model uses a simple speed/RPM classifier to assign each record to
one of three operating modes:
- idle
- city
- highway

It then computes summary statistics per mode and writes them to the data/
folder so anomaly detection can compare current behavior against mode-aware
reference distributions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OBD_FILE = DATA_DIR / "obd_timeseries.csv"
BASELINE_FILE = DATA_DIR / "mode_baselines.csv"

SIGNAL_COLUMNS = [
    "engine_rpm",
    "vehicle_speed_kmh",
    "coolant_temperature_C",
    "throttle_position_pct",
    "engine_load_pct",
    "mass_airflow_gps",
]


def classify_driving_mode(row: pd.Series) -> str:
    """Assign one of the three driving modes using simple speed/RPM rules.

    The heuristic is intentionally lightweight and interpretable:
    - idle: very low speed and low engine speed
    - highway: high speed or high RPM
    - city: all other operating points
    """
    speed_kmh = float(row["vehicle_speed_kmh"])
    rpm = float(row["engine_rpm"])

    if speed_kmh < 8.0 and rpm < 1100.0:
        return "idle"
    if speed_kmh >= 70.0 or rpm >= 2600.0:
        return "highway"
    return "city"


def compute_mode_baselines(obd_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-mode summary statistics for key vehicle signals."""
    annotated = obd_df.copy()
    annotated["driving_mode"] = annotated.apply(classify_driving_mode, axis=1)

    records = []
    for mode in ("idle", "city", "highway"):
        mode_df = annotated[annotated["driving_mode"] == mode]
        if mode_df.empty:
            continue

        for signal in SIGNAL_COLUMNS:
            series = mode_df[signal].astype(float)
            records.append(
                {
                    "driving_mode": mode,
                    "signal": signal,
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    """Generate mode-aware baseline summary statistics and save them to disk."""
    if not OBD_FILE.exists():
        raise FileNotFoundError(f"Expected operational dataset at {OBD_FILE}")

    obd_df = pd.read_csv(OBD_FILE, parse_dates=["timestamp"])
    baseline_df = compute_mode_baselines(obd_df)
    baseline_df.to_csv(BASELINE_FILE, index=False)

    print("Mode-aware baselines generated successfully.")
    print(f"Saved to: {BASELINE_FILE.resolve()}")
    print("Mode summary rows:", len(baseline_df))
    print("Modes covered:", sorted(baseline_df["driving_mode"].unique().tolist()))


if __name__ == "__main__":
    main()
