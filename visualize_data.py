"""Generate simple plots for the synthetic vehicle diagnostics dataset."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("visualizations")


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all generated CSV files into memory."""
    metadata = pd.read_csv(DATA_DIR / "vehicle_metadata.csv")
    obd = pd.read_csv(DATA_DIR / "obd_timeseries.csv", parse_dates=["timestamp"])
    faults = pd.read_csv(DATA_DIR / "fault_windows.csv", parse_dates=["start_timestamp", "end_timestamp"])
    dtc = pd.read_csv(DATA_DIR / "dtc_logs.csv", parse_dates=["timestamp"])
    return metadata, obd, faults, dtc


def plot_metadata_distribution(metadata: pd.DataFrame) -> None:
    """Plot mileage, age, and model distribution summaries."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metadata["mileage_km"].hist(bins=20, ax=axes[0], color="#4c78a8")
    axes[0].set_title("Mileage distribution")
    axes[0].set_xlabel("Mileage (km)")
    axes[0].set_ylabel("Vehicles")

    metadata["age_years"].hist(bins=15, ax=axes[1], color="#72b7b2")
    axes[1].set_title("Age distribution")
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Vehicles")

    metadata["model"].value_counts().plot(kind="bar", ax=axes[2], color=["#f58518", "#54a24b", "#b279a2"])
    axes[2].set_title("Model mix")
    axes[2].set_xlabel("Model")
    axes[2].set_ylabel("Vehicles")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "vehicle_metadata_summary.png", dpi=150)
    plt.close(fig)


def plot_fault_summary(faults: pd.DataFrame, obd: pd.DataFrame) -> None:
    """Plot fault-window counts and label distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    faults["fault_type"].value_counts().reindex(["sensor_drift", "spike", "stuck_at", "noise"], fill_value=0).plot(
        kind="bar", ax=axes[0], color=["#f58518", "#54a24b", "#b279a2", "#e45756"]
    )
    axes[0].set_title("Injected fault types")
    axes[0].set_xlabel("Fault type")
    axes[0].set_ylabel("Number of faults")

    label_counts = obd["fault_label"].value_counts().sort_index()
    label_counts.plot(kind="bar", ax=axes[1], color="#4c78a8")
    axes[1].set_title("Fault label distribution")
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Records")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fault_summary.png", dpi=150)
    plt.close(fig)


def plot_sample_vehicle_timeseries(obd: pd.DataFrame) -> None:
    """Plot a sample vehicle's signals over the first 30 minutes of operation."""
    sample_vehicle = obd["vehicle_id"].iloc[0]
    sample_df = obd[obd["vehicle_id"] == sample_vehicle].head(1800).copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    sample_df.set_index("timestamp")["engine_rpm"].plot(ax=axes[0], color="#4c78a8", label="RPM")
    sample_df.set_index("timestamp")["vehicle_speed_kmh"].plot(ax=axes[0], color="#72b7b2", label="Speed (km/h)")
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"Sample diagnostics for vehicle {sample_vehicle}")
    axes[0].legend()

    sample_df.set_index("timestamp")["coolant_temperature_C"].plot(ax=axes[1], color="#f58518", label="Coolant °C")
    sample_df.set_index("timestamp")["battery_voltage"].plot(ax=axes[1], color="#e45756", label="Battery V")
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_vehicle_timeseries.png", dpi=150)
    plt.close(fig)


def plot_dtc_distribution(dtc: pd.DataFrame) -> None:
    """Plot the DTC code distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    dtc["dtc_code"].value_counts().sort_values(ascending=False).plot(kind="bar", ax=ax, color="#54a24b")
    ax.set_title("DTC event distribution")
    ax.set_xlabel("DTC code")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dtc_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate all requested visualizations and save them to the visualizations folder."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    metadata, obd, faults, dtc = load_datasets()

    plot_metadata_distribution(metadata)
    plot_fault_summary(faults, obd)
    plot_sample_vehicle_timeseries(obd)
    plot_dtc_distribution(dtc)

    print(f"Saved plots to {OUTPUT_DIR.resolve()}")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
