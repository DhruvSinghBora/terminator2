from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_DIR = Path("data")

PREDICTIONS_FILE = DATA_DIR / "anomaly_baselines.csv"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth_windows.csv"


def main():

    # ---- Load files ----
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(PREDICTIONS_FILE)

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(GROUND_TRUTH_FILE)

    preds = pd.read_csv(PREDICTIONS_FILE, parse_dates=["window_start", "window_end"])
    gt = pd.read_csv(GROUND_TRUTH_FILE, parse_dates=["window_start", "window_end"])

    # ---- Align on window keys ----
    df = preds.merge(
        gt,
        on=["vehicle_id", "window_start", "window_end"],
        how="inner",
    )

    if df.empty:
        raise ValueError("No matching rows between predictions and ground truth")

    # ---- TRUE LABEL ----
    y_true = df["actual_anomaly"].astype(int)

    # ---- PREDICTION (Isolation Forest) ----
    y_pred = df["isolation_label"].fillna(False).astype(int)

    # ---- METRICS ----
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=== Isolation Forest Evaluation ===")
    print(f"Samples   : {len(df)}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()