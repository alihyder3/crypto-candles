from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd

from src.phase1.scaler import load_scaler, transform_features, transform_target_series, DEFAULT_FEATURES, DEFAULT_TARGET
from src.phase1.windowing import build_windows

def parse_args():
    p = argparse.ArgumentParser(description="Build sliding-window dataset from CSV + scaler.")
    p.add_argument("--csv", required=True, help="Path to raw OHLCV CSV (from M1).")
    p.add_argument("--scaler", required=True, help="Path to scaler JSON (from M1).")
    p.add_argument("--lookback", type=int, default=100)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--step", type=int, default=1, help="Stride between consecutive windows (default 1).")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of windows for validation (tail).")
    p.add_argument("--out", required=True, help="Output .npz path to save arrays and metadata.")
    return p.parse_args()

def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # Load data
    df = pd.read_csv(a.csv)
    # Ensure ascending by open_time
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)
    # Basic columns check
    missing = [c for c in (DEFAULT_FEATURES + [DEFAULT_TARGET]) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    # Load scaler and transform
    scaler = load_scaler(a.scaler)
    feat_norm = transform_features(df[DEFAULT_FEATURES], scaler).to_numpy(dtype=np.float32)
    target_norm = transform_target_series(df[DEFAULT_TARGET], scaler).to_numpy(dtype=np.float32)

    # Build windows
    X, y = build_windows(feat_norm, target_norm, a.lookback, a.horizon, step=a.step)

    if X.shape[0] == 0:
        raise SystemExit("Not enough rows to build any window. Increase --limit in M1 or reduce lookback/horizon.")

    # Time-based split (train = head, val = tail)
    n_total = X.shape[0]
    n_train = max(1, int(round(n_total * (1.0 - a.val_ratio))))
    n_train = min(n_train, n_total - 1)  # keep at least 1 val window

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Metadata
    meta = dict(
        csv=os.path.abspath(a.csv),
        scaler=os.path.abspath(a.scaler),
        lookback=a.lookback,
        horizon=a.horizon,
        step=a.step,
        features=DEFAULT_FEATURES,
        target=DEFAULT_TARGET,
        n_total=int(n_total),
        n_train=int(X_train.shape[0]),
        n_val=int(X_val.shape[0]),
    )

    # Save
    np.savez_compressed(a.out,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, meta=np.array([meta], dtype=object)
    )

    # Report
    print("[OK] Windows built and saved.")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  Saved -> {a.out}")

if __name__ == "__main__":
    main()
