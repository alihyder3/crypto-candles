from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

from src.phase1.binance_client import fetch_ohlcv
from src.phase1.scaler import (
    fit_minmax, transform_features, save_scaler, DEFAULT_FEATURES, DEFAULT_TARGET
)

def parse_args():
    p = argparse.ArgumentParser(description="Fetch OHLCV from Binance and (optionally) min-max normalize.")
    p.add_argument("--symbol", required=True, help="e.g., BTCUSDT")
    p.add_argument("--interval", required=True, help="e.g., 1m, 5m, 15m, 1h, 4h, 1d")
    p.add_argument("--limit", type=int, default=1000, help="Number of most recent candles (supports >1000 via pagination)")
    p.add_argument("--save-csv", default=None, help="Optional path to save raw OHLCV CSV (e.g., data/raw/BTCUSDT_1h.csv)")
    p.add_argument("--normalize", action="store_true", help="Compute scaler and print normalized head()")
    p.add_argument("--scaler-out", default=None, help="Where to save scaler JSON (e.g., artifacts/scalers/BTCUSDT_1h_scaler.json)")
    return p.parse_args()

def main():
    args = parse_args()

    print(f"[INFO] Fetching {args.symbol} {args.interval} last {args.limit} klines...")
    df = fetch_ohlcv(args.symbol, args.interval, args.limit)

    if df.empty:
        print("[ERROR] No data returned. Check symbol/interval and network.")
        sys.exit(2)

    print(f"[OK] Fetched shape: {df.shape}")
    print(df[["open_time_dt","open","high","low","close","volume"]].tail(3).to_string(index=False))

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print(f"[OK] Saved raw CSV -> {args.save_csv}")

    if args.normalize:
        print("[INFO] Computing Min–Max scaler for features and target...")
        scaler = fit_minmax(
            df,
            feature_cols=DEFAULT_FEATURES,
            target_col=DEFAULT_TARGET,
            meta={"symbol": args.symbol, "interval": args.interval}
        )
        norm_df = transform_features(df[DEFAULT_FEATURES], scaler)
        print("[OK] Normalized features head():")
        print(norm_df.head().to_string(index=False))

        if args.scaler_out:
            os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)
            save_scaler(args.scaler_out, scaler)
            print(f"[OK] Scaler saved -> {args.scaler_out}")

        # Sanity: values in [0,1]
        mins = norm_df.min().min()
        maxs = norm_df.max().max()
        print(f"[CHECK] Normalized min≈{mins:.4f}, max≈{maxs:.4f} (expected within [0,1])")

if __name__ == "__main__":
    main()
