from __future__ import annotations
import time
from typing import List, Optional, Dict
import pandas as pd
from binance.client import Client

# Map friendly interval strings to python-binance constants
INTERVAL_MAP: Dict[str, str] = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "3d": Client.KLINE_INTERVAL_3DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH,
}

def get_client(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> Client:
    """
    Public market data does not require keys; pass None for both by default.
    """
    return Client(api_key=api_key, api_secret=api_secret)

def _to_dataframe(klines: List[List]) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    # Types
    for c in ["open", "high", "low", "close", "volume",
              "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["open_time", "close_time", "number_of_trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")

    # Datetime convenience
    df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def fetch_ohlcv(symbol: str, interval: str, limit: int = 1000, sleep_sec: float = 0.2) -> pd.DataFrame:
    """
    Fetch up to `limit` most recent klines for symbol@interval.
    Handles pagination using `endTime` to go back in time (Binance max per call = 1000).
    Returns ascending by open_time.
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {sorted(INTERVAL_MAP.keys())}")

    client = get_client()
    interval_const = INTERVAL_MAP[interval]

    remaining = int(limit)
    end_time: Optional[int] = None
    frames: List[pd.DataFrame] = []

    while remaining > 0:
        batch = min(1000, remaining)
        kl = client.get_klines(symbol=symbol, interval=interval_const, limit=batch, endTime=end_time)
        if not kl:
            break
        df = _to_dataframe(kl)
        frames.append(df)
        remaining -= len(df)
        # Next page: everything before earliest open_time in this batch
        end_time = int(df["open_time"].iloc[0]) - 1
        time.sleep(sleep_sec)  # gentle rate-limit

        # Safety: if Binance returns fewer rows than requested repeatedly, stop after one more loop
        if len(df) < batch and remaining > 0 and end_time is not None:
            # try one more fetch
            kl2 = client.get_klines(symbol=symbol, interval=interval_const, limit=min(1000, remaining), endTime=end_time)
            if not kl2:
                break
            df2 = _to_dataframe(kl2)
            frames.append(df2)
            remaining -= len(df2)
            end_time = int(df2["open_time"].iloc[0]) - 1
            time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame(columns=[
            "open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore",
            "open_time_dt","close_time_dt"
        ])

    out = pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    # Keep only the most recent `limit` rows in case we over-fetched
    if len(out) > limit:
        out = out.iloc[-limit:].reset_index(drop=True)
    out["symbol"] = symbol
    out["interval"] = interval
    return out
