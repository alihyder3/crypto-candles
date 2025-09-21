from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import json
import math
import pandas as pd

DEFAULT_FEATURES = ["open", "high", "low", "close", "volume"]
DEFAULT_TARGET = "close"

@dataclass
class MinMaxScalerFT:
    feature_mins: Dict[str, float]
    feature_maxs: Dict[str, float]
    target_min: float
    target_max: float
    feature_cols: List[str]
    target_col: str
    version: int = 1
    meta: Dict[str, str] = None  # e.g., {"symbol": "...", "interval": "..."}

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "MinMaxScalerFT":
        d = json.loads(s)
        return MinMaxScalerFT(
            feature_mins=d["feature_mins"],
            feature_maxs=d["feature_maxs"],
            target_min=d["target_min"],
            target_max=d["target_max"],
            feature_cols=d["feature_cols"],
            target_col=d["target_col"],
            version=d.get("version", 1),
            meta=d.get("meta"),
        )

def _safe_scale(x: float, mn: float, mx: float) -> float:
    denom = (mx - mn)
    if denom == 0 or math.isclose(denom, 0.0):
        return 0.0  # constant column -> map to 0
    return (x - mn) / denom

def _safe_inverse(x: float, mn: float, mx: float) -> float:
    return x * (mx - mn) + mn

def fit_minmax(df: pd.DataFrame, feature_cols: List[str] = None, target_col: str = None,
               meta: Dict[str, str] = None) -> MinMaxScalerFT:
    feature_cols = feature_cols or DEFAULT_FEATURES
    target_col = target_col or DEFAULT_TARGET

    fmins = {c: float(df[c].min()) for c in feature_cols}
    fmaxs = {c: float(df[c].max()) for c in feature_cols}
    tmin = float(df[target_col].min())
    tmax = float(df[target_col].max())

    return MinMaxScalerFT(
        feature_mins=fmins,
        feature_maxs=fmaxs,
        target_min=tmin,
        target_max=tmax,
        feature_cols=feature_cols,
        target_col=target_col,
        meta=meta or {},
    )

def transform_features(df: pd.DataFrame, scaler: MinMaxScalerFT) -> pd.DataFrame:
    out = df.copy()
    for c in scaler.feature_cols:
        out[c] = out[c].apply(lambda v: _safe_scale(float(v), scaler.feature_mins[c], scaler.feature_maxs[c]))
    return out

def transform_target_series(series: pd.Series, scaler: MinMaxScalerFT) -> pd.Series:
    return series.apply(lambda v: _safe_scale(float(v), scaler.target_min, scaler.target_max))

def inverse_target_series(series: pd.Series, scaler: MinMaxScalerFT) -> pd.Series:
    return series.apply(lambda v: _safe_inverse(float(v), scaler.target_min, scaler.target_max))

def save_scaler(path: str, scaler: MinMaxScalerFT) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(scaler.to_json())

def load_scaler(path: str) -> MinMaxScalerFT:
    with open(path, "r", encoding="utf-8") as f:
        return MinMaxScalerFT.from_json(f.read())
