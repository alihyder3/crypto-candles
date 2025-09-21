from __future__ import annotations
from typing import Tuple
import numpy as np

def compute_num_windows(n_rows: int, lookback: int, horizon: int, step: int = 1) -> int:
    if lookback <= 0 or horizon <= 0 or step <= 0:
        raise ValueError("lookback, horizon, and step must be positive.")
    usable = n_rows - lookback - horizon + 1
    if usable <= 0:
        return 0
    return 1 + (usable - 1) // step

def build_windows(
    features: np.ndarray,  # shape [T, F] normalized features
    target: np.ndarray,    # shape [T] normalized target series
    lookback: int,
    horizon: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: [N, lookback, F]
      y: [N, horizon]
    """
    if features.ndim != 2:
        raise ValueError("features must be 2D [T, F]")
    if target.ndim != 1:
        raise ValueError("target must be 1D [T]")
    if features.shape[0] != target.shape[0]:
        raise ValueError("features and target must have same length")

    T, F = features.shape
    N = compute_num_windows(T, lookback, horizon, step)
    if N == 0:
        return (np.empty((0, lookback, F), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32))

    X = np.empty((N, lookback, F), dtype=np.float32)
    y = np.empty((N, horizon), dtype=np.float32)

    idx = 0
    last_start = T - lookback - horizon
    for start in range(0, last_start + 1, step):
        end = start + lookback
        X[idx] = features[start:end]
        y[idx] = target[end:end + horizon]
        idx += 1

    return X, y
