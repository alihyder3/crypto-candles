import numpy as np
from src.phase1.windowing import compute_num_windows, build_windows

def test_compute_num_windows_basic():
    n = compute_num_windows(n_rows=120, lookback=100, horizon=12, step=1)
    # 120 - 100 - 12 + 1 = 9
    assert n == 9

def test_compute_num_windows_step():
    n = compute_num_windows(n_rows=120, lookback=100, horizon=12, step=5)
    # usable=9 -> windows = 1 + (9-1)//5 = 2
    assert n == 2

def test_exact_fit_single_window():
    n = compute_num_windows(n_rows=112, lookback=100, horizon=12, step=1)
    assert n == 1
    feats = np.arange(112*2, dtype=np.float32).reshape(112, 2) / 1000.0
    target = np.arange(112, dtype=np.float32) / 1000.0
    X, y = build_windows(feats, target, lookback=100, horizon=12, step=1)
    assert X.shape == (1, 100, 2)
    assert y.shape == (1, 12)

def test_not_enough_rows():
    assert compute_num_windows(100, 100, 12, 1) == 0
    feats = np.zeros((100, 3), dtype=np.float32)
    target = np.zeros((100,), dtype=np.float32)
    X, y = build_windows(feats, target, 100, 12, 1)
    assert X.shape[0] == 0 and y.shape[0] == 0

def test_window_values_correct():
    T, F = 130, 1
    feats = np.arange(T * F, dtype=np.float32).reshape(T, F)  # [[0],[1],[2],...]
    target = np.arange(T, dtype=np.float32)
    L, H = 100, 12
    X, y = build_windows(feats, target, L, H, step=1)
    # Check first window
    assert np.allclose(X[0, :, 0], np.arange(0, L))
    assert np.allclose(y[0, :], np.arange(L, L + H))
    # Check last window alignment
    n = 130 - L - H + 1  # 19
    assert X.shape[0] == n
    assert np.allclose(X[-1, :, 0], np.arange(18, 18 + L))  # last start = 18
    assert np.allclose(y[-1, :], np.arange(118, 118 + H))
