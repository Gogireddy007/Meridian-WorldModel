"""Windowing for the Meridian neural core, with leakage-safe scaling.

Builds, per asset, overlapping windows of length L:
  x_ctx = features[t-L+1 : t+1]        (known at close of t)
  x_fut = features[t-L+2 : t+2]        (JEPA target window, ends at t+1)
  y     = log RV_{t+1}                 (frame column 'y' at row t)

Scaling stats are ALWAYS computed on training rows only and passed in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["har_d", "har_w", "har_m", "log_rv", "ret", "ret_abs",
            "ret_5", "rv_cc", "vix", "term", "vix_chg",
            "nsv_d", "nsv_w", "nsv_m", "psv_d"]

# multi-horizon targets, in addition to the primary h=1 target "y"
AUX_HORIZONS = ["y_5", "y_22"]


def asset_matrix(frame: pd.DataFrame):
    """Return (dates, F matrix, y vector, aux target dict) with rows having full
    features + all targets (h=1, h=5, h=22) finite.

    The last rows (y is NaN near the end of history) are kept for inference-only
    use via a separate path; here we drop non-finite rows for training/eval
    windowing, requiring every horizon target to be finite so ctx/fut/y/y_5/y_22
    all stay row-aligned.
    """
    cols = FEATURES + ["y"] + AUX_HORIZONS
    df = frame[cols].copy()
    finite = np.isfinite(df.to_numpy()).all(1)
    df = df[finite]
    aux = {h: df[h].to_numpy(np.float32) for h in AUX_HORIZONS}
    return df.index, df[FEATURES].to_numpy(np.float32), df["y"].to_numpy(np.float32), aux


def build_windows(dates, X, y, L: int, mean, std, aux=None):
    """Vectorized windowing. Returns dict of arrays (ctx, fut, y, y_5, y_22, date_idx).

    Needs t from L-1 .. len-2 so that a future window (ending t+1) exists.
    """
    Xs = (X - mean) / std
    n = len(Xs)
    idx = np.arange(L - 1, n - 1)                 # valid anchor positions t
    if len(idx) == 0:
        return None
    # context windows end at t
    ctx = np.stack([Xs[i - L + 1: i + 1] for i in idx])      # (N, L, F)
    fut = np.stack([Xs[i - L + 2: i + 2] for i in idx])      # (N, L, F) ends t+1
    yy = y[idx]                                              # log RV_{t+1}
    d = dates[idx]
    out = {"ctx": ctx.astype(np.float32), "fut": fut.astype(np.float32),
           "y": yy.astype(np.float32), "dates": d}
    if aux is not None:
        for h, av in aux.items():
            out[h] = av[idx].astype(np.float32)
    return out


def train_scaler(frames: dict[str, pd.DataFrame], asset_data, end_date):
    """Feature mean/std computed on all assets' rows strictly before end_date."""
    chunks = []
    for a, (dates, X, y, aux) in asset_data.items():
        m = dates < end_date
        if m.any():
            chunks.append(X[m])
    allX = np.concatenate(chunks, 0)
    mean = allX.mean(0)
    std = allX.std(0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)
