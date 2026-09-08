"""Higher-order implication engine — Diebold-Yilmaz volatility CONNECTEDNESS.

A VAR on cross-asset realized volatility; its GENERALIZED forecast-error-variance
decomposition (Pesaran-Shin, order-invariant) quantifies, to ALL orders over horizon H,
the share of each asset's future-uncertainty caused by a shock to every other asset —
i.e. how an event propagates through the whole system. Outputs the connectedness matrix,
net transmitters/receivers, total systemic connectedness, and its time-variation (a
real systemic-risk gauge that spikes in crises). This is the rigorous 'maximum-order
implications of an event' capability, econometrically grounded — not hype.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_broad import load_broad, ASSET_CLASS

H = 10          # propagation horizon (days)
LAG = 3


def realized_vol_panel(d, names):
    """log realized variance (5-day) per asset — the connectedness input."""
    out = {}
    for n in names:
        r = np.log(d[n]["adjclose"]).diff()
        out[n] = np.log((r ** 2).rolling(5).mean().clip(lower=1e-10))
    return pd.DataFrame(out).dropna()


def generalized_fevd(res, H):
    """Pesaran-Shin generalized FEVD → row-normalized connectedness matrix (N,N)."""
    Sigma = res.sigma_u.to_numpy() if hasattr(res.sigma_u, "to_numpy") else np.asarray(res.sigma_u)
    N = Sigma.shape[0]
    Theta = res.ma_rep(maxn=H)                        # (H+1, N, N) MA coefficients
    sig = np.sqrt(np.diag(Sigma))
    num = np.zeros((N, N)); den = np.zeros(N)
    for h in range(H):
        Th = Theta[h]
        TS = Th @ Sigma
        for i in range(N):
            den[i] += (Th[i] @ Sigma @ Th[i])
            for j in range(N):
                num[i, j] += (TS[i, j] ** 2) / (sig[j] ** 2)
    theta = num / den[:, None]
    theta = theta / theta.sum(axis=1, keepdims=True)   # row-normalize
    return theta


def main():
    d = load_broad()
    names = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
             "GLD", "USO", "DBC", "EURUSD", "USDJPY", "AUDUSD"]
    names = [n for n in names if n in d]
    rv = realized_vol_panel(d, names)
    print(f"connectedness on {len(names)} assets, {rv.index.min().date()}→{rv.index.max().date()}\n")

    res = VAR(rv).fit(LAG)
    C = generalized_fevd(res, H) * 100
    N = len(names)
    frm = C.sum(1) - np.diag(C)                        # received FROM others
    to = C.sum(0) - np.diag(C)                          # transmitted TO others
    net = to - frm
    total = frm.sum() / N

    order = np.argsort(-net)
    print(f"{'asset':>8} {'class':>10} {'TO':>7} {'FROM':>7} {'NET':>7}")
    for i in order:
        print(f"{names[i]:>8} {ASSET_CLASS.get(names[i],'?'):>10} {to[i]:>7.1f} {frm[i]:>7.1f} {net[i]:>+7.1f}")
    print(f"\n  TOTAL system connectedness: {total:.1f}%  (higher = more contagion / systemic risk)")
    print(f"  Top NET transmitters (shock sources): {', '.join(names[i] for i in order[:3])}")
    print(f"  Top NET receivers (shock absorbers):  {', '.join(names[i] for i in order[-3:])}")

    # strongest single propagation pairs (shock j -> asset i)
    Coff = C.copy(); np.fill_diagonal(Coff, 0)
    flat = np.dstack(np.unravel_index(np.argsort(-Coff, axis=None), Coff.shape))[0][:6]
    print("\n  Strongest shock-propagation channels (from → to, % of variance):")
    for i, j in flat:
        print(f"    {names[j]:>7} → {names[i]:<7} {Coff[i, j]:.1f}%")

    # rolling total connectedness (systemic-risk time series)
    win = 250; roll = []
    for t in range(win, len(rv), 10):
        try:
            r = VAR(rv.iloc[t - win:t]).fit(LAG)
            roll.append((rv.index[t], generalized_fevd(r, H).sum() * 100 / N - np.diag(generalized_fevd(r, H) * 100).sum() / N))
        except Exception:
            pass
    rc = pd.Series(dict(roll))
    print(f"\n  rolling connectedness: min {rc.min():.1f}% ({rc.idxmin().date()}), "
          f"max {rc.max():.1f}% ({rc.idxmax().date()}), latest {rc.iloc[-1]:.1f}%")
    rc.to_frame("connectedness").to_parquet(Path(__file__).resolve().parent.parent / "results" / "connectedness.parquet")


if __name__ == "__main__":
    main()
