"""Multivariate covariance module — the portfolio-risk pillar (daily data, no intraday).
Forecasts the asset covariance matrix and evaluates it the rigorous way: whichever
covariance gives the lowest-variance GLOBAL MINIMUM-VARIANCE portfolio out-of-sample is
the best forecast (Engle-Colacito criterion). Compares sample, EWMA (RiskMetrics),
and Ledoit-Wolf SHRINKAGE (reduces estimation error — critical in high dimension).
Enables portfolio VaR/optimization the univariate vol + connectedness modules cannot.
Causal (rolling/recursive), net-of-nothing (GMV variance is the pure covariance test).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_broad import load_broad

ANN = 252
WIN = 252
NAMES = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
         "GLD", "USO", "DBC", "EURUSD", "USDJPY", "AUDUSD"]


def gmv_weights(S):
    """Global minimum-variance weights w = S^-1 1 / (1' S^-1 1), long-short, sum=1."""
    n = S.shape[0]
    try:
        inv = np.linalg.inv(S + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        return np.ones(n) / n
    one = np.ones(n); w = inv @ one / (one @ inv @ one)
    return w


def main():
    d = load_broad()
    names = [n for n in NAMES if n in d]
    ret = pd.DataFrame({n: np.log(d[n]["adjclose"]).diff() for n in names}).dropna()
    ret = ret[ret.index >= "2010-01-01"]
    R = ret.to_numpy(); dates = ret.index; N = len(names)

    # forecasters produce Sigma_t from data < t; GMV weights held to next rebalance (monthly)
    reb = pd.to_datetime(pd.Series(dates).dt.to_period("M").astype(str)).drop_duplicates().index
    methods = ["equal-weight", "sample", "EWMA", "Ledoit-Wolf"]
    port = {m: [] for m in methods}
    ew_cov = None
    for t in range(WIN, len(R)):
        if t == WIN or (dates[t].month != dates[t - 1].month):    # rebalance monthly
            past = R[:t]
            S_sample = np.cov(past[-WIN:].T)
            S_lw = LedoitWolf().fit(past[-WIN:]).covariance_
            # EWMA covariance (RiskMetrics lambda=0.94), recursive
            lam = 0.94; S_ewma = np.cov(past[:WIN].T)
            for r in past[WIN:]:
                S_ewma = lam * S_ewma + (1 - lam) * np.outer(r, r)
            w = {"equal-weight": np.ones(N) / N, "sample": gmv_weights(S_sample),
                 "EWMA": gmv_weights(S_ewma), "Ledoit-Wolf": gmv_weights(S_lw)}
        for m in methods:
            port[m].append(w[m] @ R[t])

    print(f"multivariate covariance — OOS global min-variance portfolio ({dates[WIN].date()}→{dates[-1].date()})\n")
    print(f"{'covariance':>14} {'GMV realized vol':>18} {'ann Sharpe':>11} {'max|weight|':>11}")
    for m in methods:
        p = np.array(port[m]); vol = p.std() * np.sqrt(ANN)
        sharpe = p.mean() / p.std() * np.sqrt(ANN)
        print(f"{m:>14} {vol*100:>16.2f}% {sharpe:>11.2f} {'—' if m=='equal-weight' else '':>11}")
    best = min(("sample", "EWMA", "Ledoit-Wolf"), key=lambda m: np.array(port[m]).std())
    ewv = np.array(port["equal-weight"]).std() * np.sqrt(ANN) * 100
    bv = np.array(port[best]).std() * np.sqrt(ANN) * 100
    print(f"\n  best covariance forecast: {best} (GMV vol {bv:.2f}% vs equal-weight {ewv:.2f}% — "
          f"{(1-bv/ewv)*100:.0f}% lower risk from covariance-optimized weights)")
    print("  → enables portfolio VaR, minimum-variance & risk-parity allocation. Coherent with EVT-ES.")


if __name__ == "__main__":
    main()
