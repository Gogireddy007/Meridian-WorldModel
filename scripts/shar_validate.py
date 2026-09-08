"""Does the SEMIVARIANCE (leverage / bad-vol) channel beat plain HAR out-of-sample?

The 104-agent research pass flagged realized-semivariance (Patton-Sheppard 2015; Bollerslev
2022 SHAR) as the single best-evidenced daily-feasible refinement: "bad" volatility (variance
from DOWN moves, RS-) predicts future vol far more strongly than "good" (RS+). Daily proxy:
split each day's squared return by its sign; the daily RV term becomes two terms (RS+, RS-),
weekly/monthly stay total. We test HAR vs SHAR OOS with QLIKE on the pre-registered universe,
causal expanding fit. Only wire SHAR into the engine if it genuinely wins here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import realized_variance
from meridian.evalproto import qlike

EPS = 1e-12


def oos_qlike(X, y, frac=0.5):
    n = len(y); k = int(n * frac)
    A = np.column_stack([np.ones(k), X[:k]])
    beta, *_ = np.linalg.lstsq(A, y[:k], rcond=None)
    jb = 0.5 * np.var(y[:k] - A @ beta)
    Xte = X[k:]; pred = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    return float(np.nanmean(qlike(np.exp(y[k:]), np.exp(pred + jb))))


def main():
    d = load_all()
    print("HAR vs SHAR (semivariance / bad-vol channel) — OOS QLIKE, causal\n")
    print(f"{'asset':>8} {'HAR':>9} {'SHAR':>9} {'edge':>8}")
    edges = []
    for a, ohlc in d["prices"].items():
        rvf = realized_variance(ohlc)
        rv = rvf["rv"]; ret = rvf["ret"]
        neg = (ret.clip(upper=0) ** 2)                 # bad (down-move) variance proxy
        pos = (ret.clip(lower=0) ** 2)                 # good (up-move) variance proxy
        rv_w = rv.rolling(5).mean(); rv_m = rv.rolling(22).mean()
        y = np.log(rv + EPS).shift(-1)
        har = np.column_stack([np.log(rv + EPS), np.log(rv_w + EPS), np.log(rv_m + EPS)])
        # additive leverage: keep the full robust GK HAR, ADD the bad-vol (down-move) term
        shar = np.column_stack([np.log(rv + EPS), np.log(rv_w + EPS), np.log(rv_m + EPS),
                                np.log(neg + EPS)])
        ok = np.isfinite(y) & np.isfinite(har).all(1) & np.isfinite(shar).all(1)
        yv = y[ok].to_numpy(); harv = har[ok.to_numpy()]; sharv = shar[ok.to_numpy()]
        if len(yv) < 400:
            continue
        qh = oos_qlike(harv, yv); qs = oos_qlike(sharv, yv)
        edge = (qh - qs) / qh * 100; edges.append(edge)
        print(f"{a:>8} {qh:>9.4f} {qs:>9.4f} {edge:>+7.2f}%")
    print(f"\n  mean SHAR edge over HAR: {np.mean(edges):+.2f}%  "
          f"({'SHAR helps → wire into engine' if np.mean(edges) > 0 else 'no gain → keep plain HAR'})")
    print(f"  positive in {sum(e>0 for e in edges)}/{len(edges)} assets")


if __name__ == "__main__":
    main()
