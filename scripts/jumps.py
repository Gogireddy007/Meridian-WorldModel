"""Jump/continuous decomposition (Barndorff-Nielsen-Shephard) on real intraday (hourly)
data, and HAR-CJ vs HAR-RV. Realized Variance RV = sum r_i^2 captures TOTAL variation;
Bipower Variation BV = (pi/2) sum|r_i||r_{i-1}| captures only the CONTINUOUS part; the
JUMP component J = max(RV-BV, 0). HAR-CJ forecasts RV from separate continuous + jump
HAR terms. Tests whether separating jumps improves OOS vol prediction on the proper
intraday RV target. Honest scope: ~3yr hourly, ~7 bars/day (few for realized measures) —
a prototype/validation of the jump lever, not a full study.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.intraday_research import hourly, ASSETS
from meridian.evalproto import qlike

EPS = 1e-10
MU1 = np.sqrt(2 / np.pi)


def daily_measures(s):
    """Per trading day: RV (total), BV (continuous), J (jump)."""
    lr = np.log(s).diff()
    day = s.index.normalize()
    rows = []
    for d, idx in pd.Series(s.index, index=s.index).groupby(day):
        r = np.log(s.loc[idx]).diff().dropna().to_numpy()
        if len(r) < 4:
            continue
        rv = float((r ** 2).sum())
        M = len(r)
        bv = float((np.pi / 2) * (M / (M - 1)) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
        rows.append({"date": d, "rv": max(rv, EPS), "bv": max(min(bv, rv), EPS)})
    df = pd.DataFrame(rows).set_index("date")
    df["jump"] = (df["rv"] - df["bv"]).clip(lower=0.0)
    df["cont"] = (df["rv"] - df["jump"]).clip(lower=EPS)
    return df


def har_terms(x):
    lx = np.log(x + EPS)
    return np.column_stack([lx, np.log(x.rolling(5).mean() + EPS), np.log(x.rolling(22).mean() + EPS)])


def oos_qlike(X, y):
    n = len(y); tr = slice(0, int(n * 0.6))
    A = np.column_stack([np.ones(int(n * 0.6)), X[tr]])
    beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    resid = y[tr] - A @ beta; jb = 0.5 * resid.var()
    Xte = X[int(n * 0.6):]; yte = y[int(n * 0.6):]
    pred = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    return float(np.nanmean(qlike(np.exp(yte), np.exp(pred + jb))))


def main():
    print("Jump/continuous decomposition + HAR-CJ vs HAR-RV (proper intraday RV, ~3yr)\n")
    print(f"{'asset':>6} {'HAR-RV':>8} {'HAR-CJ':>8} {'edge':>7} {'jump_share':>11}")
    edges = []
    for a in ASSETS:
        try:
            m = daily_measures(hourly(a))
        except Exception:
            continue
        if len(m) < 300:
            continue
        y = np.log(m["rv"].shift(-1).to_numpy() + EPS)
        rv_X = har_terms(m["rv"])
        cj_X = np.column_stack([har_terms(m["cont"]),
                                np.log(1 + m["jump"]).to_numpy()[:, None],
                                np.log(1 + m["jump"].rolling(5).mean()).to_numpy()[:, None],
                                np.log(1 + m["jump"].rolling(22).mean()).to_numpy()[:, None]])
        ok = np.isfinite(y) & np.isfinite(rv_X).all(1) & np.isfinite(cj_X).all(1)
        y, rv_X, cj_X = y[ok], rv_X[ok], cj_X[ok]
        if len(y) < 200:
            continue
        q_rv = oos_qlike(rv_X, y); q_cj = oos_qlike(cj_X, y)
        edge = (q_rv - q_cj) / q_rv * 100
        js = float((m["jump"] / m["rv"]).mean())
        edges.append(edge)
        print(f"{a:>6} {q_rv:>8.4f} {q_cj:>8.4f} {edge:>+6.2f}% {js*100:>10.1f}%")
    if edges:
        print(f"\n  mean HAR-CJ edge over HAR-RV: {np.mean(edges):+.2f}%  "
              f"({'jumps help' if np.mean(edges) > 0 else 'jumps do not help'} on this ~3yr sample)")
    print("  jump_share = avg fraction of daily variance from jumps.")


if __name__ == "__main__":
    main()
