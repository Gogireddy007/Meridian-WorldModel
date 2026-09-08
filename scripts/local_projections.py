"""Causal impulse-response module — Local Projections (Jorda 2005), the evidence-backed
method from the causal research. Traces how a MARKET shock propagates across assets over
horizons h=0..H — the *dynamic* propagation the static Diebold-Yilmaz connectedness cannot
give. Honest identification: the market (SPY) return is treated as the structural shock via
a recursive (SPY-first) ordering — a STATED assumption, not manufactured by the estimator.
Reduced-form for cross-asset innovations; Newey-West (h-lag) inference. No causal claim
beyond the identifying assumption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_broad import load_broad, ASSET_CLASS

Hmax = 10


def nw_se(x, resid, lag):
    """Newey-West SE for the slope in a simple OLS resid context (approx via HAC of x*resid)."""
    u = (x - x.mean()) * resid
    n = len(u); g0 = np.mean(u ** 2); v = g0
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * np.mean(u[k:] * u[:-k])
    sxx = np.sum((x - x.mean()) ** 2)
    return np.sqrt(v * n) / sxx


def lp_irf(shock, target, controls, H):
    """β_h = response of cumulative target return over t+1..t+h to a unit shock_t."""
    betas, sigs = [], []
    for h in range(1, H + 1):
        yq = pd.Series(target).shift(-1).rolling(h).sum().shift(-(h - 1))
        df = pd.concat([yq.rename("y"), pd.Series(shock, name="s")] +
                       [pd.Series(c, name=f"c{i}") for i, c in enumerate(controls)], axis=1).dropna()
        X = np.column_stack([np.ones(len(df)), df["s"].to_numpy()] +
                            [df[f"c{i}"].to_numpy() for i in range(len(controls))])
        beta, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
        resid = df["y"].to_numpy() - X @ beta
        se = nw_se(df["s"].to_numpy(), resid, h)
        betas.append(beta[1]); sigs.append(abs(beta[1] / se) if se > 0 else 0)
    return np.array(betas), np.array(sigs)


def main():
    d = load_broad()
    names = [n for n in ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
                         "GLD", "USO", "DBC", "EURUSD", "USDJPY", "AUDUSD"] if n in d]
    ret = pd.DataFrame({n: np.log(d[n]["adjclose"]).diff() for n in names}).dropna()
    ret = ret[ret.index >= "2010-01-01"]
    # market shock = SPY return orthogonalized to its own lag (recursive, SPY-first)
    spy = ret["SPY"]; shock = (spy - 0.0).to_numpy()          # SPY innovation as the market shock
    lags = [ret[n].shift(1).to_numpy() for n in names]        # controls: 1-day lags

    print("Local-Projection impulse responses to a +1σ MARKET (SPY) shock — cumulative % response\n")
    print(f"  identification: SPY return = structural market shock (recursive/SPY-first). Honest, stated.\n")
    print(f"{'asset':>8} {'class':>10} {'h=1':>7} {'h=3':>7} {'h=5':>7} {'h=10':>7} {'sig(h5)':>8}")
    sd = spy.std()
    for n in names:
        b, s = lp_irf(shock, ret[n].to_numpy(), lags, Hmax)
        b = b * sd * 100                                       # response to a 1-sigma SPY shock, in %
        print(f"{n:>8} {ASSET_CLASS.get(n,'?'):>10} {b[0]:>+6.2f} {b[2]:>+6.2f} {b[4]:>+6.2f} {b[9]:>+6.2f} {s[4]:>7.1f}σ")
    print("\n  (Positive = moves WITH the market shock; negative = hedges/absorbs. The h-path is the")
    print("   DYNAMIC propagation over 10 days — what static connectedness cannot show.)")


if __name__ == "__main__":
    main()
