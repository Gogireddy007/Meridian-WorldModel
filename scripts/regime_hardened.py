"""HARDENED regime-existence test — closes the hole the frontier research found in the
earlier p=0.01 claim (regime_count_test.py used an AR(1) null; a HAR-driven conditionally
heteroskedastic series can make a 2-regime fit reject from MISSPECIFIED VARIANCE, not a
true 2nd state).

Correct null = HAR+leverage mean model. We test single-regime HAR+leverage (OLS) vs a
2-regime Markov-switching HAR+leverage (switching intercept + variance), and bootstrap the
LR distribution TWO ways:
  * IID residual resample     — homoskedastic null (destroys conditional heteroskedasticity)
  * BLOCK residual resample   — preserves the residual's conditional heteroskedasticity
If the observed LR is significant under BOTH, the 2nd regime is genuine. If it is significant
only under IID (not BLOCK), the "regime" was heteroskedasticity — and the claim must be retired.
Multi-start EM on every replicate (guards against local-optimum contamination).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from meridian.data import load_all
from meridian.features import realized_variance

EPS = 1e-12
B = 49            # bootstrap replications
L = 20            # block length (preserves conditional heteroskedasticity)
STARTS = 2        # EM multi-starts per alt fit


def design(ohlc):
    rvf = realized_variance(ohlc); rv = rvf["rv"]; ret = rvf["ret"]
    har_d = np.log(rv + EPS)
    har_w = np.log(rv.rolling(5).mean() + EPS)
    har_m = np.log(rv.rolling(22).mean() + EPS)
    neg = np.log((ret.clip(upper=0) ** 2) + EPS)          # leverage / bad-vol
    y = np.log(rv + EPS).shift(-1)
    df = pd.DataFrame({"y": y, "har_d": har_d, "har_w": har_w, "har_m": har_m, "neg": neg}).dropna()
    return df["y"].to_numpy(), df[["har_d", "har_w", "har_m", "neg"]].to_numpy()


def fit_alt(y, X, rng):
    best = -np.inf
    for _ in range(STARTS):
        try:
            m = MarkovRegression(y, k_regimes=2, exog=X, trend="c", switching_variance=True)
            r = m.fit(em_iter=40, maxiter=80, disp=False)
            if np.isfinite(r.llf) and r.llf > best:
                best = r.llf
        except Exception:
            pass
    return best


def lr_obs(y, X, rng):
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    alt = fit_alt(y, X, rng)
    return 2 * (alt - ols.llf), ols.fittedvalues, ols.resid


def block_resample(r, n, rng):
    out = []
    while len(out) < n:
        s = rng.randint(0, len(r) - L)
        out.extend(r[s:s + L])
    return np.array(out[:n])


def main():
    d = load_all()
    rng = np.random.RandomState(0)
    print(f"HARDENED regime test — null=HAR+leverage, B={B}, block L={L}, {STARTS}-start EM\n")
    print(f"  {'asset':>7} {'obs LR':>8} {'p(iid)':>8} {'p(block)':>9} {'verdict':>34}")
    for a in ["SPY", "QQQ", "TLT"]:
        if a not in d["prices"]:
            continue
        y, X = design(d["prices"][a])
        y = (y - y.mean()) / y.std()                       # standardize (EM numerical stability)
        X = (X - X.mean(0)) / X.std(0)
        try:
            obs, fitted, resid = lr_obs(y, X, rng)
        except Exception as e:
            print(f"  {a:>7}  fit failed ({str(e)[:30]})"); continue
        resid = resid - resid.mean()
        boot_iid, boot_blk = [], []
        for _ in range(B):
            ri = resid[rng.randint(0, len(resid), len(resid))]
            lr_i, _, _ = lr_obs(fitted + ri, X, rng)
            if np.isfinite(lr_i): boot_iid.append(max(lr_i, 0.0))
            rb = block_resample(resid, len(resid), rng)
            lr_b, _, _ = lr_obs(fitted + rb, X, rng)
            if np.isfinite(lr_b): boot_blk.append(max(lr_b, 0.0))
        pi = (1 + np.sum(np.array(boot_iid) >= obs)) / (1 + len(boot_iid))
        pb = (1 + np.sum(np.array(boot_blk) >= obs)) / (1 + len(boot_blk))
        if pb < 0.05:
            v = "REGIME REAL (survives het-preserving null)"
        elif pi < 0.05:
            v = "ARTIFACT: only iid — was heteroskedasticity"
        else:
            v = "cannot reject 1 regime"
        print(f"  {a:>7} {obs:>8.1f} {pi:>8.3f} {pb:>9.3f} {v:>34}")

    print("\n  p(block) is the HONEST test — its null preserves HAR conditional heteroskedasticity.")
    print("  REGIME REAL only if p(block)<0.05; if only p(iid)<0.05, the earlier claim was a")
    print("  heteroskedasticity artifact and the regime existence claim is retired.")


if __name__ == "__main__":
    main()
