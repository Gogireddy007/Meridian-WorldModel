"""REGIME-COUNT TEST — does the data actually support >1 volatility regime? (research build #1)

The interpretability research (§4c, verified) is blunt: testing the number of regimes with a
naive likelihood-ratio test DRAMATICALLY OVER-REJECTS, because under the 1-regime null the
2nd-regime parameters and the transition probabilities are UNIDENTIFIED and the information
matrix is singular (Hansen 1992; Carrasco-Hu-Ploberger). So a big LR statistic is NOT
evidence of regimes. The accepted honest fix is a PARAMETRIC BOOTSTRAP of the LR distribution
under the 1-regime null (McLachlan 1987).

We fit a standard Markov-switching AR(1) with switching variance on log realized variance
(the interpretable foundational question: is the volatility process 1-regime or 2-regime?),
compute the observed LR, then bootstrap the null LR distribution and get a proper p-value.
If we cannot reject 1 regime at 5%, the modular philosophy says RETIRE the regime claim and
ship the simpler single-regime module — a valuable, honest negative result.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.ar_model import AutoReg
from meridian.data import load_all
from meridian.features import realized_variance

B = 99            # bootstrap replications (proper null LR distribution)
SEED = 0


def fit_null(y):
    """1-regime null = plain AR(1)."""
    r = AutoReg(y, lags=1, trend="c", old_names=False).fit()
    c, phi = r.params[0], r.params[1]
    return r.llf, c, phi, float(r.sigma2)


def fit_alt(y):
    """2-regime Markov-switching AR(1) with switching variance."""
    m = MarkovAutoregression(y, k_regimes=2, order=1, switching_variance=True)
    r = m.fit(em_iter=50, maxiter=100, disp=False)
    return r.llf


def lr_stat(y):
    ll1, c, phi, s2 = fit_null(y)
    ll2 = fit_alt(y)
    return 2 * (ll2 - ll1), (c, phi, np.sqrt(s2))


def sim_ar1(par, n, rng):
    """simulate from the fitted 1-regime AR(1): y_t = c + phi y_{t-1} + eps."""
    c, phi, sig = par
    y = np.empty(n); y[0] = c / (1 - phi) if abs(phi) < 1 else c
    e = rng.normal(0, sig, n)
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + e[t]
    return y


def main():
    d = load_all()
    rng = np.random.RandomState(SEED)
    print(f"Regime-count test (parametric bootstrap LR, B={B}) — is vol 1-regime or 2-regime?\n")
    print(f"  {'asset':>8} {'obs LR':>9} {'naive p':>9} {'BOOTSTRAP p':>12} {'verdict':>22}")
    from scipy import stats as st
    for a in ["SPY", "QQQ", "TLT", "EURUSD"]:
        if a not in d["prices"]:
            continue
        y = np.log(realized_variance(d["prices"][a])["rv"].dropna().to_numpy())
        y = y[np.isfinite(y)]
        y = (y - y.mean()) / y.std()                  # standardize (numerical stability of EM)
        try:
            obs, par = lr_stat(y)
        except Exception as e:
            print(f"  {a:>8}  fit failed ({str(e)[:30]})"); continue
        naive_p = 1 - st.chi2.cdf(obs, df=3)          # WRONG dist (shown for contrast)
        # bootstrap null
        boot = []
        for _ in range(B):
            ys = sim_ar1(par, len(y), rng)
            try:
                lr, _ = lr_stat(ys)
                if np.isfinite(lr):
                    boot.append(max(lr, 0.0))
            except Exception:
                pass
        boot = np.array(boot)
        bp = (1 + np.sum(boot >= obs)) / (1 + len(boot))    # bootstrap p-value
        verdict = "2 regimes REAL" if bp < 0.05 else "cannot reject 1 regime"
        print(f"  {a:>8} {obs:>9.1f} {naive_p:>9.3f} {bp:>12.3f} {verdict:>22}")

    print(f"\n  naive p uses the (invalid) chi^2(3); BOOTSTRAP p is the honest test.")
    print("  Where bootstrap p<0.05, a 2nd volatility regime is genuinely supported; where not,")
    print("  the modular philosophy says ship the single-regime module and retire the regime claim.")


if __name__ == "__main__":
    main()
