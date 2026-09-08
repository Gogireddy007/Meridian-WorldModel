"""Adaptive Conformal Inference (ACI, Gibbs-Candes 2021) VaR — volatility-normalized.
Tests whether ACI fixes the CLUSTERED VaR violations (Christoffersen failure) of the
parametric Student-t tail module. ACI adapts the quantile level online to guarantee
long-run coverage under non-stationarity: after a violation it widens the interval,
directly attacking clustering. Vol-normalization (score = return/sigma) handles
heteroskedasticity. Backtested with Kupiec + Christoffersen vs the parametric VaR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_wm import kupiec_pof, christoffersen_ind

RESULTS = Path(__file__).resolve().parent.parent / "results"
Q = 0.05          # target left-tail VaR level
GAMMA = 0.02      # ACI learning rate
WIN = 500         # trailing window for the standardized-return quantile


def aci_var(sigma, r, q=Q, gamma=GAMMA, win=WIN):
    """Volatility-normalized ACI VaR. Returns (var_series, violations)."""
    z = r / np.clip(sigma, 1e-8, None)                       # standardized returns
    n = len(r); var = np.full(n, np.nan); viol = np.zeros(n, bool)
    a = q                                                    # adaptive level
    for t in range(win, n):
        zt = z[t - win:t]
        a_cl = min(max(a, 1e-3), 0.5)
        var[t] = sigma[t] * np.quantile(zt, a_cl)            # a-quantile of past standardized returns
        viol[t] = r[t] < var[t]
        a = a + gamma * (q - (1.0 if viol[t] else 0.0))      # ACI update (widen after a violation)
    return var, viol


def static_conformal_var(sigma, r, q=Q, win=WIN):
    z = r / np.clip(sigma, 1e-8, None); n = len(r); var = np.full(n, np.nan)
    for t in range(win, n):
        var[t] = sigma[t] * np.quantile(z[t - win:t], q)
    return var


def regime_conditional_aci(sigma, r, regime, q=Q, gamma=GAMMA, win=WIN):
    """ACI with a PER-REGIME adaptive level — links the regime module to the tail module
    for CONDITIONAL coverage (each regime gets its own tail calibration)."""
    z = r / np.clip(sigma, 1e-8, None); n = len(r); var = np.full(n, np.nan)
    a = {g: q for g in np.unique(regime)}
    for t in range(win, n):
        g = regime[t]
        zt = z[t - win:t][regime[t - win:t] == g]           # past standardized returns IN this regime
        if len(zt) < 30:
            zt = z[t - win:t]
        var[t] = sigma[t] * np.quantile(zt, min(max(a[g], 1e-3), 0.5))
        viol = r[t] < var[t]
        a[g] = a[g] + gamma * (q - (1.0 if viol else 0.0))
    return var


def main():
    wm = pd.read_parquet(RESULTS / "wm3_predictions.parquet")
    wm["date"] = pd.to_datetime(wm["date"])
    print("VaR backtest — parametric Student-t vs conformal (ACI) — does ACI fix clustering?\n")
    print(f"{'method':>22} {'exceed%':>8} {'target%':>8} {'Kupiec_p':>9} {'Christoff_p':>12}")
    rows_par_v, rows_aci_v, rows_stat_v, rows_rc_v = [], [], [], []
    for a, s in wm.groupby("asset"):
        s = s.sort_values("date")
        sigma = np.sqrt(np.exp(s["y_pred_log"].to_numpy()))
        r = s["r_next"].to_numpy()
        # parametric Student-t VaR (module's own df/scale)
        par = s["scale"].to_numpy() * stats.t.ppf(Q, s["df"].to_numpy())
        rows_par_v.append(r < par)
        av, avv = aci_var(sigma, r); rows_aci_v.append((r < av) & ~np.isnan(av))
        sv = static_conformal_var(sigma, r); rows_stat_v.append((r < sv) & ~np.isnan(sv))
        rc = regime_conditional_aci(sigma, r, s["regime"].to_numpy()); rows_rc_v.append((r < rc) & ~np.isnan(rc))
    def report(name, viols):
        v = np.concatenate(viols); v = v[np.isfinite(v.astype(float))]
        _, pk, pi = kupiec_pof(v, Q); _, pc = christoffersen_ind(v)
        print(f"{name:>22} {pi*100:>7.2f}% {Q*100:>7.2f}% {pk:>9.3f} {pc:>12.3f}")
    report("parametric Student-t", rows_par_v)
    report("static conformal", rows_stat_v)
    report("ADAPTIVE conformal (ACI)", rows_aci_v)
    report("regime-conditional ACI", rows_rc_v)
    print("\n  (Christoffersen p>0.05 = violations are INDEPENDENT, no clustering — the goal ACI targets.)")


if __name__ == "__main__":
    main()
