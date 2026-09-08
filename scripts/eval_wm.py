"""Evaluate Meridian-WM: (1) tail VaR backtest, (2) switching-regime economic value.

Vol QLIKE vs HAR is done separately by compare_calibrated.py with
MERIDIAN_PRED=wm_predictions.parquet.

Tail: the model's predictive return distribution is StudentT(df, 0, scale) with
scale set so Var = sigma^2, sigma^2 = exp(vol forecast). We backtest left-tail VaR
at 5% and 1% with the Kupiec POF (unconditional coverage) and Christoffersen
independence tests, and contrast with a Gaussian(sigma) baseline to isolate the
heavy-tail contribution. (Proper scoring rules cannot validate extreme tails —
research caveat — so we use VaR exceedance backtests, the elicitable proxy.)

Regime: economic value of the switching regime vs a Gaussian HMM on returns, on a
fixed HAR base forecast (same test as compare_regimes2), plus dwell/persistence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.regimes import fit_hmm_states, mean_dwell
from scripts.compare_regimes2 import regime_value

RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------------- VaR backtests ----------------
def kupiec_pof(viol: np.ndarray, p: float):
    n = len(viol); x = int(viol.sum())
    if x == 0:
        return np.nan, np.nan, x / n
    pi = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1)), pi


def christoffersen_ind(viol: np.ndarray):
    v = viol.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(v[:-1], v[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan
    pi01 = n01 / (n00 + n01); pi11 = n11 / (n10 + n11); pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll1 = n00 * np.log(1 - pi01 + 1e-12) + n01 * np.log(pi01 + 1e-12) + \
          n10 * np.log(1 - pi11 + 1e-12) + n11 * np.log(pi11 + 1e-12)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def var_backtest(df):
    r = df["r_next"].to_numpy()
    nu = df["df"].to_numpy()
    scale = df["scale"].to_numpy()                          # StudentT scale (learned)
    # Gaussian baseline with the same predictive std as the Student-t
    gstd = scale * np.sqrt(nu / (nu - 2))
    print("=== TAIL: left-VaR exceedance backtests ===")
    print(f"{'level':>6} {'model':>10} {'exceed%':>8} {'target%':>8} {'Kupiec_p':>9} {'Chr_ind_p':>10}")
    for q in (0.05, 0.01):
        var_t = scale * stats.t.ppf(q, nu)                 # Student-t VaR (left tail)
        viol_t = r < var_t
        _, p_k, pi = kupiec_pof(viol_t, q)
        _, p_i = christoffersen_ind(viol_t)
        print(f"{q:>6.2f} {'Student-t':>10} {pi*100:>7.2f}% {q*100:>7.2f}% {p_k:>9.3f} {p_i:>10.3f}")
        var_g = gstd * stats.norm.ppf(q)
        viol_g = r < var_g
        _, p_kg, pig = kupiec_pof(viol_g, q)
        _, p_ig = christoffersen_ind(viol_g)
        print(f"{q:>6.2f} {'Gaussian':>10} {pig*100:>7.2f}% {q*100:>7.2f}% {p_kg:>9.3f} {p_ig:>10.3f}")
    print("  (exceed% near target% = well-calibrated; Kupiec/Christoffersen p>0.05 = not rejected)")


# ---------------- Regime economic value ----------------
def regime_eval(wm):
    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    har = base[base.model == "HAR-RV"].copy(); har["date"] = pd.to_datetime(har["date"])
    rows = []
    for a in wm["asset"].unique():
        m = wm[wm.asset == a].sort_values("date")
        h = har[har.asset == a].set_index("date")
        j = m.set_index("date").join(h[["y_pred_log", "logvar_bias"]], rsuffix="_har").dropna(subset=["y_pred_log_har"])
        if len(j) < 500:
            continue
        y_true = j["y_true_log"].to_numpy()
        har_lv = j["y_pred_log_har"].to_numpy() + j["logvar_bias"].to_numpy()
        wm_reg = j["regime"].to_numpy().astype(int)
        rets = frames[a]["ret"].reindex(j.index).ffill().to_numpy()
        try:
            _, hmm_reg, _ = fit_hmm_states(rets, k=3)
        except Exception:
            continue
        n = len(j); fit_mask = np.zeros(n, bool); fit_mask[: n // 2] = True
        _, _, wm_val = regime_value(har_lv, y_true, wm_reg, fit_mask)
        _, _, hmm_val = regime_value(har_lv, y_true, hmm_reg, fit_mask)
        rows.append({"asset": a, "wm_dwell": mean_dwell(wm_reg), "hmm_dwell": mean_dwell(hmm_reg),
                     "wm_value_%": wm_val, "hmm_value_%": hmm_val})
    df = pd.DataFrame(rows)
    print("\n=== REGIME: switching-SSM regime vs Gaussian HMM (economic value on fixed HAR) ===")
    print(df.round(2).to_string(index=False))
    agg = df[["wm_dwell", "hmm_dwell", "wm_value_%", "hmm_value_%"]].mean()
    gate = 5 <= agg["wm_dwell"] <= 60
    print(f"\n  pooled: WM dwell {agg.wm_dwell:.1f}d (HMM {agg.hmm_dwell:.1f}d)  gate[5,60]:{'PASS' if gate else 'FAIL'}")
    print(f"  pooled: WM econ value {agg['wm_value_%']:+.2f}%  vs HMM {agg['hmm_value_%']:+.2f}%  "
          f"{'WM WINS' if agg['wm_value_%'] > agg['hmm_value_%'] else 'HMM wins'}")


def main():
    wm = pd.read_parquet(RESULTS / "wm_predictions.parquet")
    wm["date"] = pd.to_datetime(wm["date"])
    var_backtest(wm)
    regime_eval(wm)


if __name__ == "__main__":
    main()
