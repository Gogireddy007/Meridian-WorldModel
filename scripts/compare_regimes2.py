"""Regime head-to-head v2 (PREREGISTRATION.md Amendment 2).

Isolates regime *information* by holding the base forecast fixed at HAR-RV for
both methods, then measuring OOS QLIKE improvement from a per-regime affine map
fit by minimizing exact QLIKE on the first half of OOS and applied to the second.

Meridian regimes  : 3-state Gaussian HMM on the JEPA surprise-energy series.
HMM baseline      : 3-state Gaussian HMM on daily returns.
Sanity gate       : pooled mean dwell in [5, 60] trading days.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.evalproto import qlike
from meridian.regimes import fit_hmm_states, mean_dwell

RESULTS = Path(__file__).resolve().parent.parent / "results"


def qlike_affine_fit(logvar_base, y_true_log):
    """Fit a,b minimizing mean QLIKE of variance = exp(a + b*logvar_base)."""
    rv = np.exp(y_true_log)

    def obj(ab):
        f = ab[0] + ab[1] * logvar_base
        s2 = np.exp(np.clip(f, -30, 30))
        r = rv / s2
        return np.mean(r - np.log(r) - 1.0)

    res = minimize(obj, x0=np.array([0.0, 1.0]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return res.x


def regime_value(logvar_base, y_true_log, states, fit_mask):
    """OOS QLIKE improvement (%) of (HAR + per-regime affine) over global affine.

    Global and per-regime affines are both fit on fit_mask (QLIKE-optimal) and
    scored on the complement (test half). Returns (global_q, regime_q, improve%).
    """
    test_mask = ~fit_mask
    # global affine baseline (fair reference: HAR already QLIKE-recalibrated)
    ga = qlike_affine_fit(logvar_base[fit_mask], y_true_log[fit_mask])
    g_f = ga[0] + ga[1] * logvar_base[test_mask]
    g_q = float(np.mean(qlike(np.exp(y_true_log[test_mask]), np.exp(g_f))))

    # per-regime affine
    f_test = np.empty(test_mask.sum())
    lv_test = logvar_base[test_mask]
    st_test = states[test_mask]
    for r in np.unique(states):
        fr = fit_mask & (states == r)
        if fr.sum() >= 40:
            ar = qlike_affine_fit(logvar_base[fr], y_true_log[fr])
        else:
            ar = ga
        sel = st_test == r
        f_test[sel] = ar[0] + ar[1] * lv_test[sel]
    r_q = float(np.mean(qlike(np.exp(y_true_log[test_mask]), np.exp(f_test))))
    return g_q, r_q, (g_q - r_q) / g_q * 100


def main():
    mer = pd.read_parquet(RESULTS / "meridian_ens_predictions.parquet")
    mer["date"] = pd.to_datetime(mer["date"])
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    har = base[base["model"] == "HAR-RV"].copy()
    har["date"] = pd.to_datetime(har["date"])
    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}

    rows = []
    for asset in mer["asset"].unique():
        m = mer[mer["asset"] == asset].sort_values("date")
        h = har[har["asset"] == asset].set_index("date")
        j = m.set_index("date").join(h[["y_pred_log", "logvar_bias"]], rsuffix="_har")
        j = j.dropna(subset=["y_pred_log_har"])
        if len(j) < 500:
            continue
        y_true = j["y_true_log"].to_numpy()
        har_logvar = j["y_pred_log_har"].to_numpy() + j["logvar_bias"].to_numpy()
        energy = j["energy"].to_numpy()
        rets = frames[asset]["ret"].reindex(j.index).ffill().to_numpy()

        try:
            _, hmm_states, _ = fit_hmm_states(rets, k=3)
            _, mer_states, _ = fit_hmm_states(energy, k=3)
        except Exception:
            continue

        # time split: first half fit, second half test
        n = len(j)
        fit_mask = np.zeros(n, bool); fit_mask[: n // 2] = True

        h_dwell, m_dwell = mean_dwell(hmm_states), mean_dwell(mer_states)
        _, _, hmm_val = regime_value(har_logvar, y_true, hmm_states, fit_mask)
        _, _, mer_val = regime_value(har_logvar, y_true, mer_states, fit_mask)
        rows.append({"asset": asset, "hmm_dwell": h_dwell, "mer_dwell": m_dwell,
                     "hmm_value_%": hmm_val, "mer_value_%": mer_val})

    df = pd.DataFrame(rows)
    print("=== Regime economic value (OOS test half; base forecast = HAR) ===")
    print(df.round(3).to_string(index=False))
    agg = df[["hmm_dwell", "mer_dwell", "hmm_value_%", "mer_value_%"]].mean()
    print("\n=== POOLED (mean across assets) ===")
    print(agg.round(3).to_string())

    gate_ok = 5 <= agg["mer_dwell"] <= 60
    value_ok = agg["mer_value_%"] > agg["hmm_value_%"]
    print(f"\n  dwell sanity gate [5,60]d: Meridian {agg['mer_dwell']:.1f}d  "
          f"{'PASS' if gate_ok else 'FAIL'}   (HMM {agg['hmm_dwell']:.1f}d)")
    print(f"  regime economic value    : Meridian {agg['mer_value_%']:+.2f}%  vs  "
          f"HMM {agg['hmm_value_%']:+.2f}%   {'PASS' if value_ok else 'FAIL'}")
    print(f"\n  >>> Meridian regimes {'BEAT' if (gate_ok and value_ok) else 'DO NOT BEAT'} "
          f"HMM (Amendment 2 criteria).")


if __name__ == "__main__":
    main()
