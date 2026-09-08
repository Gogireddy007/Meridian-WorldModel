"""Regime head-to-head: Gaussian HMM (on returns) vs Meridian belief regimes.

Metrics (see PREREGISTRATION.md / meridian.regimes):
  * mean dwell time (persistence), unit = days
  * economic check: QLIKE improvement when a common vol forecast (HAR-RV) is
    given a per-regime bias — measures how economically useful the regimes are.

Both are representation-invariant, so the comparison is fair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.regimes import fit_hmm_states, mean_dwell, regime_conditioned_qlike

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(k=3):
    mer = pd.read_parquet(RESULTS / "meridian_predictions.parquet")
    belief = np.load(RESULTS / "meridian_belief.npy")
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    har = base[base["model"] == "HAR-RV"].copy()
    har["date"] = pd.to_datetime(har["date"])
    mer["date"] = pd.to_datetime(mer["date"])

    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}

    rows = []
    for asset in mer["asset"].unique():
        m = mer[mer["asset"] == asset].sort_values("date").reset_index()
        h = har[har["asset"] == asset].set_index("date")
        # align HAR forecast to Meridian OOS rows
        j = m.set_index("date").join(h[["y_pred_log", "logvar_bias"]],
                                     rsuffix="_har").dropna(subset=["y_pred_log_har"])
        if len(j) < 300:
            continue
        # returns for HMM baseline over the same OOS window
        rets = frames[asset]["ret"].reindex(j.index).ffill().to_numpy()

        # HMM baseline on returns
        try:
            _, hmm_states, hmm_ll = fit_hmm_states(rets, k=k)
        except Exception:
            continue
        # Meridian regimes on belief states (rows for this asset)
        bidx = m["_belief_row"].to_numpy()
        # realign to joined index
        keep = m.set_index("date").loc[j.index, "_belief_row"].to_numpy().astype(int)
        bel = belief[keep]
        try:
            _, mer_states, _ = fit_hmm_states(bel, k=k)
        except Exception:
            continue

        rv_true_log = j["y_true_log"].to_numpy()
        har_pred = j["y_pred_log_har"].to_numpy()
        har_bias = j["logvar_bias_har"].to_numpy()

        h_dwell = mean_dwell(hmm_states)
        m_dwell = mean_dwell(mer_states)
        _, _, hmm_econ = regime_conditioned_qlike(rv_true_log, har_pred, har_bias, hmm_states)
        _, _, mer_econ = regime_conditioned_qlike(rv_true_log, har_pred, har_bias, mer_states)
        rows.append({"asset": asset, "hmm_dwell": h_dwell, "mer_dwell": m_dwell,
                     "hmm_econ_%": hmm_econ, "mer_econ_%": mer_econ})

    df = pd.DataFrame(rows)
    print("=== Per-asset regime comparison (K=%d) ===" % k)
    print(df.round(3).to_string(index=False))
    print("\n=== POOLED (mean across assets) ===")
    agg = df[["hmm_dwell", "mer_dwell", "hmm_econ_%", "mer_econ_%"]].mean()
    print(agg.round(3).to_string())

    dwell_gain = (agg["mer_dwell"] - agg["hmm_dwell"]) / agg["hmm_dwell"] * 100
    econ_ok = agg["mer_econ_%"] >= agg["hmm_econ_%"]
    print(f"\n  persistence: Meridian dwell {agg['mer_dwell']:.2f}d vs HMM {agg['hmm_dwell']:.2f}d "
          f"({dwell_gain:+.1f}%)   need >= +10%   {'PASS' if dwell_gain >= 10 else 'FAIL'}")
    print(f"  economic check: Meridian {agg['mer_econ_%']:.2f}% vs HMM {agg['hmm_econ_%']:.2f}% "
          f"QLIKE improvement   {'PASS' if econ_ok else 'FAIL'}")
    print(f"\n  >>> Meridian regimes {'BEAT' if (dwell_gain >= 10 and econ_ok) else 'DO NOT BEAT'} "
          f"HMM by the pre-registered margin.")


if __name__ == "__main__":
    main()
