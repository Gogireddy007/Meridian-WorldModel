"""Regime-overlay backtest: take the BEST vol forecast (Meridian EMA-target ensemble)
for vol-managing, and overlay the Meridian-WM switching regime — cut exposure in the
high-vol (stress) regime. Regime→vol ranking is learned on the FIRST HALF of OOS and
applied to the SECOND HALF (no lookahead). Does the regime signal add Sharpe/alpha?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.evalproto import walk_forward_calibrate
from scripts.backtest import calibrated_var, perf, ANN, TARGET_VOL, W_MAX, COST_BPS

RESULTS = Path(__file__).resolve().parent.parent / "results"
OVERLAY = {0: 1.0, 1: 0.7, 2: 0.3}   # calm / transition / stress exposure (fixed, not tuned)


def main():
    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}
    rnext = pd.concat([frames[a][["r_next"]].assign(asset=a).reset_index().rename(
        columns={"index": "date"}) for a in frames], ignore_index=True)
    rnext.columns = ["date", "r_next", "asset"]; rnext["date"] = pd.to_datetime(rnext["date"])

    mer = pd.read_parquet(RESULTS / "cfjepa_ens_predictions.parquet")
    mer["date"] = pd.to_datetime(mer["date"])
    mv = calibrated_var(mer, "y_pred_tgt")[["asset", "date", "pred_var"]]

    wm = pd.read_parquet(RESULTS / "wm_predictions.parquet")[["asset", "date", "regime", "y_true_log"]]
    wm["date"] = pd.to_datetime(wm["date"])

    df = mv.merge(rnext, on=["asset", "date"]).merge(wm, on=["asset", "date"]).dropna(subset=["r_next"])
    df["sigma_ann"] = np.sqrt(df["pred_var"].clip(1e-8)) * np.sqrt(ANN)
    df["w"] = (TARGET_VOL / df["sigma_ann"]).clip(0, W_MAX)

    managed, overlaid, bh = [], [], []
    for a, s in df.groupby("asset"):
        s = s.sort_values("date").reset_index(drop=True)
        half = len(s) // 2
        # rank regimes by realized vol on FIRST half (no lookahead into 2nd half)
        rv = np.exp(s["y_true_log"].to_numpy())
        order = (pd.Series(rv[:half]).groupby(s["regime"].to_numpy()[:half]).mean()
                 .sort_values().index.tolist())
        rank = {r: i for i, r in enumerate(order)}                 # 0=calm .. 2=stress
        fac = s["regime"].map(lambda r: OVERLAY.get(rank.get(r, 1), 1.0)).to_numpy()
        w = s["w"].to_numpy(); r = s["r_next"].to_numpy()
        w_ov = w * fac
        t_m = np.abs(np.diff(w, prepend=w[0])); t_o = np.abs(np.diff(w_ov, prepend=w_ov[0]))
        managed.append(pd.Series(w * r - COST_BPS * t_m, index=s["date"]))
        overlaid.append(pd.Series(w_ov * r - COST_BPS * t_o, index=s["date"]))
        bh.append(pd.Series(r, index=s["date"]))
    M = pd.concat(managed, axis=1).mean(1).sort_index()
    O = pd.concat(overlaid, axis=1).mean(1).sort_index()
    B = pd.concat(bh, axis=1).mean(1).sort_index()
    idx = M.index.intersection(O.index).intersection(B.index)
    M, O, B = M.loc[idx], O.loc[idx], B.loc[idx]
    # evaluate on the SECOND HALF only (overlay params are out-of-sample there)
    h = len(idx) // 2; sl = idx[h:]

    print(f"regime-overlay backtest (2nd-half OOS {sl.min().date()}→{sl.max().date()}, n={len(sl)})\n")
    hdr = f"{'strategy':>26} {'Sharpe':>7} {'annRet':>7} {'maxDD':>7} {'alpha%':>7} {'alpha_t':>8}"
    print(hdr)
    for name, series in [("Buy&Hold", B.loc[sl]), ("vol-managed", M.loc[sl]),
                         ("vol-managed + WM-regime", O.loc[sl])]:
        v = perf(series, B.loc[sl] if name != "Buy&Hold" else None)
        print(f"{name:>26} {v['sharpe']:>7.2f} {v['ann_ret']*100:>6.1f}% {v['maxDD']*100:>6.1f}% "
              f"{v.get('alpha_ann', float('nan'))*100:>6.1f}% {v.get('alpha_t', float('nan')):>8.2f}")


if __name__ == "__main__":
    main()
