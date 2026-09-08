"""Robustness of the headline volatility win (no retraining):
  1. per-asset QLIKE edge vs HAR-RV (is +6% broad or concentrated?)
  2. stationary block-bootstrap 95% CI on the pooled relative QLIKE reduction
     and a bootstrap p-value (handles autocorrelation the DM test only approximates)

Uses the same calibration + common support as compare_calibrated.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import diebold_mariano, qlike, walk_forward_calibrate

RESULTS = Path(__file__).resolve().parent.parent / "results"
MER_FILE = os.environ.get("MERIDIAN_PRED", "meridian_ens_predictions.parquet")


def calibrated_losses():
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    mer = pd.read_parquet(RESULTS / MER_FILE)
    keep = ["date", "asset", "model", "y_true_log", "y_pred_log"]
    df = pd.concat([base[keep], mer[keep]], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    outs = []
    for (m, a), sub in df.groupby(["model", "asset"]):
        sub = sub.sort_values("date").reset_index(drop=True)
        var, mask = walk_forward_calibrate(sub["date"].values, sub["y_true_log"].to_numpy(),
                                           sub["y_pred_log"].to_numpy())
        outs.append(sub.assign(var_cal=var, cal_ok=mask))
    df = pd.concat(outs, ignore_index=True)
    df = df[df["cal_ok"]].copy()
    df["ql"] = qlike(np.exp(df["y_true_log"].to_numpy()), df["var_cal"].to_numpy())
    # common (asset,date) support
    piv = df.pivot_table(index=["asset", "date"], columns="model", values="ql", aggfunc="first")
    common = piv.dropna().index
    df = df.set_index(["asset", "date"]).loc[
        df.set_index(["asset", "date"]).index.isin(common)].reset_index()
    return df


def stationary_bootstrap_ci(d_mer, d_har, n_boot=4000, mean_block=22, seed=0):
    """Politis-Romano stationary bootstrap CI on relative QLIKE reduction.

    Paired resampling of the two aligned loss series with geometric block
    lengths (mean `mean_block`). Returns (point, lo, hi, p_ge_0).
    """
    rng = np.random.default_rng(seed)
    n = len(d_mer)
    p = 1.0 / mean_block
    point = (d_har.mean() - d_mer.mean()) / d_har.mean() * 100
    rels = np.empty(n_boot)
    ge0 = 0
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            # geometric block length
            L = 1
            while rng.random() > p and i + L < n:
                L += 1
            for k in range(L):
                if i >= n:
                    break
                idx[i] = (start + k) % n
                i += 1
        bm, bh = d_mer[idx], d_har[idx]
        rel = (bh.mean() - bm.mean()) / bh.mean() * 100
        rels[b] = rel
        if (bm.mean() - bh.mean()) >= 0:   # Meridian NOT better in this resample
            ge0 += 1
    lo, hi = np.percentile(rels, [2.5, 97.5])
    return point, lo, hi, ge0 / n_boot


def main():
    df = calibrated_losses()
    mer = df[df.model == "Meridian"].sort_values(["asset", "date"])
    har = df[df.model == "HAR-RV"].sort_values(["asset", "date"])
    j = (mer[["asset", "date"]].assign(ql_mer=mer["ql"].to_numpy())
         .merge(har[["asset", "date"]].assign(ql_har=har["ql"].to_numpy()),
                on=["asset", "date"]))

    print(f"aligned rows: {len(j)}  ({j.date.min().date()} -> {j.date.max().date()})")

    # 1. per-asset
    print("\n=== per-asset QLIKE edge vs HAR-RV ===")
    rows = []
    for a, g in j.groupby("asset"):
        rel = (g.ql_har.mean() - g.ql_mer.mean()) / g.ql_har.mean() * 100
        dm, p = diebold_mariano(g.ql_mer.to_numpy(), g.ql_har.to_numpy())
        rows.append({"asset": a, "n": len(g), "mer": g.ql_mer.mean(),
                     "har": g.ql_har.mean(), "rel_%": rel, "DM_p": p})
    pa = pd.DataFrame(rows).sort_values("rel_%", ascending=False)
    print(pa.round(4).to_string(index=False))
    n_pos = (pa["rel_%"] > 0).sum()
    print(f"\nassets with positive edge: {n_pos}/{len(pa)}"
          f"   (significant p<0.05: {(pa['DM_p']<0.05).sum()})")

    # 2. bootstrap CI on pooled relative reduction
    print("\n=== stationary block-bootstrap (mean block 22d, 4000 reps) ===")
    point, lo, hi, p_ge0 = stationary_bootstrap_ci(
        j.ql_mer.to_numpy(), j.ql_har.to_numpy())
    print(f"relative QLIKE reduction : {point:+.2f}%")
    print(f"95% CI                   : [{lo:+.2f}%, {hi:+.2f}%]")
    print(f"bootstrap p (edge <= 0)  : {p_ge0:.4f}")
    print(f"CI excludes the +5% bar? : {'YES (lower bound > 5)' if lo > 5 else 'NO'}")
    print(f"CI excludes zero?        : {'YES' if lo > 0 else 'NO'}")


if __name__ == "__main__":
    main()
