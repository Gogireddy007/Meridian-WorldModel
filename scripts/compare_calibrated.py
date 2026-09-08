"""Fair, symmetric comparison: apply the SAME leakage-safe walk-forward
affine+Jensen calibration to EVERY model, then re-run the pre-registered
volatility verdict vs HAR-RV.

Rationale: the raw Meridian head has the best MZ-R2 but a mis-scaled variance
level. Calibrating only Meridian would be unfair, so we calibrate all models
identically (a no-op for the log-linear baselines). This is post-processing fit
on past OOS rows only -> no leakage, and disclosed in RESULTS.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import diebold_mariano, mz_r2, qlike, walk_forward_calibrate

RESULTS = Path(__file__).resolve().parent.parent / "results"
MER_FILE = os.environ.get("MERIDIAN_PRED", "meridian_predictions.parquet")


def load():
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    mer = pd.read_parquet(RESULTS / MER_FILE)
    keep = ["date", "asset", "model", "y_true_log", "y_pred_log"]
    both = pd.concat([base[keep], mer[keep]], ignore_index=True)
    both["date"] = pd.to_datetime(both["date"])
    return both


def calibrate_all(df):
    """Per-model, per-asset walk-forward calibration -> variance forecast column."""
    outs = []
    for (m, a), sub in df.groupby(["model", "asset"]):
        sub = sub.sort_values("date").reset_index(drop=True)
        var, mask = walk_forward_calibrate(sub["date"].values,
                                           sub["y_true_log"].to_numpy(),
                                           sub["y_pred_log"].to_numpy())
        sub = sub.assign(var_cal=var, cal_ok=mask)
        outs.append(sub)
    return pd.concat(outs, ignore_index=True)


def main():
    df = calibrate_all(load())
    df = df[df["cal_ok"]].copy()
    df["ql"] = qlike(np.exp(df["y_true_log"].to_numpy()), df["var_cal"].to_numpy())

    # common (asset,date) support across all models
    piv = df.pivot_table(index=["asset", "date"], columns="model", values="ql", aggfunc="first")
    common = piv.dropna().index
    df = df.set_index(["asset", "date"]).loc[df.set_index(["asset", "date"]).index.isin(common)].reset_index()
    models = df["model"].unique().tolist()
    print(f"common calibrated OOS rows: {len(common)}  "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    rows = []
    loss = {}
    for m in models:
        s = df[df["model"] == m].sort_values(["asset", "date"]).reset_index(drop=True)
        loss[m] = s[["asset", "date"]].assign(loss=s["ql"].to_numpy())
        rows.append({"model": m, "qlike": float(np.nanmean(s["ql"])),
                     "mz_r2": mz_r2(s["y_true_log"].to_numpy(), s["y_pred_log"].to_numpy())})
    lb = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
    print("\n=== POOLED OOS QLIKE (calibrated, common support) ===")
    print(lb.round(4).to_string(index=False))

    print("\n=== Diebold-Mariano vs Meridian (H1: Meridian lower loss) ===")
    md = loss["Meridian"]
    for m in models:
        if m == "Meridian":
            continue
        j = md.merge(loss[m], on=["asset", "date"], suffixes=("_mer", "_b"))
        dm, p = diebold_mariano(j["loss_mer"].to_numpy(), j["loss_b"].to_numpy())
        rel = (j["loss_b"].mean() - j["loss_mer"].mean()) / j["loss_b"].mean() * 100
        win = "WIN" if (rel >= 5 and p is not None and p < 0.05) else ""
        print(f"  vs {m:11s} QLIKE {j['loss_mer'].mean():.4f} vs {j['loss_b'].mean():.4f} "
              f"rel {rel:+6.2f}%  DM={dm:+.2f} p={p:.4f}  {win}")

    j = md.merge(loss["HAR-RV"], on=["asset", "date"], suffixes=("_mer", "_har"))
    dm, p = diebold_mariano(j["loss_mer"].to_numpy(), j["loss_har"].to_numpy())
    rel = (j["loss_har"].mean() - j["loss_mer"].mean()) / j["loss_har"].mean() * 100
    c1, c2 = rel >= 5.0, (p is not None and p < 0.05)
    print("\n=== PRE-REGISTERED PRIMARY VERDICT (vs HAR-RV, calibrated) ===")
    print(f"  QLIKE reduction: {rel:+.2f}%  (need >= +5%)  {'PASS' if c1 else 'FAIL'}")
    print(f"  DM p-value:      {p:.4f}     (need < 0.05)  {'PASS' if c2 else 'FAIL'}")
    print(f"\n  >>> Meridian {'BEATS' if (c1 and c2) else 'DOES NOT BEAT'} HAR-RV (calibrated).")


if __name__ == "__main__":
    main()
