"""Head-to-head: Meridian vs baselines on the pre-registered volatility criteria.

Aligns Meridian and baseline OOS predictions on the SAME (asset, date) rows,
computes pooled QLIKE, and runs Diebold-Mariano vs HAR-RV. Prints the verdict
against PREREGISTRATION.md (>=5% QLIKE reduction AND DM p<0.05).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import diebold_mariano, mz_r2, qlike

RESULTS = Path(__file__).resolve().parent.parent / "results"
MER_FILE = os.environ.get("MERIDIAN_PRED", "meridian_predictions.parquet")


def load():
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    mer = pd.read_parquet(RESULTS / MER_FILE)
    keep = ["date", "asset", "model", "y_true_log", "y_pred_log", "logvar_bias"]
    both = pd.concat([base[keep], mer[keep]], ignore_index=True)
    both["date"] = pd.to_datetime(both["date"])
    return both


def var_pred(df):
    return np.exp(df["y_pred_log"].to_numpy() + df["logvar_bias"].to_numpy())


def qlike_series(df):
    rv = np.exp(df["y_true_log"].to_numpy())
    return qlike(rv, var_pred(df))


def main():
    df = load()
    models = df["model"].unique().tolist()

    # Common (asset,date) support across ALL models for a fair pooled comparison
    pivot = df.pivot_table(index=["asset", "date"], columns="model",
                           values="y_pred_log", aggfunc="first")
    common = pivot.dropna().index
    df = df.set_index(["asset", "date"]).loc[
        df.set_index(["asset", "date"]).index.isin(common)].reset_index()
    print(f"common OOS rows across all models: {len(common)}  "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    # per-model pooled metrics
    rows = []
    per_model_loss = {}
    for m in models:
        sub = df[df["model"] == m].sort_values(["asset", "date"]).reset_index(drop=True)
        ql = qlike_series(sub)
        per_model_loss[m] = (sub[["asset", "date"]], ql)
        rows.append({"model": m, "n": len(sub),
                     "qlike": float(np.nanmean(ql)),
                     "mz_r2": mz_r2(sub["y_true_log"].to_numpy(), sub["y_pred_log"].to_numpy())})
    lb = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
    print("\n=== POOLED OOS (common support) ===")
    print(lb.round(4).to_string(index=False))

    # DM tests: Meridian vs each baseline (aligned rows)
    print("\n=== Diebold-Mariano vs Meridian (H1: Meridian lower loss) ===")
    key = ["asset", "date"]
    mmeta, mloss = per_model_loss["Meridian"]
    mdf = mmeta.assign(loss=mloss)
    for m in models:
        if m == "Meridian":
            continue
        bmeta, bloss = per_model_loss[m]
        bdf = bmeta.assign(loss=bloss)
        j = mdf.merge(bdf, on=key, suffixes=("_mer", "_base"))
        dm, p = diebold_mariano(j["loss_mer"].to_numpy(), j["loss_base"].to_numpy())
        q_mer, q_base = j["loss_mer"].mean(), j["loss_base"].mean()
        rel = (q_base - q_mer) / q_base * 100
        verdict = "WIN" if (rel >= 5 and p is not None and p < 0.05) else ""
        print(f"  vs {m:11s}  QLIKE {q_mer:.4f} vs {q_base:.4f}  "
              f"rel {rel:+6.2f}%  DM={dm:+.2f} p={p:.4f}  {verdict}")

    # Pre-registered verdict vs HAR-RV
    print("\n=== PRE-REGISTERED PRIMARY VERDICT (vs HAR-RV) ===")
    b = per_model_loss["HAR-RV"]
    j = mdf.merge(b[0].assign(loss=b[1]), on=key, suffixes=("_mer", "_har"))
    dm, p = diebold_mariano(j["loss_mer"].to_numpy(), j["loss_har"].to_numpy())
    rel = (j["loss_har"].mean() - j["loss_mer"].mean()) / j["loss_har"].mean() * 100
    crit1 = rel >= 5.0
    crit2 = (p is not None) and (p < 0.05)
    print(f"  QLIKE reduction vs HAR-RV: {rel:+.2f}%   (need >= +5%)   {'PASS' if crit1 else 'FAIL'}")
    print(f"  Diebold-Mariano p-value:   {p:.4f}      (need < 0.05)   {'PASS' if crit2 else 'FAIL'}")
    print(f"\n  >>> Meridian {'BEATS' if (crit1 and crit2) else 'DOES NOT BEAT'} HAR-RV "
          f"on volatility by the pre-registered margin.")


if __name__ == "__main__":
    main()
