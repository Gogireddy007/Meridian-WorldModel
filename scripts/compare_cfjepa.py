"""CF-JEPA test: does reading the vol forecast off the EMA target encoder beat
the online encoder? Compares the two read-offs (y_pred_log vs y_pred_tgt) from the
dual-head run under the identical walk-forward calibration, plus HAR-RV reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import diebold_mariano, mz_r2, qlike, walk_forward_calibrate

RESULTS = Path(__file__).resolve().parent.parent / "results"


def cal(df, col):
    outs = []
    for a, sub in df.groupby("asset"):
        sub = sub.sort_values("date").reset_index(drop=True)
        var, mask = walk_forward_calibrate(sub["date"].values, sub["y_true_log"].to_numpy(),
                                           sub[col].to_numpy())
        outs.append(sub.assign(var=var, ok=mask))
    d = pd.concat(outs)
    d = d[d.ok]
    return d


def main():
    df = pd.read_parquet(RESULTS / "cfjepa_predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    assert "y_pred_tgt" in df.columns, "run with MERIDIAN_DUALVOL=1 first"

    on = cal(df, "y_pred_log").rename(columns={"var": "var_on"})
    tg = cal(df, "y_pred_tgt").rename(columns={"var": "var_tg"})
    j = on[["asset", "date", "y_true_log", "var_on"]].merge(
        tg[["asset", "date", "var_tg"]], on=["asset", "date"])
    # common rows with HAR
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    har = base[base.model == "HAR-RV"].copy(); har["date"] = pd.to_datetime(har["date"])
    harc = cal(har.rename(columns={"y_true_log": "y_true_log"}), "y_pred_log").rename(columns={"var": "var_har"})
    j = j.merge(harc[["asset", "date", "var_har"]], on=["asset", "date"])

    rv = np.exp(j["y_true_log"].to_numpy())
    q_on = qlike(rv, j["var_on"].to_numpy())
    q_tg = qlike(rv, j["var_tg"].to_numpy())
    q_har = qlike(rv, j["var_har"].to_numpy())
    print(f"common rows: {len(j)}")
    print(f"\n  online-encoder readout   QLIKE = {q_on.mean():.4f}")
    print(f"  EMA-target readout       QLIKE = {q_tg.mean():.4f}")
    print(f"  HAR-RV                   QLIKE = {q_har.mean():.4f}")

    dm, p = diebold_mariano(q_tg, q_on)   # H1: target lower than online
    rel = (q_on.mean() - q_tg.mean()) / q_on.mean() * 100
    print(f"\n  target vs online: rel {rel:+.2f}%  DM={dm:+.2f} p={p:.4f}")
    print(f"  >>> CF-JEPA EMA-routing {'HELPS' if (rel > 0 and p < 0.05) else 'does NOT help'} "
          f"on daily vol for Meridian.")
    print("\n  (Negative/null = confirms the online-encoder design; the research flagged"
          "\n   the CF-JEPA asymmetry as benchmark-specific and unreplicated.)")


if __name__ == "__main__":
    main()
