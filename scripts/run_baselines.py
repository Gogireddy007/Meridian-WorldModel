"""Fit all baselines across the universe under the pre-registered protocol.

Writes results/baseline_predictions.parquet (per-asset per-model OOS preds)
and prints a pooled leaderboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meridian.baselines import BASELINES, run_model
from meridian.data import load_all
from meridian.evalproto import WalkForward, qlike, mse_log
from meridian.features import build_asset_frame

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    d = load_all()
    macro = d["macro"]
    wf = WalkForward(min_train=1000, test_size=126, embargo=22)

    rows = []          # long-form OOS predictions
    summaries = []
    for asset, ohlc in d["prices"].items():
        frame = build_asset_frame(ohlc, macro)
        for mcls in BASELINES:
            res = run_model(mcls, frame, asset, wf)
            summaries.append(res.summary())
            rows.append(pd.DataFrame({
                "date": res.dates,
                "asset": asset,
                "model": res.name,
                "y_true_log": res.y_true_log,
                "y_pred_log": res.y_pred_log,
                "logvar_bias": res.logvar_bias,
            }))
        print(f"  done {asset}", flush=True)

    preds = pd.concat(rows, ignore_index=True)
    preds.to_parquet(RESULTS / "baseline_predictions.parquet")

    summ = pd.DataFrame(summaries)
    # pooled leaderboard: average of per-asset QLIKE (equal asset weight)
    lb = (summ.groupby("model")
              .agg(qlike=("qlike", "mean"),
                   mse_log=("mse_log", "mean"),
                   mz_r2=("mz_r2", "mean"),
                   n=("n", "sum"))
              .sort_values("qlike"))
    print("\n=== POOLED OOS LEADERBOARD (equal-asset-weight) ===")
    print(lb.round(4).to_string())

    summ.to_parquet(RESULTS / "baseline_summary.parquet")
    print("\nsaved:", RESULTS / "baseline_predictions.parquet")
    return lb


if __name__ == "__main__":
    main()
