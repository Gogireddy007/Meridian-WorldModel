"""LightGBM baseline, run in a separate process by scripts/benchmark_vol.py.

LightGBM and PyTorch (MPS) cannot share one process on macOS: each ships its own
OpenMP runtime, and whichever loads second crashes. This module imports only
numpy/pandas/lightgbm and is executed as a subprocess:

    python -m meridian.lgbm_worker <train.parquet> <test.parquet> <out.npy> <assets,comma,sep>
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ML_FEATS = ["har_d", "har_w", "har_m", "lev_d", "lev_w", "lev_m", "ret_abs", "vix",
            "down", "har_d_dn"]


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, assets: list[str]) -> np.ndarray:
    def frame(df):
        X = df[ML_FEATS].copy()
        X["asset_code"] = pd.Categorical(df["asset"].where(df["asset"].isin(assets)), categories=assets)
        return X
    dts = np.sort(tr["date"].unique())
    val_start = dts[int(len(dts) * 0.9)]
    # time-ordered validation slice; fit rows end (date and 1-day label) before it, with a gap
    fit_m = ((tr["date"] < val_start - np.timedelta64(30, "D")) & (tr["end_1"] < val_start)).to_numpy()
    val_m = (tr["date"] >= val_start).to_numpy()
    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
    model.fit(frame(tr[fit_m]), tr["y"][fit_m],
              eval_set=[(frame(tr[val_m]), tr["y"][val_m])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    jb = 0.5 * float(np.var(tr["y"][val_m].to_numpy() - model.predict(frame(tr[val_m]))))
    return model.predict(frame(te)) + jb


if __name__ == "__main__":
    tr_path, te_path, out_path, assets = sys.argv[1:5]
    pred = fit_predict(pd.read_parquet(tr_path), pd.read_parquet(te_path), assets.split(","))
    np.save(out_path, pred)
