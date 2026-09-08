"""Realized-semivariance features (the #1 evidence-backed lever, Patton-Sheppard):
decompose RV into good/bad; NEGATIVE (bad) semivariance carries more predictive info.
Daily proxy (close-to-close): neg = ret^2*(ret<0), pos = ret^2*(ret>0), with HAR
components. Refit a LEAN RandomForest (base + neg-semivar HAR) — kept lean because the
naive implied-vol feature dump overfit. Saved for ensembling with Meridian.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import RandomForestRegressor

from meridian.data import load_all
from meridian.evalproto import WalkForward
from meridian.features import build_asset_frame, EPS

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "results"
BASE = ["har_d", "har_w", "har_m", "log_rv", "ret_abs", "ret_5", "vix"]
SV = ["nsv_d", "nsv_w", "nsv_m", "psv_d"]                # negative-semivar HAR + positive-semivar day


def add_semivar(f):
    r = f["ret"]
    nsv = (r.clip(upper=0) ** 2)                          # bad (downside) semivariance
    psv = (r.clip(lower=0) ** 2)                          # good (upside) semivariance
    f["nsv_d"] = np.log(nsv + EPS)
    f["nsv_w"] = np.log(nsv.rolling(5).mean() + EPS)
    f["nsv_m"] = np.log(nsv.rolling(22).mean() + EPS)
    f["psv_d"] = np.log(psv + EPS)
    return f


def main():
    d = load_all()
    wf = WalkForward(min_train=1000, test_size=126, embargo=22)
    feats = BASE + SV
    rows = []
    for asset, ohlc in d["prices"].items():
        f = add_semivar(build_asset_frame(ohlc, d["macro"]))
        df = f[feats + ["y"]].replace([np.inf, -np.inf], np.nan).dropna()
        X = df[feats].to_numpy(); y = df["y"].to_numpy(); idx = df.index
        dts, yt, yp = [], [], []
        for tr, te in wf.split(len(df)):
            m = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=15,
                                      n_jobs=-1, random_state=0).fit(X[tr], y[tr])
            dts.append(idx[te]); yt.append(y[te]); yp.append(m.predict(X[te]))
        if dts:
            rows.append(pd.DataFrame({"date": np.concatenate([x.values for x in dts]),
                "asset": asset, "model": "RF+SemiVar", "y_true_log": np.concatenate(yt),
                "y_pred_log": np.concatenate(yp), "logvar_bias": 0.0}))
        print(f"  done {asset}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RESULTS / "semivar_ml_predictions.parquet")
    print(f"saved {len(out)} rows: RF+SemiVar (bad/good semivariance features)")


if __name__ == "__main__":
    main()
