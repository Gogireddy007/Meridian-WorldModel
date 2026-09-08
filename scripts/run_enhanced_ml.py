"""Enhanced ML baselines with IMPLIED-VOL features (VIX term structure) — the
strongest evidence-backed incremental predictor of realized vol (forward-looking,
beyond past RV). Adds VIX level + term-structure slope + short-term stress spread,
refits RF/GBM walk-forward, saves for the ensemble/scorecard.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from meridian.data import fetch_yahoo, load_all, DATA_DIR
from meridian.evalproto import WalkForward
from meridian.features import build_asset_frame

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "results"
BASE_FEATS = ["har_d", "har_w", "har_m", "log_rv", "ret_abs", "ret_5", "rv_cc", "vix", "term", "vix_chg"]
VIX_FEATS = ["vix_slope", "vix9d_sp", "vix_lvl", "vix_mom"]


def vix_term_structure():
    """VIX curve features (forward-looking): slope (VIX3M-VIX), short stress (VIX-VIX9D)."""
    out = {}
    for t, n in [("^VIX", "vix30"), ("^VIX9D", "vix9d"), ("^VIX3M", "vix3m")]:
        f = DATA_DIR / f"ts_{n}.parquet"
        if f.exists():
            out[n] = pd.read_parquet(f)["c"]
        else:
            s = fetch_yahoo(t, start="2007-01-01")["close"].rename("c")
            s.to_frame().to_parquet(f); out[n] = s
    df = pd.DataFrame(out).sort_index().ffill()
    feat = pd.DataFrame(index=df.index)
    feat["vix_lvl"] = np.log(df["vix30"].clip(lower=1))
    feat["vix_slope"] = (df["vix3m"] - df["vix30"]) / df["vix30"]      # term structure (contango>0)
    feat["vix9d_sp"] = (df["vix30"] - df["vix9d"]) / df["vix30"]       # short-term stress
    feat["vix_mom"] = feat["vix_lvl"].diff(5)                          # vix momentum
    return feat


def main():
    d = load_all()
    vts = vix_term_structure()
    wf = WalkForward(min_train=1000, test_size=126, embargo=22)
    models = {
        "RandomForest+IV": lambda: RandomForestRegressor(n_estimators=200, max_depth=6,
                                                         min_samples_leaf=20, n_jobs=-1, random_state=0),
        "GradBoost+IV": lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                             max_iter=300, l2_regularization=1.0, random_state=0),
    }
    feats = BASE_FEATS + VIX_FEATS
    rows = []
    for asset, ohlc in d["prices"].items():
        f = build_asset_frame(ohlc, d["macro"])
        f = f.join(vts.reindex(f.index).ffill())
        df = f[feats + ["y"]].replace([np.inf, -np.inf], np.nan).dropna()
        X = df[feats].to_numpy(); y = df["y"].to_numpy(); idx = df.index
        for name, mk in models.items():
            dts, yt, yp = [], [], []
            for tr, te in wf.split(len(df)):
                m = mk().fit(X[tr], y[tr]); p = m.predict(X[te])
                dts.append(idx[te]); yt.append(y[te]); yp.append(p)
            if dts:
                rows.append(pd.DataFrame({"date": np.concatenate([x.values for x in dts]),
                    "asset": asset, "model": name, "y_true_log": np.concatenate(yt),
                    "y_pred_log": np.concatenate(yp), "logvar_bias": 0.0}))
        print(f"  done {asset}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RESULTS / "enhanced_ml_predictions.parquet")
    print(f"saved {len(out)} rows with implied-vol features: {out.model.unique().tolist()}")


if __name__ == "__main__":
    main()
