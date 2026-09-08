"""Strong MODERN ML baselines (Random Forest, Gradient Boosting) on the same purged
walk-forward, so the 'Meridian beats other models' claim is DIRECT and reproducible in
our codebase — not reliant on external citations. Predicts log RV_{t+1} from the full
feature set; saved in the baseline schema for the prediction scorecard.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from meridian.data import load_all
from meridian.evalproto import WalkForward
from meridian.features import build_asset_frame

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "results"
FEATS = ["har_d", "har_w", "har_m", "log_rv", "ret_abs", "ret_5", "rv_cc", "vix", "term", "vix_chg"]

MODELS = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, max_depth=6,
                                                   min_samples_leaf=20, n_jobs=-1, random_state=0),
    "GradBoost": lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                       max_iter=300, l2_regularization=1.0, random_state=0),
}


def main():
    d = load_all()
    wf = WalkForward(min_train=1000, test_size=126, embargo=22)
    rows = []
    for asset, ohlc in d["prices"].items():
        f = build_asset_frame(ohlc, d["macro"])
        cols = FEATS + ["y"]
        df = f[cols].replace([np.inf, -np.inf], np.nan).dropna()
        X = df[FEATS].to_numpy(); y = df["y"].to_numpy(); idx = df.index
        for name, mk in MODELS.items():
            dates, ytrue, ypred = [], [], []
            for tr, te in wf.split(len(df)):
                m = mk().fit(X[tr], y[tr])
                p = m.predict(X[te])
                dates.append(idx[te]); ytrue.append(y[te]); ypred.append(p)
            if dates:
                rows.append(pd.DataFrame({
                    "date": np.concatenate([x.values for x in dates]),
                    "asset": asset, "model": name,
                    "y_true_log": np.concatenate(ytrue), "y_pred_log": np.concatenate(ypred),
                    "logvar_bias": 0.0}))
        print(f"  done {asset}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RESULTS / "ml_predictions.parquet")
    print(f"saved {len(out)} rows, models: {out.model.unique().tolist()}")


if __name__ == "__main__":
    main()
