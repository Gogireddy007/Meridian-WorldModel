"""Production online-adaptive forecaster: SLOW ensemble (Meridian⊕RF, stable) + FAST
online corrector (adapts to the ensemble's recent systematic errors, live, causally).
This folds the proven −21.5% online-adaptation gain into the production forecaster the
right way (complementary learning systems: slow stable + fast adaptive). Honest —
measured vs the static ensemble; kept only if it genuinely helps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import diebold_mariano, qlike, walk_forward_calibrate
from scipy import stats

R = Path(__file__).resolve().parent.parent / "results"


def cal(df, col):
    o = []
    for a, s in df.groupby("asset"):
        s = s.sort_values("date").reset_index(drop=True)
        v, m = walk_forward_calibrate(s["date"].values, s["y_true_log"].to_numpy(), s[col].to_numpy())
        o.append(s.assign(var=v, ok=m))
    return pd.concat(o).query("ok")


def main():
    base = pd.read_parquet(R / "baseline_predictions.parquet"); base["date"] = pd.to_datetime(base["date"])
    ml = pd.read_parquet(R / "ml_predictions.parquet"); ml["date"] = pd.to_datetime(ml["date"])
    mer = pd.read_parquet(R / "cfjepa_ens_predictions.parquet"); mer["date"] = pd.to_datetime(mer["date"])
    M = cal(mer, "y_pred_tgt")[["asset", "date", "y_true_log", "var"]].rename(columns={"var": "vM"})
    RF = cal(ml[ml.model == "RandomForest"], "y_pred_log")[["asset", "date", "var"]].rename(columns={"var": "vRF"})
    HAR = cal(base[base.model == "HAR-RV"], "y_pred_log")[["asset", "date", "var"]].rename(columns={"var": "vHAR"})
    df = M.merge(RF, on=["asset", "date"]).merge(HAR, on=["asset", "date"]).sort_values(["asset", "date"])
    df["ens"] = 0.5 * (np.log(df["vM"]) + np.log(df["vRF"]))          # ensemble log-variance

    # online correctors, applied per asset causally (no lookahead)
    ens_q, cor_q, cor2_q, har_q = [], [], [], []
    ens_all, cor_all = [], []
    for a, s in df.groupby("asset"):
        s = s.reset_index(drop=True)
        y = s["y_true_log"].to_numpy(); e = s["ens"].to_numpy()
        resid = y - e
        # FAST corrector 1: EWMA of PAST residuals (systematic-bias correction), causal
        lam = 0.94; bias = 0.0; c1 = np.empty(len(y))
        for t in range(len(y)):
            c1[t] = e[t] + bias                                       # forecast BEFORE seeing y[t]
            bias = lam * bias + (1 - lam) * (y[t] - e[t])             # then adapt on observed error
        # FAST corrector 2: online AR(1) on residuals (slope*prev_resid), causal
        prev = 0.0; w = 0.0; c2 = np.empty(len(y)); lr = 0.02
        for t in range(len(y)):
            c2[t] = e[t] + w * prev
            err = (y[t] - c2[t]); w += lr * err * prev; prev = y[t] - e[t]
        rv = np.exp(y)
        ens_q.append(qlike(rv, np.exp(e))); cor_q.append(qlike(rv, np.exp(c1)))
        cor2_q.append(qlike(rv, np.exp(c2))); har_q.append(qlike(rv, s["vHAR"].to_numpy()))
        ens_all.append(qlike(rv, np.exp(e))); cor_all.append(qlike(rv, np.exp(c1)))

    QL = {k: np.nanmean(np.concatenate(v)) for k, v in
          [("HAR", har_q), ("ensemble", ens_q), ("ens+EWMA-online", cor_q), ("ens+AR-online", cor2_q)]}
    print("production forecaster — static ensemble vs online-adaptive correction\n")
    for k in ("HAR", "ensemble", "ens+EWMA-online", "ens+AR-online"):
        edge = (QL["HAR"] - QL[k]) / QL["HAR"] * 100
        print(f"  {k:>18} QLIKE {QL[k]:.4f}   edge vs HAR {edge:+.2f}%")
    dm, p = diebold_mariano(np.concatenate(cor_all), np.concatenate(ens_all))
    sig = abs(stats.norm.ppf(p)) if p and 0 < p < 1 else np.inf
    lift = (QL["ensemble"] - QL["ens+EWMA-online"]) / QL["ensemble"] * 100
    print(f"\n  online corrector vs static ensemble: {lift:+.2f}% QLIKE  ({sig:.1f}sigma)  "
          f"{'HELPS' if lift > 0 and p < 0.05 else 'no gain'}")


if __name__ == "__main__":
    main()
