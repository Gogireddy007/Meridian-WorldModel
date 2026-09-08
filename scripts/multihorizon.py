"""Multi-horizon forecast edge: does the ML edge over HAR grow at weekly/monthly
horizons (research pass 7: gains concentrate at longer horizons)? Forecasts h-day-ahead
AVERAGE RV for h=1,5,22; HAR(OLS) vs RF+SemiVar; QLIKE edge with h-lag Newey-West DM.
Validates the honest path to a larger '2x better prediction' multiplier.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import RandomForestRegressor

from meridian.data import load_all
from meridian.evalproto import WalkForward, diebold_mariano, qlike
from meridian.features import build_asset_frame, EPS
from scripts.run_semivar_ml import add_semivar, BASE, SV

warnings.filterwarnings("ignore")


def har_predict(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    resid = ytr - A @ beta
    return (np.column_stack([np.ones(len(Xte)), Xte]) @ beta), 0.5 * resid.var()


def main():
    d = load_all(); wf = WalkForward(min_train=1000, test_size=126, embargo=22)
    HARF = ["har_d", "har_w", "har_m"]; feats = BASE + SV
    print(f"{'horizon':>8} {'HAR_QLIKE':>10} {'RF+SV_QLIKE':>12} {'edge_vs_HAR':>12} {'sigma':>7}")
    for h in (1, 5, 22):
        qh_all, qr_all = [], []
        lh, lr = [], []
        for asset, ohlc in d["prices"].items():
            f = add_semivar(build_asset_frame(ohlc, d["macro"]))
            rv = np.exp(f["log_rv"])
            yh = np.log(rv.shift(-1).rolling(h).mean().shift(-(h - 1)) + EPS)   # avg RV over t+1..t+h
            f = f.assign(yh=yh)
            df = f[feats + HARF + ["yh"]].replace([np.inf, -np.inf], np.nan).dropna()
            X = df[feats].to_numpy(); Xh = df[HARF].to_numpy(); y = df["yh"].to_numpy()
            for tr, te in wf.split(len(df)):
                ph, b = har_predict(Xh[tr], y[tr], Xh[te])
                m = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=15,
                                          n_jobs=-1, random_state=0).fit(X[tr], y[tr])
                pr = m.predict(X[te])
                rvt = np.exp(y[te])
                rb = y[tr] - m.predict(X[tr]); rbias = 0.5 * rb.var()
                lh.append(qlike(rvt, np.exp(ph + b))); lr.append(qlike(rvt, np.exp(pr + rbias)))
        Lh = np.concatenate(lh); Lr = np.concatenate(lr)
        qh, qr = np.nanmean(Lh), np.nanmean(Lr)
        edge = (qh - qr) / qh * 100
        dm, p = diebold_mariano(Lr, Lh, h=h)
        sig = abs(stats.norm.ppf(p)) if p and 0 < p < 1 else np.inf
        print(f"{h:>6}d  {qh:>10.4f} {qr:>12.4f} {edge:>+11.2f}% {sig:>6.1f}σ")
    print("\n  (If edge grows with horizon, weekly/monthly is the honest path to a larger multiplier.)")


if __name__ == "__main__":
    main()
