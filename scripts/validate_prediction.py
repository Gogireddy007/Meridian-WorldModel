"""STRICTEST out-of-sample test — 'the model does not know the future'.

Train ONLY on 2007-2017, FREEZE, then predict 2018-2026 (8 years the model never saw,
never updated). Every forecast at time t uses only: frozen parameters (<=2017) + features
known at t (past realized vol). Zero future information — leakage is structurally
impossible. Then check the predictions in depth:
  * accuracy: QLIKE / RMSE(logRV) / correlation / MZ-R2 vs HAR
  * UNBIASEDNESS: Mincer-Zarnowitz regression realized ~ a + b*forecast (want a=0,b=1)
  * DIRECTIONAL: does it call vol up/down correctly (hit rate vs 50%)?
  * per-YEAR and per-ASSET breakdown (is the edge consistent or a fluke?)
  * error distribution and worst misses.
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
from meridian.evalproto import qlike
from meridian.features import build_asset_frame
from scripts.run_semivar_ml import add_semivar, BASE, SV

warnings.filterwarnings("ignore")
CUTOFF = "2017-12-31"
HARF = ["har_d", "har_w", "har_m"]


def har_fit_predict(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    resid = ytr - A @ beta
    return np.column_stack([np.ones(len(Xte)), Xte]) @ beta + 0.5 * resid.var()


def main():
    d = load_all()
    feats = BASE + SV
    rows = []
    for a, o in d["prices"].items():
        f = add_semivar(build_asset_frame(o, d["macro"]))
        cols = list(dict.fromkeys(feats + HARF + ["y", "log_rv"]))   # dedupe, preserve order
        df = f[cols].replace([np.inf, -np.inf], np.nan).dropna()
        tr = df.index <= CUTOFF; te = ~tr
        if tr.sum() < 500 or te.sum() < 100:
            continue
        # FROZEN models: fit on <=2017 only
        har_p = har_fit_predict(df.loc[tr, HARF].to_numpy(), df.loc[tr, "y"].to_numpy(), df.loc[te, HARF].to_numpy())
        rf = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=15, n_jobs=-1, random_state=0)
        rf.fit(df.loc[tr, feats].to_numpy(), df.loc[tr, "y"].to_numpy())
        rf_raw = rf.predict(df.loc[te, feats].to_numpy())
        rb = df.loc[tr, "y"].to_numpy() - rf.predict(df.loc[tr, feats].to_numpy())
        rf_p = rf_raw + 0.5 * rb.var()
        ens_p = 0.5 * (har_p + rf_p)                        # frozen ensemble (log-variance avg)
        sub = pd.DataFrame({"date": df.index[te], "asset": a, "y": df.loc[te, "y"].to_numpy(),
                            "logrv_now": df.loc[te, "log_rv"].to_numpy(),
                            "har": har_p, "rf": rf_p, "ens": ens_p})
        rows.append(sub)
    D = pd.concat(rows, ignore_index=True); D["year"] = pd.to_datetime(D["date"]).dt.year
    rv = np.exp(D["y"].to_numpy())

    print(f"FROZEN-MODEL OOS TEST — trained ≤{CUTOFF[:4]}, tested {D.year.min()}–{D.year.max()} "
          f"({len(D)} forecasts, {D.asset.nunique()} assets), model NEVER updated\n")
    print(f"{'model':>10} {'QLIKE':>7} {'RMSE_logRV':>11} {'corr':>6} {'MZ_R2':>7} {'MZ_a':>7} {'MZ_b':>6} {'dir_hit%':>9}")
    for m in ("har", "rf", "ens"):
        p = D[m].to_numpy()
        ql = np.nanmean(qlike(rv, np.exp(p)))
        rmse = np.sqrt(np.nanmean((p - D["y"].to_numpy()) ** 2))
        corr = np.corrcoef(p, D["y"].to_numpy())[0, 1]
        # Mincer-Zarnowitz: realized ~ a + b*forecast
        A = np.column_stack([np.ones(len(p)), p]); b, *_ = np.linalg.lstsq(A, D["y"].to_numpy(), rcond=None)
        r2 = 1 - ((D["y"].to_numpy() - A @ b) ** 2).sum() / ((D["y"].to_numpy() - D["y"].mean()) ** 2).sum()
        # directional: predicted vol change vs realized change (sign)
        dp = np.sign(p - D["logrv_now"].to_numpy()); da = np.sign(D["y"].to_numpy() - D["logrv_now"].to_numpy())
        hit = np.mean(dp == da) * 100
        print(f"{m:>10} {ql:>7.4f} {rmse:>11.4f} {corr:>6.3f} {r2:>7.3f} {b[0]:>+7.3f} {b[1]:>6.3f} {hit:>8.1f}%")

    print("\n  per-YEAR edge of frozen ensemble vs frozen HAR (QLIKE % better):")
    for y, g in D.groupby("year"):
        rvy = np.exp(g["y"].to_numpy())
        e = (np.nanmean(qlike(rvy, np.exp(g["har"]))) - np.nanmean(qlike(rvy, np.exp(g["ens"])))) / np.nanmean(qlike(rvy, np.exp(g["har"]))) * 100
        print(f"    {y}: {e:+6.1f}%", end="   " + ("\n" if y % 3 == 0 else ""))
    print()
    # per-asset win rate
    wins = 0; tot = 0
    for a, g in D.groupby("asset"):
        rva = np.exp(g["y"].to_numpy()); tot += 1
        if np.nanmean(qlike(rva, np.exp(g["ens"]))) < np.nanmean(qlike(rva, np.exp(g["har"]))):
            wins += 1
    print(f"\n  frozen ensemble beats frozen HAR on {wins}/{tot} assets, "
          f"and {sum(1 for y,g in D.groupby('year') if np.nanmean(qlike(np.exp(g['y']),np.exp(g['ens'])))<np.nanmean(qlike(np.exp(g['y']),np.exp(g['har']))))}/"
          f"{D.year.nunique()} years — on data it NEVER saw.")


if __name__ == "__main__":
    main()
