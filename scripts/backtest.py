"""Economic backtest — convert the vol forecast into a strategy and measure the
metrics a PM actually cares about: annualized Sharpe, alpha vs buy-&-hold (with a
Newey-West t-stat = the "sigma"), information ratio, max drawdown, turnover — net of
transaction costs, on the same purged OOS.

Strategy: volatility-managed positioning (Moreira & Muir 2017). Scale each asset's
exposure to a target annualized vol using the next-day vol forecast:
    w_t = clip(target_vol / (sigma_hat_t * sqrt(252)), 0, w_max)
Daily P&L = w_t * r_{t+1} - cost_bps * |w_t - w_{t-1}|. Portfolio = equal weight.

Compares Meridian's forecast vs HAR's vs buy-&-hold. Honest by construction: if the
better forecast does not produce better risk-adjusted P&L net of costs, it shows.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.evalproto import walk_forward_calibrate

RESULTS = Path(__file__).resolve().parent.parent / "results"
MER_FILE = os.environ.get("MERIDIAN_PRED", "cfjepa_ens_predictions.parquet")
MER_COL = os.environ.get("MERIDIAN_COL", "y_pred_tgt")   # EMA-target readout
TARGET_VOL = 0.10       # 10% annualized
W_MAX = 2.0
COST_BPS = float(os.environ.get("COST_BPS", "1.0")) / 1e4
ANN = 252


def calibrated_var(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    outs = []
    for a, s in pred.groupby("asset"):
        s = s.sort_values("date").reset_index(drop=True)
        var, mask = walk_forward_calibrate(s["date"].values, s["y_true_log"].to_numpy(),
                                           s[col].to_numpy())
        outs.append(s.assign(pred_var=var, ok=mask))
    d = pd.concat(outs)
    return d[d.ok]


def nw_tstat(x, lag=10):
    x = x[np.isfinite(x)]; n = len(x); mu = x.mean()
    g0 = np.mean((x - mu) ** 2); var = g0
    for k in range(1, lag + 1):
        c = np.mean((x[k:] - mu) * (x[:-k] - mu))
        var += 2 * (1 - k / (lag + 1)) * c
    se = np.sqrt(var / n)
    return mu / se if se > 0 else np.nan


def perf(ret, bench=None):
    ret = np.asarray(ret); m = np.isfinite(ret); ret = ret[m]
    sharpe = ret.mean() / ret.std() * np.sqrt(ANN) if ret.std() > 0 else np.nan
    curve = np.cumprod(1 + ret); dd = (curve / np.maximum.accumulate(curve) - 1).min()
    out = {"ann_ret": ret.mean() * ANN, "ann_vol": ret.std() * np.sqrt(ANN),
           "sharpe": sharpe, "maxDD": dd}
    if bench is not None:
        b = np.asarray(bench)[m]
        # alpha vs benchmark: OLS intercept of strat on bench
        X = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(X, ret, rcond=None)
        resid = ret - X @ beta
        out["alpha_ann"] = beta[0] * ANN
        out["alpha_t"] = nw_tstat(resid + beta[0])   # t-stat of the intercept series
    return out


def vol_managed(pred_var: pd.DataFrame, rnext: pd.DataFrame):
    """Return per-day equal-weight portfolio returns for the managed strategy and
    buy-&-hold, aligned on common (asset,date)."""
    df = pred_var.merge(rnext, on=["asset", "date"], how="inner").dropna(subset=["r_next"])
    df["sigma_ann"] = np.sqrt(df["pred_var"].clip(1e-8)) * np.sqrt(ANN)
    df["w"] = (TARGET_VOL / df["sigma_ann"]).clip(0, W_MAX)
    parts_managed, parts_bh = [], []
    for a, s in df.groupby("asset"):
        s = s.sort_values("date")
        w = s["w"].to_numpy(); r = s["r_next"].to_numpy()
        turn = np.abs(np.diff(w, prepend=w[0]))
        managed = w * r - COST_BPS * turn
        parts_managed.append(pd.Series(managed, index=s["date"]))
        parts_bh.append(pd.Series(r, index=s["date"]))
    M = pd.concat(parts_managed, axis=1).mean(axis=1)   # equal-weight portfolio
    B = pd.concat(parts_bh, axis=1).mean(axis=1)
    turnover = float(np.mean([np.abs(np.diff(g.sort_values("date")["w"])).mean()
                              for _, g in df.groupby("asset")]))
    return M.sort_index(), B.sort_index(), turnover


def main():
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet")
    base["date"] = pd.to_datetime(base["date"])
    mer = pd.read_parquet(RESULTS / MER_FILE); mer["date"] = pd.to_datetime(mer["date"])
    col = MER_COL if MER_COL in mer.columns else "y_pred_log"

    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}
    rnext = pd.concat([frames[a][["r_next"]].assign(asset=a).reset_index().rename(
        columns={"index": "date"}) for a in frames], ignore_index=True)
    rnext.columns = ["date", "r_next", "asset"]; rnext["date"] = pd.to_datetime(rnext["date"])

    har_var = calibrated_var(base[base.model == "HAR-RV"].copy(), "y_pred_log")
    mer_var = calibrated_var(mer, col)

    M_mer, B, turn_m = vol_managed(mer_var[["asset", "date", "pred_var"]], rnext)
    M_har, _, turn_h = vol_managed(har_var[["asset", "date", "pred_var"]], rnext)
    # align all on common dates
    idx = M_mer.index.intersection(M_har.index).intersection(B.index)
    M_mer, M_har, B = M_mer.loc[idx], M_har.loc[idx], B.loc[idx]

    print(f"vol-managed backtest  (target {TARGET_VOL:.0%} ann, cost {COST_BPS*1e4:.0f}bp, "
          f"{idx.min().date()}→{idx.max().date()}, n={len(idx)})\n")
    rows = {"Buy&Hold (EW)": perf(B),
            "HAR vol-managed": perf(M_har, B),
            f"Meridian vol-managed": perf(M_mer, B)}
    hdr = f"{'strategy':>22} {'Sharpe':>7} {'annRet':>7} {'annVol':>7} {'maxDD':>7} {'alpha%':>7} {'alpha_t':>8}"
    print(hdr)
    for k, v in rows.items():
        print(f"{k:>22} {v['sharpe']:>7.2f} {v['ann_ret']*100:>6.1f}% {v['ann_vol']*100:>6.1f}% "
              f"{v['maxDD']*100:>6.1f}% {v.get('alpha_ann',float('nan'))*100:>6.1f}% "
              f"{v.get('alpha_t',float('nan')):>8.2f}")

    # PM bars
    am, ah = rows["Meridian vol-managed"], rows["HAR vol-managed"]
    print("\n--- vs the PM's bars ---")
    print(f"  Meridian Sharpe {am['sharpe']:.2f}  (want >= 1.5): {'PASS' if am['sharpe']>=1.5 else 'FAIL'}")
    print(f"  Meridian alpha t-stat {am['alpha_t']:.2f}  (want >= 1.5 sigma): {'PASS' if am['alpha_t']>=1.5 else 'FAIL'}")
    if ah['alpha_ann'] > 0:
        print(f"  Meridian alpha vs HAR alpha: {am['alpha_ann']/ah['alpha_ann']:.2f}x  (want >= 2x): "
              f"{'PASS' if am['alpha_ann']>=2*ah['alpha_ann'] else 'FAIL'}")
    print(f"  turnover/day: Meridian {turn_m:.3f}, HAR {turn_h:.3f}")


if __name__ == "__main__":
    main()
