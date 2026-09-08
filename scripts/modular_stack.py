"""Modular interpretable alpha stack — bridge specialist modules with a transparent
online combiner (BOA-style exponential weights). Each module is one interpretable job;
the combiner's weights are inspectable (which module the capital trusts, when).

Modules (all from daily prices we already have):
  * VOL-MANAGED  : scale exposure inverse to the Meridian vol forecast (risk module)
  * TS-MOMENTUM  : sign of trailing 12m-ex-1m return, vol-scaled (alpha module; the
                   strongest-evidence single alpha signal — Moskowitz-Ooi-Pedersen)
  * REGIME-OVERLAY: cut exposure in the WM stress regime (risk module)

Combiner: Bernstein/exponential-weights online aggregation over the module return
streams — w_k(t) ∝ exp(eta · cumret_k(t-1)), convex, reweighted daily, inspectable.

HONEST: with only 11 liquid assets + daily OHLC, breadth is thin; the research says
1.5 net Sharpe needs cross-asset breadth we lack. We measure what IS achievable and
report the combiner weights (interpretability) and drawdown (BOA's real value).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.evalproto import walk_forward_calibrate
from scripts.backtest import calibrated_var, perf, ANN, TARGET_VOL, W_MAX, COST_BPS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def ew_portfolio(per_asset: dict[str, pd.Series]) -> pd.Series:
    return pd.concat(per_asset.values(), axis=1).mean(axis=1).sort_index()


def vol_managed_stream(mv, rnext):
    df = mv.merge(rnext, on=["asset", "date"]).dropna(subset=["r_next"])
    df["sig_ann"] = np.sqrt(df["pred_var"].clip(1e-8)) * np.sqrt(ANN)
    df["w"] = (TARGET_VOL / df["sig_ann"]).clip(0, W_MAX)
    out = {}
    for a, s in df.groupby("asset"):
        s = s.sort_values("date"); w = s["w"].to_numpy(); r = s["r_next"].to_numpy()
        turn = np.abs(np.diff(w, prepend=w[0]))
        out[a] = pd.Series(w * r - COST_BPS * turn, index=s["date"])
    return ew_portfolio(out)


def tsmom_stream(frames):
    """Time-series momentum: position = sign(12m-ex-1m return), vol-scaled."""
    out = {}
    for a, f in frames.items():
        px = np.log(load_all()["prices"][a]["close"])
        ret = px.diff()
        mom = px.shift(21) - px.shift(252)                      # 12m return, skip last 1m
        sig = np.sign(mom)
        rv = ret.rolling(60).std()
        w = (sig * TARGET_VOL / (rv * np.sqrt(ANN))).clip(-W_MAX, W_MAX)
        r_next = ret.shift(-1)
        turn = w.diff().abs().fillna(0.0)
        strat = (w * r_next - COST_BPS * turn).dropna()
        out[a] = strat
    return ew_portfolio(out)


def regime_overlay_stream(mv, rnext, wm):
    df = mv.merge(rnext, on=["asset", "date"]).merge(wm, on=["asset", "date"]).dropna(subset=["r_next"])
    df["sig_ann"] = np.sqrt(df["pred_var"].clip(1e-8)) * np.sqrt(ANN)
    df["w"] = (TARGET_VOL / df["sig_ann"]).clip(0, W_MAX)
    ov = {0: 1.0, 1: 0.7, 2: 0.3}
    out = {}
    for a, s in df.groupby("asset"):
        s = s.sort_values("date").reset_index(drop=True)
        half = len(s) // 2
        rv = np.exp(s["y_true_log"].to_numpy())
        order = (pd.Series(rv[:half]).groupby(s["regime"].to_numpy()[:half]).mean().sort_values().index.tolist())
        rank = {r: i for i, r in enumerate(order)}
        fac = s["regime"].map(lambda r: ov.get(rank.get(r, 1), 1.0)).to_numpy()
        w = s["w"].to_numpy() * fac; r = s["r_next"].to_numpy()
        turn = np.abs(np.diff(w, prepend=w[0]))
        out[a] = pd.Series(w * r - COST_BPS * turn, index=s["date"])
    return ew_portfolio(out)


def boa_combine(streams: dict[str, pd.Series], eta: float = 20.0):
    """Exponential-weights online aggregation over module return streams.
    Returns (combined return series, weight DataFrame). Weights use only past info."""
    R = pd.concat(streams, axis=1).dropna()
    names = list(R.columns); K = len(names)
    cum = np.zeros(K); w = np.ones(K) / K
    combined, W = [], []
    for t in range(len(R)):
        W.append(w.copy())
        r_t = R.iloc[t].to_numpy()
        combined.append(float(w @ r_t))
        cum += r_t                                              # update AFTER trading
        w = np.exp(eta * (cum - cum.max())); w /= w.sum()       # weights for next day
    return (pd.Series(combined, index=R.index),
            pd.DataFrame(W, index=R.index, columns=names))


def main():
    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}
    rnext = pd.concat([frames[a][["r_next"]].assign(asset=a).reset_index().rename(columns={"index": "date"})
                       for a in frames], ignore_index=True)
    rnext.columns = ["date", "r_next", "asset"]; rnext["date"] = pd.to_datetime(rnext["date"])
    mer = pd.read_parquet(RESULTS / "cfjepa_ens_predictions.parquet"); mer["date"] = pd.to_datetime(mer["date"])
    mv = calibrated_var(mer, "y_pred_tgt")[["asset", "date", "pred_var"]]
    wm = pd.read_parquet(RESULTS / "wm_predictions.parquet")[["asset", "date", "regime", "y_true_log"]]
    wm["date"] = pd.to_datetime(wm["date"])

    streams = {
        "vol-managed": vol_managed_stream(mv, rnext),
        "ts-momentum": tsmom_stream(frames),
        "regime-overlay": regime_overlay_stream(mv, rnext, wm),
    }
    combined, W = boa_combine(streams)
    B = ew_portfolio({a: frames[a]["r_next"].dropna() for a in frames})

    idx = combined.index
    for s in streams.values():
        idx = idx.intersection(s.index)
    idx = idx.intersection(B.index)
    print(f"modular stack ({idx.min().date()}→{idx.max().date()}, n={len(idx)}, cost {COST_BPS*1e4:.0f}bp)\n")
    print(f"{'module':>18} {'Sharpe':>7} {'annRet':>7} {'maxDD':>7} {'alpha_t':>8}")
    for name, s in streams.items():
        v = perf(s.loc[idx], B.loc[idx]); print(f"{name:>18} {v['sharpe']:>7.2f} {v['ann_ret']*100:>6.1f}% {v['maxDD']*100:>6.1f}% {v.get('alpha_t',float('nan')):>8.2f}")
    vc = perf(combined.loc[idx], B.loc[idx])
    print(f"{'BOA combined':>18} {vc['sharpe']:>7.2f} {vc['ann_ret']*100:>6.1f}% {vc['maxDD']*100:>6.1f}% {vc.get('alpha_t',float('nan')):>8.2f}")
    print(f"\n  mean BOA weights (interpretable): " +
          ", ".join(f"{c} {W.loc[idx, c].mean():.2f}" for c in W.columns))
    print(f"  Sharpe >= 1.5? {'PASS' if vc['sharpe']>=1.5 else 'FAIL'}  | alpha_t >= 1.5? "
          f"{'PASS' if vc.get('alpha_t',0)>=1.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
