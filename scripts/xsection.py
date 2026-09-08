"""Cross-sectional equity module — the breadth lever the research flags as the
strongest daily-data alpha source (Gu-Kelly-Xiu 2020). ~100 liquid large-caps,
total-return prices, dollar-neutral long/short on cross-sectionally-ranked factors,
cost-aware, walk-forward (rules → inherently OOS). Honest result, whatever it is.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import fetch_yahoo, DATA_DIR
from scripts.backtest import perf, nw_tstat

ANN = 252
COST_BPS = 5.0 / 1e4          # per-trade, single-name equities (realistic ~5bp)

TICKERS = ("AAPL MSFT AMZN GOOGL META NVDA TSLA JPM V MA JNJ WMT PG HD BAC XOM CVX "
           "ABBV PFE KO PEP COST MRK TMO AVGO CSCO ADBE CRM ACN NKE MCD DHR TXN NEE "
           "LIN PM UNP LOW HON INTC IBM GS CAT AMD SBUX BLK AXP GE BA MMM CVS AMGN "
           "GILD LMT MDT ISRG NOW INTU ADP BKNG MDLZ CB SO DUK PLD AMT SPG T VZ CMCSA "
           "DIS NFLX ORCL QCOM MU AMAT F GM DAL UAL WFC C USB PNC SCHW COF MET AIG "
           "TGT DG DLTR KR SYY ADM MO CL KMB GIS K HSY").split()

SDIR = DATA_DIR / "stocks"
SDIR.mkdir(exist_ok=True)


def load_stocks(refresh=False):
    out = {}
    for t in TICKERS:
        f = SDIR / f"{t}.parquet"
        if f.exists() and not refresh:
            out[t] = pd.read_parquet(f); continue
        try:
            df = fetch_yahoo(t, start="2008-01-01")
            if len(df) > 1500:
                df.to_parquet(f); out[t] = df
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.25)
    return out


def zscore_xs(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)


def ls_portfolio(signal, ret, vol):
    """Dollar-neutral long/short on cross-sectional signal, inverse-vol within legs,
    vol-targeted; net of turnover cost."""
    s = signal.copy()
    r = s.rank(axis=1, pct=True)
    w = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    w[r >= 0.8] = 1.0; w[r <= 0.2] = -1.0
    w = w / (vol + 1e-6)                                    # inverse-vol sizing
    # dollar-neutral + unit gross
    w = w.sub(w.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).clip(lower=1e-6), axis=0)
    wl = w.shift(1).fillna(0.0)
    gross = (wl * ret).sum(axis=1)
    turn = (w - wl).abs().sum(axis=1).fillna(0.0)
    net = gross - COST_BPS * turn
    rv = net.rolling(60).std().shift(1) * np.sqrt(ANN)
    scale = (0.10 / rv).clip(0, 3).bfill().fillna(1.0)
    return (net * scale).iloc[60:]


def main():
    d = load_stocks()
    px = pd.DataFrame({t: np.log(v["adjclose"]) for t, v in d.items()}).sort_index().ffill()
    ret = px.diff()
    print(f"cross-section: {px.shape[1]} stocks, {px.index.min().date()}→{px.index.max().date()}")
    vol = ret.rolling(60).std()
    factors = {
        "mom12": px.shift(21) - px.shift(252),            # 12-1m momentum
        "rev1": -(px - px.shift(21)),                      # 1m short-term reversal
        "lowvol": -vol,                                    # low-volatility
        "rev5": -(px - px.shift(5)),                       # 1w reversal
    }
    spy = ret.mean(axis=1)                                 # EW market proxy
    print(f"\n{'factor':>10} {'Sharpe':>7} {'annRet':>7} {'annVol':>7} {'maxDD':>7} {'t-stat':>7}")
    streams = {}
    for name, f in factors.items():
        sig = zscore_xs(f)
        strat = ls_portfolio(sig, ret, vol)
        streams[name] = strat
        v = perf(strat)
        print(f"{name:>10} {v['sharpe']:>7.2f} {v['ann_ret']*100:>6.1f}% {v['ann_vol']*100:>6.1f}% {v['maxDD']*100:>6.1f}% {nw_tstat(strat.values):>7.2f}")
    # combined multi-factor (equal-risk)
    S = pd.DataFrame(streams).dropna()
    comb = (S / S.std()).mean(axis=1); comb = comb / comb.std() * (0.10/np.sqrt(ANN))
    vc = perf(comb)
    print(f"{'COMBINED':>10} {vc['sharpe']:>7.2f} {'':>7} {'':>7} {vc['maxDD']*100:>6.1f}% {nw_tstat(comb.values):>7.2f}")
    print(f"\n  vs PM bar: combined Sharpe {vc['sharpe']:.2f} (>=1.5 {'PASS' if vc['sharpe']>=1.5 else 'FAIL'}), "
          f"t {nw_tstat(comb.values):.2f}")
    print(f"  factor correlations:\n{S.corr().round(2).to_string()}")


if __name__ == "__main__":
    main()
