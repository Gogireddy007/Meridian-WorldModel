"""Frontier strategies on newly-acquired data — the genuinely-unexplored, potentially
higher-Sharpe avenues, tested HONESTLY (realistic costs, deflated Sharpe, crash risk).

  A. CRYPTO trend — less-efficient 24/7 market; TSMOM across 15 coins, 10bp cost.
     Caveat: my coins are SURVIVORS (survivorship bias inflates this); crypto slippage
     is real; the high-Sharpe era was largely 2017-2021.
  B. VRP short-vol — harvest the variance risk premium via SVXY (short VIX futures),
     with a VIX term-structure (contango) crash filter. Caveat: short-vol has
     CATASTROPHIC tails (2018 volmageddon); the filter is the whole ballgame.
  C. Combined (crypto-trend + VRP + traditional futures-trend) — decorrelated sleeves.

Reports net Sharpe + deflated Sharpe + worst drawdown. No fabrication.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import fetch_yahoo
from scripts.backtest import nw_tstat

ANN = 252
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
          "LTC-USD", "LINK-USD", "DOT-USD", "AVAX-USD", "XLM-USD", "BCH-USD", "ETC-USD", "ATOM-USD"]


def get(t, start="2014-01-01"):
    return fetch_yahoo(t, start=start)["adjclose"]


def dsr(sr, n, trials=18):
    e = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / trials) + np.euler_gamma * stats.norm.ppf(1 - 1.0 / (trials * np.e))
    return float(stats.norm.cdf((sr / np.sqrt(ANN) - e / np.sqrt(n)) * np.sqrt(n - 1)))


def report(name, r, ann=ANN):
    r = r.dropna()
    sh = r.mean() / r.std() * np.sqrt(ann)
    curve = np.cumprod(1 + r.values); dd = (curve / np.maximum.accumulate(curve) - 1).min()
    print(f"  {name:32} Sharpe {sh:5.2f} | ann {r.mean()*ann*100:6.1f}% | maxDD {dd*100:6.1f}% "
          f"| t {nw_tstat(r.values):5.2f} | defl-p {dsr(sh, len(r)):.2f}")
    return r, sh


def crypto_trend():
    px = pd.DataFrame({t: np.log(get(t)) for t in CRYPTO}).sort_index()
    ret = px.diff()
    vol = ret.rolling(30).std()
    sig = sum(np.tanh((px - px.shift(lb)) / (vol * np.sqrt(lb) + 1e-9)) for lb in (14, 30, 90)) / 3
    pos = sig * (0.30 / np.sqrt(365)) / vol.shift(1).clip(lower=1e-3)   # crypto uses 365d
    gross = (pos.shift(1) * ret).sum(axis=1)
    turn = (pos - pos.shift(1)).abs().sum(axis=1)
    r = gross - 10e-4 * turn                                            # 10bp crypto cost
    rv = r.rolling(30).std().shift(1) * np.sqrt(365)
    return (r * (0.20 / rv).clip(0, 3).bfill().fillna(1.0)).iloc[90:]


def vrp_shortvol():
    svxy = np.log(get("SVXY", "2014-01-01")).diff()                    # short-VIX-futures ETF
    vix = get("^VIX", "2014-01-01"); vix3m = get("^VIX3M", "2014-01-01")
    idx = svxy.index.intersection(vix.index).intersection(vix3m.index)
    svxy, vix, vix3m = svxy.loc[idx], vix.loc[idx], vix3m.loc[idx]
    contango = (vix3m > vix).astype(float)                            # curve in contango → short vol OK
    # crash filter: also require VIX not spiking (below its 20d mean rising)
    calm = (vix < vix.rolling(20).mean() * 1.15).astype(float)
    signal = (contango * calm).shift(1).fillna(0.0)                    # causal
    naive = svxy                                                       # always short vol
    filtered = signal * svxy - 2e-4 * signal.diff().abs().fillna(0)    # 2bp on switches
    return naive.loc[idx], filtered.loc[idx]


def main():
    print("=== A. CRYPTO trend (survivor coins, 10bp cost — survivorship-biased UP) ===")
    ct, ct_sh = report("crypto TSMOM (2015-26)", crypto_trend())
    ct_recent = ct[ct.index >= "2022-01-01"]
    report("crypto TSMOM (2022-26 only)", ct_recent)

    print("\n=== B. VRP short-vol (SVXY) — the premium AND the crash ===")
    naive, filt = vrp_shortvol()
    report("naive short-vol (no filter)", naive)
    fr, fr_sh = report("short-vol + contango/crash filter", filt)

    print("\n=== C. Decorrelated combination ===")
    from scripts.futures_strategy import load_futures
    fut = load_futures()
    fpx = pd.DataFrame({t: np.log(v["adjclose"]) for t, v in fut.items()}).sort_index().ffill()
    fret = fpx.diff(); fvol = fret.rolling(60).std()
    fsig = sum(np.tanh((fpx - fpx.shift(lb)) / (fvol * np.sqrt(lb) + 1e-9)) for lb in (21, 63, 126, 252)) / 4
    fpos = fsig * (0.15 / np.sqrt(ANN)) / fvol.shift(1).clip(lower=1e-4)
    ftrend = ((fpos.shift(1) * fret).sum(axis=1)).rename("fut_trend")
    S = pd.DataFrame({"crypto": ct, "vrp": fr, "fut_trend": ftrend}).dropna()
    print(f"  sleeve correlations:\n{S.corr().round(2).to_string()}")
    comb = (S / S.std()).mean(axis=1); comb = comb / comb.std() * (0.12 / np.sqrt(ANN))
    report("EQUAL-RISK combined (crypto+VRP+trend)", comb)
    print(f"\n  vs 1.5 bar: combined Sharpe {comb.mean()/comb.std()*np.sqrt(ANN):.2f}")


if __name__ == "__main__":
    main()
