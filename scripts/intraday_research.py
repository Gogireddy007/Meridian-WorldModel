"""Intraday research (new frontier, real data): use Yahoo 1-hour bars (~3yr) to
(1) build a PROPER realized-variance target from intraday returns — the real RV HAR
was designed for, which we could only proxy with daily range before; and
(2) decompose OVERNIGHT vs INTRADAY returns — a genuinely different, known-anomaly
signal that daily bars cannot isolate (Cliff-Cooper-Gulen; Lachance).

Honest scope: ~3 years is thin — this is a real prototype/validation of the
data-modality thesis, not a full study. Reported truthfully.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.backtest import perf, nw_tstat

UA = {"User-Agent": "Mozilla/5.0"}
ASSETS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "GLD", "TLT", "EFA", "EEM", "AAPL", "MSFT"]
ANN = 252


def hourly(sym):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=730d&interval=1h"
    j = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read())
    res = j["chart"]["result"][0]
    c = res["indicators"]["quote"][0]["close"]
    s = pd.Series(c, index=pd.to_datetime(np.array(res["timestamp"]), unit="s")).dropna()
    return s


def daily_from_intraday(s):
    """Per trading day: realized variance (sum sq hourly log-ret), overnight & intraday
    returns, close-to-close return."""
    lr = np.log(s).diff()
    day = s.index.normalize()
    rows = []
    for d, idx in pd.Series(s.index, index=s.index).groupby(day):
        bars = s.loc[idx]
        r = np.log(bars).diff().dropna()
        rows.append({"date": d, "rv": float((r ** 2).sum()),
                     "open": float(bars.iloc[0]), "close": float(bars.iloc[-1]),
                     "n": len(bars)})
    df = pd.DataFrame(rows).set_index("date")
    df = df[df["n"] >= 4]
    df["intraday"] = np.log(df["close"] / df["open"])           # open->close
    df["overnight"] = np.log(df["open"] / df["close"].shift(1))  # prev close->open
    df["cc"] = np.log(df["close"] / df["close"].shift(1))
    return df


def har_rv_qlike(rv):
    """Walk-forward HAR-RV on the PROPER intraday RV; return pooled QLIKE + AR1/EWMA."""
    lrv = np.log(rv.clip(lower=1e-10))
    d = pd.DataFrame({"y": lrv.shift(-1), "d": lrv,
                      "w": lrv.rolling(5).mean(), "m": lrv.rolling(22).mean()}).dropna()
    n = len(d); tr0 = max(200, n // 2)
    qh, qa, qe = [], [], []
    ewma_v = np.exp(d["d"].iloc[:tr0]).mean()
    for i in range(tr0, n):
        X = np.column_stack([np.ones(i), d[["d", "w", "m"]].iloc[:i]])
        beta, *_ = np.linalg.lstsq(X, d["y"].iloc[:i], rcond=None)
        pred = np.array([1, d["d"].iloc[i], d["w"].iloc[i], d["m"].iloc[i]]) @ beta
        resid = d["y"].iloc[:i].to_numpy() - X @ beta
        var_h = np.exp(pred + 0.5 * resid.var())
        a = np.polyfit(d["d"].iloc[:i], d["y"].iloc[:i], 1)
        var_a = np.exp(a[1] + a[0] * d["d"].iloc[i] + 0.5 * (d["y"].iloc[:i] - np.polyval(a, d["d"].iloc[:i])).var())
        rv_t = np.exp(d["y"].iloc[i])
        for q, v in [(qh, var_h), (qa, var_a)]:
            q.append(rv_t / v - np.log(rv_t / v) - 1)
    return float(np.mean(qh)), float(np.mean(qa))


def main():
    data = {}
    for a in ASSETS:
        try:
            data[a] = daily_from_intraday(hourly(a)); time.sleep(0.3)
            print(f"  {a:5} {len(data[a])} days", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {a:5} FAIL {str(e)[:40]}")
    print(f"\n=== (1) PROPER intraday realized-variance forecasting (HAR-RV) ===")
    qh = [har_rv_qlike(v["rv"])[0] for v in data.values()]
    print(f"  HAR-RV pooled QLIKE on TRUE intraday RV: {np.mean(qh):.4f} (n={len(qh)} assets, ~3yr)")

    print(f"\n=== (2) OVERNIGHT vs INTRADAY return decomposition (the anomaly) ===")
    rows = []
    for a, v in data.items():
        on, it = v["overnight"].dropna(), v["intraday"].dropna()
        rows.append({"asset": a, "overnight_ann%": on.mean() * ANN * 100, "on_sharpe": on.mean() / on.std() * np.sqrt(ANN),
                     "intraday_ann%": it.mean() * ANN * 100, "it_sharpe": it.mean() / it.std() * np.sqrt(ANN),
                     "cc_ann%": v["cc"].mean() * ANN * 100})
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    # equal-weight overnight-only vs intraday-only vs buy&hold
    on_p = pd.concat([data[a]["overnight"] for a in data], axis=1).mean(axis=1).dropna()
    it_p = pd.concat([data[a]["intraday"] for a in data], axis=1).mean(axis=1).dropna()
    cc_p = pd.concat([data[a]["cc"] for a in data], axis=1).mean(axis=1).dropna()
    print(f"\n  EW overnight-only : Sharpe {on_p.mean()/on_p.std()*np.sqrt(ANN):5.2f}  ann {on_p.mean()*ANN*100:5.1f}%  t {nw_tstat(on_p.values):.2f}")
    print(f"  EW intraday-only  : Sharpe {it_p.mean()/it_p.std()*np.sqrt(ANN):5.2f}  ann {it_p.mean()*ANN*100:5.1f}%  t {nw_tstat(it_p.values):.2f}")
    print(f"  EW buy&hold (cc)  : Sharpe {cc_p.mean()/cc_p.std()*np.sqrt(ANN):5.2f}  ann {cc_p.mean()*ANN*100:5.1f}%")
    print("\n  (Overnight anomaly: if overnight Sharpe >> intraday, a genuinely different,")
    print("   intraday-only signal exists that daily bars cannot isolate. ~3yr — prototype.)")


if __name__ == "__main__":
    main()
