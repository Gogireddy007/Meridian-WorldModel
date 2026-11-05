"""Build the data snapshot for the frozen-model deployment (scripts/deploy_frozen.py).

    python scripts/deploy_data.py            # fetch every source, write results/deploy/data/

The audited benchmark read Yahoo daily OHLC, so Yahoo is tried first; when it rate-limits
(HTTP 429, seen repeatedly from cloud hosts) a fallback source is used. Each source is CHECKED against the training-era truth stored in
results/benchmark_vol_predictions.parquet (column rv_next = the next-day Garman-Klass
variance the audited benchmark scored against, built from Yahoo bars). The check result is
written to the manifest; scripts/audit_deploy.py fails the run if a source does not match.

  equities / ETFs  Yahoo chart API (as in training); fallback Nasdaq historical API (~10 years)
  FX               Yahoo chart API (as in training); fallback Dukascopy daily BID candles
  macro            FRED CSV (VIXCLS, DGS10, DGS2)

Every raw file is stored with its sha256 so the snapshot cannot change silently.
"""
from __future__ import annotations

import hashlib
import io
import json
import lzma
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import EQUITIES, FX, BREADTH, FRED_SERIES, fetch_yahoo
from meridian.features import realized_variance

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "deploy" / "data"
RAW = ROOT / "results" / "deploy" / "raw"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
NASDAQ_CLASS = {"SPY": "etf", "QQQ": "etf", "IWM": "etf", "TLT": "etf", "GLD": "etf"}
DUKA_SCALE = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3}
FX_FIRST_YEAR = 2016


def curl(url, ua=UA, timeout=60, tries=6):
    last = ""
    for k in range(tries):
        r = subprocess.run(["curl", "-sS", "-m", str(timeout), "-A", ua, "-w", "\n%{http_code}", url],
                           capture_output=True, timeout=timeout + 10)
        body, _, code = r.stdout.rpartition(b"\n")
        if code == b"200" and body:
            return body
        last = code.decode() or r.stderr.decode()[:200]
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"GET failed ({last}): {url}")


def save_raw(name, blob):
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / name).write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def nasdaq(sym):
    cls = NASDAQ_CLASS.get(sym, "stocks")
    url = (f"https://api.nasdaq.com/api/quote/{sym}/historical?assetclass={cls}"
           f"&fromdate=2006-01-01&todate={datetime.now().date()}&limit=99999")
    blob = curl(url)
    h = save_raw(f"nasdaq_{sym}.json", blob)
    rows = json.loads(blob)["data"]["tradesTable"]["rows"]
    num = lambda x: float(str(x).replace("$", "").replace(",", "")) if x not in (None, "", "N/A") else np.nan
    df = pd.DataFrame({"open": [num(r["open"]) for r in rows], "high": [num(r["high"]) for r in rows],
                       "low": [num(r["low"]) for r in rows], "close": [num(r["close"]) for r in rows],
                       "volume": [num(r["volume"]) for r in rows]},
                      index=pd.to_datetime([r["date"] for r in rows], format="%m/%d/%Y"))
    df.index.name = "date"
    df = df.sort_index().dropna(subset=["open", "high", "low", "close"])
    df["adjclose"] = df["close"]
    return df, {"source": "nasdaq", "url": url, "sha256": {f"nasdaq_{sym}.json": h}}


def duka_file(pair, year, month=None):
    path = f"{pair}/{year}/" + (f"{month:02d}/" if month is not None else "") + "BID_candles_day_1.bi5"
    blob = curl(f"https://datafeed.dukascopy.com/datafeed/{path}", timeout=40, tries=8)
    raw = lzma.decompress(blob)                       # raises if the body is not a bi5 file
    base = pd.Timestamp(year=year, month=(month or 0) + 1, day=1)
    recs = [struct.unpack(">5if", raw[i:i + 24]) for i in range(0, len(raw), 24)]
    s = DUKA_SCALE[pair]
    df = pd.DataFrame([{"date": base + pd.Timedelta(seconds=r[0]), "open": r[1] / s, "close": r[2] / s,
                        "low": r[3] / s, "high": r[4] / s, "volume": r[5]} for r in recs])
    name = f"duka_{pair}_{year}" + (f"_{month:02d}" if month is not None else "") + ".bi5"
    return df, name, save_raw(name, blob)


def dukascopy(pair):
    now = datetime.now(timezone.utc)
    parts, hashes = [], {}
    for y in range(FX_FIRST_YEAR, now.year):          # closed years: one yearly file
        df, n, h = duka_file(pair, y); parts.append(df); hashes[n] = h
        time.sleep(1.5)
    for m in range(0, now.month):                     # current year: monthly files (0-based)
        try:
            df, n, h = duka_file(pair, now.year, m); parts.append(df); hashes[n] = h
        except Exception as e:                        # the running month may not exist yet
            print(f"    {pair} {now.year}/{m:02d}: {e}", flush=True)
        time.sleep(1.5)
    df = pd.concat(parts).drop_duplicates("date", keep="last").set_index("date").sort_index()
    df = df[df.index.dayofweek < 5]                   # Dukascopy has thin Sunday-evening candles
    df["adjclose"] = df["close"]
    return df, {"source": "dukascopy_bid_gmt_day", "sha256": hashes}


def equity(sym):
    try:
        return fetch_yahoo(sym), {"source": "yahoo"}
    except Exception as e:
        print(f"    yahoo {sym} unavailable ({str(e)[:80]}); using Nasdaq", flush=True)
        return nasdaq(sym)


def fx(name, sym):
    try:
        df = fetch_yahoo(sym)
        return df, {"source": "yahoo"}
    except Exception as e:
        print(f"    yahoo {sym} unavailable ({str(e)[:80]}); using Dukascopy", flush=True)
        return dukascopy(name)


def fred():
    cols, hashes = {}, {}
    for s in FRED_SERIES:
        blob = curl(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", ua="curl/8.7.1", timeout=90)
        hashes[f"fred_{s}.csv"] = save_raw(f"fred_{s}.csv", blob)
        df = pd.read_csv(io.BytesIO(blob))
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        cols[s] = pd.to_numeric(df.set_index(df.columns[0])[s], errors="coerce").dropna()
    return pd.DataFrame(cols).sort_index(), {"source": "fred", "sha256": hashes}


def complete_bars(df):
    """Drop a bar of the New York trading day still in progress (before 17:00 ET)."""
    now = pd.Timestamp.now(tz="America/New_York")
    last = now.normalize().tz_localize(None) - (pd.Timedelta(0) if now.hour >= 17 else pd.Timedelta(days=1))
    return df[df.index <= last]


def check_against_benchmark(name, ohlc):
    """Compare next-day GK variance from this source with the audited benchmark's rv_next
    (built from Yahoo) on every overlapping origin date."""
    P = pd.read_parquet(ROOT / "results" / "benchmark_vol_predictions.parquet", columns=["asset", "date", "rv_next"])
    p = P[P["asset"] == name].set_index("date")["rv_next"]
    from scripts.benchmark_vol import clean_ohlc
    rv = realized_variance(clean_ohlc(ohlc))["rv"]
    nxt = rv.shift(-1).reindex(p.index)
    m = np.isfinite(nxt) & np.isfinite(p) & (p > 1e-11)
    if m.sum() == 0:
        return {"overlap": 0}
    d = np.log(nxt[m]) - np.log(p[m])
    return {"overlap": int(m.sum()), "first": str(p.index[m].min().date()), "last": str(p.index[m].max().date()),
            "median_abs_log_diff": float(np.median(np.abs(d))), "share_exact_1e-3": float((np.abs(d) < 1e-3).mean()),
            "corr_log": float(np.corrcoef(np.log(nxt[m]), np.log(p[m]))[0, 1])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "assets": {}}
    for name in list(EQUITIES) + list(BREADTH):
        print(f"  {name}", flush=True)
        df, info = equity(name)
        df = complete_bars(df)
        df.to_parquet(OUT / f"px_{name}.parquet")
        info.update(rows=len(df), first=str(df.index.min().date()), last=str(df.index.max().date()),
                    check_vs_benchmark=check_against_benchmark(name, df))
        manifest["assets"][name] = info
    for name, sym in FX.items():
        print(f"  {name} (fx)", flush=True)
        df, info = fx(name, sym)
        df = complete_bars(df)
        df.to_parquet(OUT / f"px_{name}.parquet")
        info.update(rows=len(df), first=str(df.index.min().date()), last=str(df.index.max().date()),
                    check_vs_benchmark=check_against_benchmark(name, df))
        manifest["assets"][name] = info
    macro, info = fred()
    macro.to_parquet(OUT / "macro.parquet")
    info.update(last={c: str(macro[c].dropna().index.max().date()) for c in macro})
    manifest["macro"] = info
    manifest["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.parquet"))}
    (OUT.parent / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    for a, v in manifest["assets"].items():
        c = v["check_vs_benchmark"]
        print(f"  {a:7} {v['source']:22} {v['first']}..{v['last']}  overlap={c.get('overlap')} "
              f"med|dlog|={c.get('median_abs_log_diff', float('nan')):.4f} corr={c.get('corr_log', float('nan')):.4f}")


if __name__ == "__main__":
    main()
