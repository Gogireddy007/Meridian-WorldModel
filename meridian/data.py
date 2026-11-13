"""Data loaders for Meridian.

Free/delayed daily sources only (per project scope):
  - Yahoo chart API  -> daily OHLCV for equities, ETFs, FX
  - FRED CSV         -> VIX, treasury yields (macro context features)

Everything is cached to data/*.parquet so reruns are offline and deterministic.
No API keys required.
"""
from __future__ import annotations

import io
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Pre-registered universe (see PREREGISTRATION.md) — do not extend this in place;
# the win-criteria comparisons in PREREGISTRATION.md are scoped to exactly this set.
EQUITIES = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "JPM", "XOM", "JNJ"]
FX = {"EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X"}
FRED_SERIES = ["VIXCLS", "DGS10", "DGS2"]

# Breadth extension (see PREREGISTRATION.md Amendment 4) — additive, non-pre-registered
# assets used only for the exploratory breadth cut in scripts/benchmark_vol.py, never
# mixed into the locked EQUITIES/FX win-criteria universe above.
BREADTH = {"TLT": "TLT", "GLD": "GLD"}   # rates (20y+ Treasury ETF), commodities (gold)

DEFAULT_START = "2007-01-01"


def _get(url: str, tries: int = 3, timeout: int = 30) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    # Fallback: some hosts (e.g. FRED) block urllib but serve curl fine.
    try:
        out = subprocess.run(
            ["curl", "-sS", "--http1.1", "--retry", "3", "-m", str(timeout),
             "-A", _UA["User-Agent"], url],
            capture_output=True, timeout=timeout + 5,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout
        last = RuntimeError(out.stderr.decode()[:200] or "curl empty")
    except Exception as e:  # noqa: BLE001
        last = e
    raise RuntimeError(f"failed GET {url}: {last}")


def fetch_yahoo(symbol: str, start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Daily OHLC(V) from the Yahoo chart API. Returns tz-naive DatetimeIndex."""
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp()) if end else int(pd.Timestamp.utcnow().timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    j = json.loads(_get(url))
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    cols = {
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "volume": q.get("volume"),
    }
    # total-return (dividend+split adjusted) close — REQUIRED for strategy P&L
    adj = res.get("indicators", {}).get("adjclose")
    cols["adjclose"] = adj[0].get("adjclose") if adj else q.get("close")
    df = pd.DataFrame(cols, index=pd.to_datetime(np.array(ts), unit="s").normalize())
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["adjclose"] = df["adjclose"].ffill().fillna(df["close"])
    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_fred(series: str, start: str = DEFAULT_START) -> pd.Series:
    # Prefer a local raw CSV if present (FRED is flaky to urllib/subprocess).
    local = DATA_DIR / f"fred_{series}.csv"
    if local.exists():
        raw = local.read_text()
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        raw = _get(url).decode()
        local.write_text(raw)
    df = pd.read_csv(io.StringIO(raw))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    s = pd.to_numeric(df[series], errors="coerce")
    s = s[s.index >= pd.Timestamp(start)]
    s.name = series
    return s.dropna()


def load_all(refresh: bool = False, include_breadth: bool = False) -> dict[str, pd.DataFrame]:
    """Load the full pre-registered universe, cached to parquet.

    include_breadth=True additionally loads the non-pre-registered breadth
    assets (BREADTH) — used only for the exploratory breadth cut in
    scripts/benchmark_vol.py (see PREREGISTRATION.md Amendment 4); the
    default (False) is byte-for-byte the original pre-registered universe.

    Returns {'prices': {sym: ohlc df}, 'macro': df}.
    """
    cache = DATA_DIR / "universe.parquet"
    macro_cache = DATA_DIR / "macro.parquet"
    prices: dict[str, pd.DataFrame] = {}

    symbols = {s: s for s in EQUITIES} | FX
    if include_breadth:
        symbols = symbols | BREADTH
    for name, sym in symbols.items():
        f = DATA_DIR / f"px_{name}.parquet"
        if f.exists() and not refresh:
            prices[name] = pd.read_parquet(f)
        else:
            print(f"  fetching {name} ({sym}) ...", flush=True)
            df = fetch_yahoo(sym)
            df.to_parquet(f)
            prices[name] = df

    if macro_cache.exists() and not refresh:
        macro = pd.read_parquet(macro_cache)
    else:
        cols = {}
        for s in FRED_SERIES:
            print(f"  fetching FRED {s} ...", flush=True)
            cols[s] = fetch_fred(s)
        macro = pd.DataFrame(cols).sort_index()
        macro.to_parquet(macro_cache)

    return {"prices": prices, "macro": macro}


if __name__ == "__main__":
    d = load_all()
    for k, v in d["prices"].items():
        print(f"{k:8} {v.shape[0]:5d} rows  {v.index.min().date()} -> {v.index.max().date()}")
    print("macro:", d["macro"].shape, d["macro"].columns.tolist(),
          d["macro"].index.min().date(), "->", d["macro"].index.max().date())


# --------------------------------------------------------------------------- #
# FX bar dating (found by the deployment audit, check D4)
# --------------------------------------------------------------------------- #
# Yahoo daily FX bars cover one London trading day and are stamped at London midnight.
# fetch_yahoo() normalizes the UTC timestamp, so during UK summer time (BST, UTC+1) the
# stamp 23:00 UTC falls on the PREVIOUS calendar day: a bar labelled d is the session of
# d+1 (labels run Sun-Thu in summer, Mon-Fri in winter). Every FX row in this project
# since 2007 carries that label. Within one asset the ordering is unaffected and nothing
# from the future enters a forecast; the label only matters when FX rows are aligned with
# other assets by date or when deciding whether the latest bar is complete.
def fx_true_day(labels) -> pd.DatetimeIndex:
    """Map Yahoo FX bar labels to the London trading day they actually cover."""
    labels = pd.DatetimeIndex(labels)
    nxt = labels + pd.Timedelta(days=1)
    london_midnight_utc = (nxt.tz_localize("Europe/London", nonexistent="shift_forward", ambiguous=False)
                           .tz_convert("UTC").normalize().tz_localize(None))
    shifted = london_midnight_utc == labels           # BST: the session starts 23:00 UTC on the label day
    return pd.DatetimeIndex(np.where(shifted, nxt, labels))


def fx_bar_end_utc(labels) -> pd.DatetimeIndex:
    """UTC time at which the Yahoo FX bar with each label is complete (next London midnight)."""
    day = fx_true_day(labels) + pd.Timedelta(days=1)
    return (day.tz_localize("Europe/London", nonexistent="shift_forward", ambiguous=False)
            .tz_convert("UTC").tz_localize(None))
