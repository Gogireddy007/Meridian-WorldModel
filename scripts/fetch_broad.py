"""Fetch a BROAD multi-asset-class universe (real daily data, Yahoo) — the breadth
the research says is the only evidence-backed path toward Sharpe ~1.5.

~40 liquid instruments across 5 asset classes with long history. Cached to data/broad/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import fetch_yahoo, DATA_DIR

BROAD = {
    # US equity beta
    "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "DIA": "DIA", "MDY": "MDY",
    # US sectors (cross-sectional breadth)
    "XLF": "XLF", "XLK": "XLK", "XLE": "XLE", "XLV": "XLV", "XLI": "XLI",
    "XLY": "XLY", "XLP": "XLP", "XLU": "XLU", "XLB": "XLB",
    # international equity
    "EFA": "EFA", "EEM": "EEM", "EWJ": "EWJ", "EWG": "EWG", "EWU": "EWU", "EWZ": "EWZ",
    # rates / bonds
    "TLT": "TLT", "IEF": "IEF", "SHY": "SHY", "LQD": "LQD", "HYG": "HYG",
    "TIP": "TIP", "AGG": "AGG",
    # commodities
    "GLD": "GLD", "SLV": "SLV", "USO": "USO", "DBC": "DBC", "DBA": "DBA",
    # FX (vs USD)
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X",
}

ASSET_CLASS = {}
for k in ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLF", "XLK", "XLE", "XLV", "XLI",
          "XLY", "XLP", "XLU", "XLB", "EFA", "EEM", "EWJ", "EWG", "EWU", "EWZ"]:
    ASSET_CLASS[k] = "equity"
for k in ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG"]:
    ASSET_CLASS[k] = "bond"
for k in ["GLD", "SLV", "USO", "DBC", "DBA"]:
    ASSET_CLASS[k] = "commodity"
for k in ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]:
    ASSET_CLASS[k] = "fx"

BDIR = DATA_DIR / "broad"
BDIR.mkdir(exist_ok=True)


def load_broad(refresh=False):
    out = {}
    for name, sym in BROAD.items():
        f = BDIR / f"{name}.parquet"
        if f.exists() and not refresh:
            out[name] = pd.read_parquet(f)
            continue
        try:
            df = fetch_yahoo(sym, start="2005-01-01")
            if len(df) > 500:
                df.to_parquet(f)
                out[name] = df
                print(f"  {name:8} {len(df):5d} rows {df.index.min().date()}→{df.index.max().date()}", flush=True)
            else:
                print(f"  {name:8} SKIP (short history {len(df)})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:8} FAIL {str(e)[:50]}", flush=True)
        time.sleep(0.3)
    return out


if __name__ == "__main__":
    d = load_broad()
    print(f"\nloaded {len(d)} instruments across "
          f"{len(set(ASSET_CLASS[k] for k in d))} asset classes")
