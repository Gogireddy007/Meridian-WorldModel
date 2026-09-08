"""Forecast tracking — the honest live track record. Logs every forecast the model
makes (append-only, deduped) and, once a forecast date has realized, scores it against
the ACTUAL realized volatility. Over time this builds a genuine out-of-sample record:
what the model said BEFORE the fact vs what happened. No lookahead — scoring uses only
data that has since become available.

Run daily: it logs new forecasts and scores any past ones that have now realized.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import realized_variance

RESULTS = Path(__file__).resolve().parent.parent / "results"
LOG = RESULTS / "forecast_log.csv"


def log_current():
    fc = json.loads((RESULTS / "forecast_week.json").read_text())
    made_on = fc["last_close_date"]
    rows = []
    for a, v in fc["assets"].items():
        for w in v["week"]:
            rows.append({"made_on": made_on, "asset": a, "forecast_date": w["date"],
                         "vol_forecast_ann_pct": w["vol_forecast_ann_pct"], "realized_vol_ann_pct": np.nan})
    new = pd.DataFrame(rows)
    if LOG.exists():
        old = pd.read_csv(LOG)
        combined = pd.concat([old, new]).drop_duplicates(subset=["made_on", "asset", "forecast_date"], keep="first")
    else:
        combined = new
    return combined


def score(log):
    """Fill realized_vol for forecast_dates that have now realized, and report accuracy."""
    d = load_all()
    realized = {}
    for a, ohlc in d["prices"].items():
        rv = realized_variance(ohlc)["rv"].dropna()
        realized[a] = (np.sqrt(rv * 252) * 100)              # annualized vol %
    for i, row in log.iterrows():
        if not np.isnan(row["realized_vol_ann_pct"]):
            continue
        a = row["asset"]; fd = pd.Timestamp(row["forecast_date"])
        if a in realized and fd in realized[a].index:
            log.at[i, "realized_vol_ann_pct"] = float(realized[a].loc[fd])
    return log


def main():
    log = log_current()
    log = score(log)
    log.to_csv(LOG, index=False)
    scored = log.dropna(subset=["realized_vol_ann_pct"])
    print(f"forecast log: {len(log)} entries, {len(scored)} now scored (realized)\n")
    if len(scored) >= 5:
        e = scored["vol_forecast_ann_pct"] - scored["realized_vol_ann_pct"]
        mae = e.abs().mean()
        corr = scored["vol_forecast_ann_pct"].corr(scored["realized_vol_ann_pct"])
        bias = e.mean()
        print(f"  LIVE track record (realized forecasts):")
        print(f"    MAE  {mae:.2f} vol-pts | bias {bias:+.2f} | forecast-vs-realized corr {corr:.3f} | n={len(scored)}")
    else:
        print("  Not enough realized forecasts to score yet — the record starts building as days pass.")
        print(f"  Logged {len(log)} forecasts (through {log['forecast_date'].max()}) for future scoring.")
    print(f"\n  saved -> results/forecast_log.csv  (append-only live track record)")


if __name__ == "__main__":
    main()
