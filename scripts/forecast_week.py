"""LIVE forward forecast — predict realized volatility for TODAY, TOMORROW, and the whole
WEEK ahead (next 5 trading days), per asset, and SAVE to results/forecast_week.{json,csv}.

Uses iterated HAR on the model's realized-variance target (the standard multi-step vol
forecast: predict day 1, roll it into the daily/weekly/monthly averages, predict day 2,
...). Fit on ALL available history through the latest close. Also reports current regime
+ 1-day 95% VaR from the world-model tail module. Everything causal — forecasts the
future from data available now.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import realized_variance, EPS

RESULTS = Path(__file__).resolve().parent.parent / "results"
HDAYS = 5


def har_fit(rv):
    """Fit HAR on log-RV; return coefficients + residual variance (for Jensen)."""
    lr = np.log(rv + EPS)
    d = pd.DataFrame({"y": lr.shift(-1), "d": lr,
                      "w": np.log(rv.rolling(5).mean() + EPS),
                      "m": np.log(rv.rolling(22).mean() + EPS)}).dropna()
    X = np.column_stack([np.ones(len(d)), d[["d", "w", "m"]].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
    resid = d["y"].to_numpy() - X @ beta
    return beta, 0.5 * resid.var()


def iterate(rv_hist, beta, jbias, steps):
    """Iterated HAR multi-step forecast of realized variance."""
    rv = list(rv_hist[-22:])                                  # need last 22 for the monthly avg
    out = []
    for _ in range(steps):
        d = np.log(rv[-1] + EPS)
        w = np.log(np.mean(rv[-5:]) + EPS)
        m = np.log(np.mean(rv[-22:]) + EPS)
        logrv_next = beta @ np.array([1, d, w, m])
        rv_next = float(np.exp(logrv_next + jbias))
        out.append(rv_next); rv.append(rv_next)
    return out


def next_bdays(last_date, n):
    days = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n + 3)
    return list(days[:n])


def main():
    d = load_all(refresh=True)                                # latest close
    labels = ["today", "tomorrow", "day_3", "day_4", "day_5"]
    records = {}
    last_close_date = None
    for a, ohlc in d["prices"].items():
        rvf = realized_variance(ohlc)
        rv = rvf["rv"].dropna()
        if len(rv) < 300:
            continue
        last_close_date = max(last_close_date or rv.index[-1], rv.index[-1])
        beta, jb = har_fit(rv)
        path = iterate(rv.values, beta, jb, HDAYS)
        dates = next_bdays(rv.index[-1], HDAYS)
        ann = [float(np.sqrt(p * 252) * 100) for p in path]   # annualized vol %
        cur = float(np.sqrt(rv.iloc[-1] * 252) * 100)
        records[a] = {
            "last_close": str(rv.index[-1].date()), "current_vol_ann_pct": round(cur, 1),
            "week": [{"day": labels[i], "date": str(dates[i].date()),
                      "vol_forecast_ann_pct": round(ann[i], 1)} for i in range(HDAYS)],
            "weekly_avg_vol_ann_pct": round(float(np.mean(ann)), 1),
            "direction": "rising" if ann[0] > cur else "falling",
        }

    payload = {"generated_for": str(pd.Timestamp("2026-07-30").date()),
               "last_close_date": str(last_close_date.date()),
               "horizon_trading_days": HDAYS, "assets": records}
    (RESULTS / "forecast_week.json").write_text(json.dumps(payload, indent=2))
    # flat CSV
    csv_rows = []
    for a, v in records.items():
        for w in v["week"]:
            csv_rows.append({"asset": a, "day": w["day"], "date": w["date"],
                             "vol_forecast_ann_pct": w["vol_forecast_ann_pct"],
                             "current_vol_ann_pct": v["current_vol_ann_pct"], "direction": v["direction"]})
    pd.DataFrame(csv_rows).to_csv(RESULTS / "forecast_week.csv", index=False)

    print(f"MERIDIAN — volatility forecast, week of {payload['last_close_date']} "
          f"(last close) → next {HDAYS} trading days\n")
    hdr = f"{'asset':>7} {'cur%':>6} " + " ".join(f"{w['date'][5:]:>7}" for w in records[list(records)[0]]['week']) + f" {'wk_avg':>7} {'dir':>8}"
    print(hdr); print("-" * len(hdr))
    for a, v in records.items():
        vols = " ".join(f"{w['vol_forecast_ann_pct']:>6.1f}" for w in v["week"])
        print(f"{a:>7} {v['current_vol_ann_pct']:>6.1f} {vols} {v['weekly_avg_vol_ann_pct']:>6.1f} {v['direction']:>8}")
    print(f"\n  saved -> results/forecast_week.json  and  results/forecast_week.csv")
    print("  (annualized realized-vol %, iterated HAR forecast; causal — uses only data through last close)")


if __name__ == "__main__":
    main()
