"""Score Meridian-WM from the audited walk-forward (PREREGISTRATION.md Amendment 8, V2).

    python scripts/score_wm.py     # results/wm/bench_wm.parquet -> results/wm/wm_score.json

Same protocol as scripts/score_vol_benchmark.py: development (< 2019-01-01) and holdout
(>= 2019-01-01) on the pre-registered core universe, one common row support per table,
date-level DM (HAC), block-bootstrap CI, MCS; raw forecasts against per-asset HAR, and the same
leakage-safe walk-forward recalibration applied to every model (ref = calibrated HAR).
Secondary: the WM Student-t head's VaR coverage and the WM regime's persistence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.benchmark_wm as bw
from meridian.evalproto import qlike
from meridian.features import realized_variance
from meridian.riskscore import kupiec, christoffersen_cc
from scripts.score_vol_benchmark import calibrate, common_table, BASE_MODELS

ROOT = Path(__file__).resolve().parent.parent
WM = ROOT / "results" / "wm"
SPLIT = pd.Timestamp("2019-01-01")


def main():
    P = pd.read_parquet(WM / "bench_wm.parquet")
    P["date"] = pd.to_datetime(P["date"])
    P = P.sort_values(["asset", "date"]).reset_index(drop=True)
    seeds = sorted(c for c in P.columns if c.startswith("WM_s"))
    raw = [m for m in BASE_MODELS if m in P.columns] + ["Meridian_M0", "MeridianWM"] + seeds
    for m in ["HAR", "LightGBM", "Meridian_M0", "MeridianWM"]:
        P[f"{m}_cal"] = calibrate(P, m)
    core = P[~P["breadth"]]
    rep = {"tables": {}, "meta": json.loads((WM / "bench_wm_meta.json").read_text())}
    for pname, df in {"dev": core[core["date"] < SPLIT], "holdout": core[core["date"] >= SPLIT]}.items():
        rep["tables"][f"{pname}_raw"] = common_table(df, raw, "HAR", f"{pname}: raw forecasts")
        rep["tables"][f"{pname}_cal"] = common_table(
            df, ["HAR_cal", "LightGBM_cal", "Meridian_M0_cal", "MeridianWM_cal", "HAR", "Meridian_M0", "MeridianWM"],
            "HAR_cal", f"{pname}: walk-forward recalibrated")
    # sensitivity: date-level tests with FX rows aligned to the trading day they cover
    # (Yahoo FX bars are labelled one day early in UK summer time; audit D4)
    from meridian.data import FX, fx_true_day
    al = core[core["date"] >= SPLIT].copy()
    fxm = al["asset"].isin(list(FX))
    al.loc[fxm, "date"] = fx_true_day(al.loc[fxm, "date"])
    rep["tables"]["holdout_raw_tradeday"] = common_table(al, ["HAR", "Meridian_M0", "MeridianWM", "LightGBM", "Ridge"], "HAR",
                                                         "holdout: raw, FX aligned to trading day")
    # reproduction check: the stored M0 / HAR holdout numbers must be reproduced on the same rows
    stored = json.loads((ROOT / "results" / "benchmark_vol.json").read_text())["tables"]["holdout_raw"]
    rep["reproduction_vs_stored_benchmark"] = {m: {"stored_QLIKE": stored[m]["QLIKE"],
                                                   "rescored_QLIKE": rep["tables"]["holdout_raw"][m]["QLIKE"],
                                                   "stored_rows": stored["_meta"]["n_rows"],
                                                   "rescored_rows": rep["tables"]["holdout_raw"]["_meta"]["n_rows"]}
                                               for m in ("HAR", "Meridian_M0") if m in stored}
    # by year / asset (raw, vs per-asset HAR)
    sub = core[np.isfinite(core[["HAR", "Meridian_M0", "MeridianWM", "rv_next"]]).all(axis=1)].copy()
    for m in ("HAR", "Meridian_M0", "MeridianWM"):
        sub[f"q_{m}"] = qlike(sub["rv_next"].to_numpy(), np.exp(sub[m].to_numpy()))
    yr = sub.groupby(sub["date"].dt.year)
    rep["by_year_pct_vs_HAR"] = {m: {int(y): float(v) for y, v in (yr[f"q_{m}"].mean() / yr["q_HAR"].mean() - 1).items()}
                                 for m in ("Meridian_M0", "MeridianWM")}
    ho = sub[sub["date"] >= SPLIT].groupby("asset")
    rep["holdout_by_asset_pct_vs_HAR"] = {m: {a: float(v) for a, v in (ho[f"q_{m}"].mean() / ho["q_HAR"].mean() - 1).items()}
                                          for m in ("Meridian_M0", "MeridianWM")}
    # seed dispersion (holdout raw)
    rep["seed_pct_vs_HAR_holdout"] = {s: rep["tables"]["holdout_raw"][s]["pct_vs_ref"] for s in seeds}

    # Student-t head: next-day return VaR coverage on the holdout
    d = bw.load_truncated()
    r_next = []
    for a, o in d["prices"].items():
        from scripts.benchmark_vol import clean_ohlc
        ret = realized_variance(clean_ohlc(o))["ret"]
        r_next.append(pd.DataFrame({"asset": a, "date": ret.index, "r_next": ret.shift(-1).to_numpy()}))
    T = core.merge(pd.concat(r_next), on=["asset", "date"], how="left")
    T = T[(T["date"] >= SPLIT) & np.isfinite(T[["r_next", "WM_df", "WM_scale"]]).all(axis=1)]
    tail = {}
    for k, al in (("025", 0.025), ("010", 0.010)):
        var = T["WM_scale"].to_numpy() * stats.t.ppf(al, T["WM_df"].to_numpy())
        hits = (T["r_next"].to_numpy() <= var).astype(float)
        per = {a: (float(kupiec(h.to_numpy(), al)[0]), float(christoffersen_cc(h.to_numpy(), al)), float(h.mean()))
               for a, h in pd.Series(hits, index=T["asset"].to_numpy()).groupby(level=0)}
        tail[k] = {"violation_rate": float(hits.mean()), "n": int(len(hits)),
                   "assets_pass_kupiec": int(sum(v[0] > 0.05 for v in per.values())),
                   "assets_pass_cc": int(sum(v[1] > 0.05 for v in per.values())), "n_assets": len(per),
                   "per_asset_violation_rate": {a: v[2] for a, v in per.items()}}
    rep["wm_studentt_var_holdout"] = tail
    # regime persistence and ordering
    R = core[core["date"] >= SPLIT].dropna(subset=["WM_regime"])
    dw = []
    for _, g in R.groupby("asset"):
        s = g["WM_regime"].to_numpy()
        runs = np.diff(np.flatnonzero(np.r_[True, s[1:] != s[:-1], True]))
        dw.append(runs.mean())
    rep["wm_regime_holdout"] = {"mean_dwell_days": float(np.mean(dw)),
                                "share_by_regime": {int(k): float(v) for k, v in R["WM_regime"].value_counts(normalize=True).items()},
                                "mean_log_rv_next_by_regime": {int(k): float(v) for k, v in R.groupby("WM_regime")["y"].mean().items()}}
    (WM / "wm_score.json").write_text(json.dumps(rep, indent=2, default=float))
    for t in ("holdout_raw", "holdout_cal", "dev_raw"):
        tb = rep["tables"][t]
        print(t, tb["_meta"]["n_rows"], tb["_meta"]["n_dates"])
        for m in sorted([k for k in tb if not k.startswith("_")], key=lambda x: tb[x]["QLIKE"])[:12]:
            r = tb[m]
            print(f"   {m:18} {r['QLIKE']:.4f} {100*r['pct_vs_ref']:+6.1f}% {r['ci95_pct_vs_ref']} p={r['DM_p']}")
    print(json.dumps(rep["reproduction_vs_stored_benchmark"], indent=1))
    print(json.dumps(rep["wm_studentt_var_holdout"]["025"], default=float)[:300])
    print(rep["wm_regime_holdout"])


if __name__ == "__main__":
    main()
