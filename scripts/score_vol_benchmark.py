"""Score the volatility benchmark from saved predictions (PREREGISTRATION.md Amendment 6).

Reads one or more prediction parquets written by scripts/benchmark_vol.py (later files
add/override model columns on the same asset-date rows), applies the same leakage-safe
walk-forward calibration to every model, forms walk-forward Granger-Ramanathan
combinations, and scores every table on ONE common row support per table.

Periods: full OOS, development (dates < 2019-01-01, used for all choices) and holdout
(dates >= 2019-01-01, used for nothing but the final report).

    python scripts/score_vol_benchmark.py [preds1.parquet preds2.parquet ...]
    -> results/benchmark_vol.json, results/benchmark_vol.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import walk_forward_calibrate, walk_forward_combine
from meridian.scoring import score_models

RESULTS = Path(__file__).resolve().parent.parent / "results"
SPLIT = pd.Timestamp("2019-01-01")
REF = "HAR"
EPS = 1e-12
MIN_COVERAGE = 0.98      # models below this share of rows are reported separately, not in the main table

BASE_MODELS = ["RW", "MA22", "EWMA", "GARCH", "GJR_GARCH", "EGARCH", "HAR_pooled", "HAR",
               "LHAR", "SHAR", "HARX", "Ridge", "RandomForest", "LightGBM", "TimeMixer"]
META_COLS = ["asset", "date", "y", "y_5", "y_22", "rv_next", "rv_next_5", "rv_next_22",
             "har_d", "vix", "fold", "breadth"]
CAL_EMBARGO = {"y": 22, "y_5": 30, "y_22": 45}   # calendar days >= label length + margin


def load(paths):
    P = None
    for p in paths:
        df = pd.read_parquet(p)
        if P is None:
            P = df
        else:
            new = [c for c in df.columns if c not in META_COLS]
            P = P.drop(columns=[c for c in new if c in P.columns]).merge(
                df[["asset", "date"] + new], on=["asset", "date"], how="left")
    P["date"] = pd.to_datetime(P["date"])
    return P.sort_values(["asset", "date"]).reset_index(drop=True)


def calibrate(P, col, target="y"):
    out = np.full(len(P), np.nan)
    for _, g in P.groupby("asset"):
        var, mask = walk_forward_calibrate(g["date"].to_numpy(), g[target].to_numpy(), g[col].to_numpy(),
                                           embargo=CAL_EMBARGO[target])
        idx = g.index.to_numpy()
        out[idx[mask]] = np.log(np.clip(var[mask], EPS, None))
    return out


def combine(P, cols, target="y"):
    out = np.full(len(P), np.nan)
    for _, g in P.groupby("asset"):
        var, mask = walk_forward_combine(g["date"].to_numpy(), g[target].to_numpy(),
                                         {c: g[c].to_numpy() for c in cols}, embargo=CAL_EMBARGO[target])
        idx = g.index.to_numpy()
        out[idx[mask]] = np.log(np.clip(var[mask], EPS, None))
    return out


def common_table(df, models, ref, label, rv_col="rv_next", y_col="y"):
    """Evaluation universe = rows where the reference and the target are available; a model
    is kept if it covers >= MIN_COVERAGE of that universe, and every kept model is scored on
    the intersection of their supports (one common row set for the whole table)."""
    base = df[np.isfinite(df[[ref, y_col, rv_col]].to_numpy()).all(axis=1)]
    cov = {m: float(np.isfinite(base[m].to_numpy()).mean()) for m in models}
    keep = [m for m in models if cov[m] >= MIN_COVERAGE]
    dropped = {m: cov[m] for m in models if m not in keep}
    sub = base[np.isfinite(base[keep].to_numpy()).all(axis=1)]
    res = score_models(sub, keep, ref, rv_col=rv_col, y_col=y_col)
    res["_meta"].update({"label": label, "coverage": cov, "excluded_low_coverage": dropped})
    return res


def main():
    paths = sys.argv[1:] or [str(RESULTS / "benchmark_vol_predictions.parquet")]
    P = load(paths)
    merid = sorted(c for c in P.columns if c.startswith("Meridian_M") and not c.endswith(("_y_5", "_y_22")))
    raw = [m for m in BASE_MODELS + merid if m in P.columns]

    for m in raw:
        P[f"{m}_cal"] = calibrate(P, m)
    cal = [f"{m}_cal" for m in raw]
    # walk-forward Granger-Ramanathan combinations
    combos = {"Combo_HAR_LGBM": ["HAR", "LightGBM"]}
    for v in merid:
        combos[f"Combo_HAR_LGBM_{v}"] = ["HAR", "LightGBM", v]
    for name, cols in combos.items():
        if all(c in P.columns for c in cols):
            P[name] = combine(P, cols)
    combo_cols = [c for c in combos if c in P.columns]

    core = P[~P["breadth"]]
    periods = {"full": core, "dev": core[core["date"] < SPLIT], "holdout": core[core["date"] >= SPLIT]}
    report = {"inputs": paths, "split_date": str(SPLIT.date()), "reference": REF, "tables": {}}

    for pname, df in periods.items():
        if df["date"].nunique() < 100:
            continue
        report["tables"][f"{pname}_raw_vs_calibrated"] = common_table(
            df, raw + cal + combo_cols, f"{REF}_cal", f"{pname}: raw + calibrated + combinations")
        report["tables"][f"{pname}_raw"] = common_table(df, raw, REF, f"{pname}: raw forecasts")

    # conservative check: against the STRONGER of the two HAR forms (raw per-asset HAR scores
    # better than its calibrated version on this data), plus per-year and per-asset ratios
    if "Meridian_M0" in P.columns:
        strong = [m for m in ["HAR", "HAR_cal", "LightGBM_cal", "Combo_HAR_LGBM", "Meridian_M0",
                              "Meridian_M0_cal", "Combo_HAR_LGBM_Meridian_M0"] if m in P.columns]
        hold = periods["holdout"]
        report["tables"]["holdout_vs_raw_HAR"] = common_table(hold, strong, "HAR",
                                                              "holdout vs raw per-asset HAR (strongest HAR form)")
        full = periods["full"]
        sub = full[np.isfinite(full[strong + ["y", "rv_next"]].to_numpy()).all(axis=1)].copy()
        from meridian.evalproto import qlike
        for m in strong:
            sub[f"_q_{m}"] = qlike(sub["rv_next"].to_numpy(), np.exp(sub[m].to_numpy()))
        by_year = sub.groupby(sub["date"].dt.year)
        by_asset = sub[sub["date"] >= SPLIT].groupby("asset")
        report["by_year_pct_vs_raw_HAR"] = {
            m: {int(y): float(v) for y, v in (by_year[f"_q_{m}"].mean() / by_year["_q_HAR"].mean() - 1).items()}
            for m in strong if m != "HAR"}
        report["holdout_by_asset_pct_vs_raw_HAR"] = {
            m: {a: float(v) for a, v in (by_asset[f"_q_{m}"].mean() / by_asset["_q_HAR"].mean() - 1).items()}
            for m in strong if m != "HAR"}

    # horizon cuts (calibrated per-asset HAR vs calibrated Meridian aux heads)
    for h, rvc in (("y_5", "rv_next_5"), ("y_22", "rv_next_22")):
        cols = [f"HAR_{h}"] + [f"{v}_{h}" for v in merid if f"{v}_{h}" in P.columns]
        cols = [c for c in cols if c in P.columns]
        if len(cols) > 1:
            for c in cols:
                P[f"{c}_cal"] = calibrate(P, c, target=h)
            core = P[~P["breadth"]]
            hold = core[core["date"] >= SPLIT]
            if hold["date"].nunique() < 100:
                continue
            report["tables"][f"holdout_{h}"] = common_table(
                hold, [f"{c}_cal" for c in cols], f"HAR_{h}_cal", f"holdout {h} (calibrated)",
                rv_col=rvc, y_col=h)

    # breadth (TLT, GLD): pooled models never trained on them
    br = P[P["breadth"]]
    if br[br["date"] >= SPLIT]["date"].nunique() >= 100:
        report["tables"]["breadth_holdout"] = common_table(
            br[br["date"] >= SPLIT], raw + cal, f"{REF}_cal", "breadth assets, holdout")

    # VIX-tercile regime cut on the full core calibrated table support
    full = report["tables"]["full_raw_vs_calibrated"]
    models_full = [m for m in full if m != "_meta"]
    sub = core[np.isfinite(core[models_full + ["y", "rv_next"]]).all(1)]
    terc = sub["vix"].quantile([1 / 3, 2 / 3]).to_numpy()
    lab = np.where(sub["vix"] <= terc[0], "calm", np.where(sub["vix"] <= terc[1], "normal", "stress"))
    report["regimes"] = {}
    for rg in ("calm", "normal", "stress"):
        r = score_models(sub[lab == rg], models_full, f"{REF}_cal", run_mcs=False)
        report["regimes"][rg] = {m: {"QLIKE": v["QLIKE"], "pct_vs_ref": v["pct_vs_ref"], "DM_p": v["DM_p"]}
                                 for m, v in r.items() if m != "_meta"}
        report["regimes"][rg]["_meta"] = r["_meta"]

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    (RESULTS / "benchmark_vol.json").write_text(json.dumps(clean(report), indent=2))
    hold = report["tables"].get("holdout_raw_vs_calibrated") or report["tables"]["full_raw_vs_calibrated"]
    pd.DataFrame({m: v for m, v in hold.items() if m != "_meta"}).T.to_csv(RESULTS / "benchmark_vol.csv")

    for key in ("dev_raw_vs_calibrated", "holdout_raw_vs_calibrated"):
        if key not in report["tables"]:
            continue
        t = report["tables"][key]
        m = t["_meta"]
        print(f"\n== {m['label']} | {m['date_range'][0]}..{m['date_range'][1]} | "
              f"{m['n_dates']} dates x {m['n_assets']} assets | ref={m['ref']} ==")
        rows = sorted(((k, v) for k, v in t.items() if k != "_meta"), key=lambda kv: kv[1]["QLIKE"])
        for k, v in rows:
            p = "" if v["DM_p"] is None else f"{v['DM_p']:.3f}"
            ci = "" if v["ci95_pct_vs_ref"] is None else \
                f"[{v['ci95_pct_vs_ref'][0]*100:+.1f},{v['ci95_pct_vs_ref'][1]*100:+.1f}]"
            print(f"  {k:28s} QLIKE={v['QLIKE']:.4f} {v['pct_vs_ref']*100:+6.1f}% {ci:16s} DM_p={p:6s} "
                  f"IC={v['IC']:.3f} hit={v['hit_rate']*100:.1f}% won={v['assets_beating_ref']} "
                  f"MCS={'Y' if v.get('in_MCS90') else 'n'}")
        if m["excluded_low_coverage"]:
            print("  excluded (coverage < 98%):", m["excluded_low_coverage"])
    print("\nsaved -> results/benchmark_vol.json, results/benchmark_vol.csv")


if __name__ == "__main__":
    main()
