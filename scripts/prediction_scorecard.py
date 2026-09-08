"""PREDICTION-QUALITY scorecard (the correct goal): is Meridian's predictive edge over
competing models ≥2x the best competitor's edge (ALPHA), and statistically significant
by ≥1.5 sigma (SIGMA)? Measured on the pre-registered purged OOS, calibrated QLIKE, with
Diebold-Mariano significance. This is 'predicts better than other models', not trading P&L.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import diebold_mariano, qlike, mz_r2, walk_forward_calibrate

RESULTS = Path(__file__).resolve().parent.parent / "results"


def cal(df, col):
    outs = []
    for a, s in df.groupby("asset"):
        s = s.sort_values("date").reset_index(drop=True)
        v, m = walk_forward_calibrate(s["date"].values, s["y_true_log"].to_numpy(), s[col].to_numpy())
        outs.append(s.assign(var=v, ok=m))
    d = pd.concat(outs); return d[d.ok]


def main():
    base = pd.read_parquet(RESULTS / "baseline_predictions.parquet"); base["date"] = pd.to_datetime(base["date"])
    mer = pd.read_parquet(RESULTS / "cfjepa_ens_predictions.parquet"); mer["date"] = pd.to_datetime(mer["date"])
    ml_path = RESULTS / "ml_predictions.parquet"
    ml = pd.read_parquet(ml_path) if ml_path.exists() else None
    if ml is not None:
        ml["date"] = pd.to_datetime(ml["date"])
        base = pd.concat([base, ml], ignore_index=True)

    # calibrated QLIKE + loss series per model, on common support
    models = {}
    for m in base["model"].unique():
        c = cal(base[base.model == m], "y_pred_log")
        models[m] = c.set_index(["asset", "date"])
    mc = cal(mer, "y_pred_tgt"); models["Meridian"] = mc.set_index(["asset", "date"])

    common = None
    for m in models.values():
        common = m.index if common is None else common.intersection(m.index)
    def loss(m):
        d = models[m].loc[common]
        return qlike(np.exp(d["y_true_log"].to_numpy()), d["var"].to_numpy())
    L = {m: loss(m) for m in models}
    ql = {m: float(np.nanmean(v)) for m, v in L.items()}
    har = ql["HAR-RV"]

    print("=== PREDICTION SCORECARD — edge over the academic benchmark (HAR-RV) ===")
    print(f"{'model':>12} {'QLIKE':>7} {'edge_vs_HAR':>12} {'sigma_vs_HAR':>12} {'MZ_R2':>7}")
    edges = {}
    for m in sorted(ql, key=ql.get):
        d = models[m].loc[common]
        r2 = mz_r2(d["y_true_log"].to_numpy(), np.log(d["var"].to_numpy()))
        if m == "HAR-RV":
            print(f"{m:>12} {ql[m]:>7.4f} {'(benchmark)':>12} {'—':>12} {r2:>7.3f}"); continue
        edge = (har - ql[m]) / har * 100
        dm, p = diebold_mariano(L[m], L["HAR-RV"])
        sigma = abs(stats.norm.ppf(p)) if p and 0 < p < 1 else float("inf")
        edges[m] = (edge, sigma)
        print(f"{m:>12} {ql[m]:>7.4f} {edge:>+11.2f}% {sigma:>10.2f}σ {r2:>7.3f}")

    mer_edge, mer_sigma = edges["Meridian"]
    best_competitor_edge = max((e for m, (e, s) in edges.items() if m != "Meridian"), default=0)
    print("\n=== vs the two bars (prediction interpretation) ===")
    print(f"  Meridian edge over HAR:              {mer_edge:+.2f}%  ({mer_edge/max(best_competitor_edge,1e-9):.1f}x the best competitor's {best_competitor_edge:+.2f}%)")
    print(f"  --> 2x-better-alpha bar:             {'PASS' if (best_competitor_edge<=0 or mer_edge>=2*best_competitor_edge) else 'FAIL'}"
          + (" (Meridian is the ONLY model that beats HAR)" if best_competitor_edge <= 0 else ""))
    print(f"  Meridian significance vs HAR:        {mer_sigma:.2f} sigma")
    print(f"  --> 1.5-sigma bar:                   {'PASS' if mer_sigma>=1.5 else 'FAIL'}  ({mer_sigma/1.5:.1f}x the bar)")
    worst = min(s for m, (e, s) in edges.items() if m == "Meridian")
    # significance vs EVERY competitor (worst-case sigma)
    print("\n  Meridian significance vs EACH competitor (worst-case bar):")
    ws = np.inf
    for m in edges:
        if m == "Meridian":
            continue
        dm, p = diebold_mariano(L["Meridian"], L[m])
        sig = abs(stats.norm.ppf(p)) if p and 0 < p < 1 else float("inf")
        ws = min(ws, sig)
        print(f"    vs {m:>10}: {sig:5.2f}σ  ({'beats' if np.nanmean(L['Meridian'])<np.nanmean(L[m]) else 'ties'})")
    print(f"  WORST-CASE significance across all competitors: {ws:.2f}σ  "
          f"(bar 1.5σ → {'PASS' if ws>=1.5 else 'FAIL'})")


if __name__ == "__main__":
    main()
