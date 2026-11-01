"""Score the tail-risk benchmark from saved predictions (PREREGISTRATION.md Amendment 7).

    python scripts/score_tail_benchmark.py --dev-only   # development period only (selection)
    python scripts/score_tail_benchmark.py              # all tables -> results/benchmark_tail.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.riskscore import score_tail

RESULTS = Path(__file__).resolve().parent.parent / "results"
SPLIT = pd.Timestamp("2019-01-01")
REF = "GJR_FHS"
ALPHAS = {"010": 0.010, "025": 0.025, "050": 0.050}
PRIMARY = "025"
MERIDIAN_CANDIDATES = ["Meridian_FHS", "Meridian_EVT", "Meridian_FHS_ACI"]
LABEL = {"HS": "Historical simulation (500d)", "RiskMetrics": "RiskMetrics EWMA, normal",
         "GARCH_N": "GARCH(1,1)-normal", "GARCH_t": "GARCH(1,1)-t", "GJR_skewt": "GJR-GARCH-skewed-t",
         "GJR_FHS": "GJR-GARCH + FHS (reference)", "GJR_EVT": "GJR-GARCH + EVT (McNeil-Frey)",
         "CAViaR_AS": "CAViaR asymmetric slope", "HAR_FHS": "HAR-RV + FHS", "HAR_EVT": "HAR-RV + EVT",
         "LightGBM_FHS": "LightGBM vol + FHS", "Meridian_FHS": "Meridian + FHS",
         "Meridian_EVT": "Meridian + EVT", "Meridian_FHS_ACI": "Meridian + FHS + adaptive conformal"}


def models_in(P):
    var = sorted({c[4:-4] for c in P.columns if c.startswith("VaR_")})
    es = [m for m in var if f"ES_{m}_{PRIMARY}" in P.columns]
    return var, es


def support(df, models, es_models):
    cols = ["r_next"] + [f"VaR_{m}_{k}" for m in models for k in ALPHAS] + \
           [f"ES_{m}_{k}" for m in es_models for k in ALPHAS]
    return df[np.isfinite(df[cols].to_numpy(float)).all(1)]


def tables_for(df, models, es_models, run_mcs=True):
    return {k: score_tail(df, models, REF, k, a, es_models, run_mcs=run_mcs) for k, a in ALPHAS.items()}


def fmt_p(p):
    return "—" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(t, k, es_only=False):
    a = ALPHAS[k]
    rows = [(m, v) for m, v in t.items() if m != "_meta" and (v["has_es"] or not es_only)]
    if es_only:
        rows.sort(key=lambda mv: mv[1]["FZ0"])
        lines = [f"| Model | FZ0 loss ↓ | Δ vs reference (95% CI) | DM p | In MCS | Violations (target {a:.1%}) "
                 "| Assets passing CC / DQ | ES ratio (realized/forecast) |", "|---|---|---|---|---|---|---|---|"]
        for m, v in rows:
            ci = "—" if v["FZ0_diff_ci95"] is None else \
                f"{v['FZ0_diff_vs_ref']:+.4f} [{v['FZ0_diff_ci95'][0]:+.4f}, {v['FZ0_diff_ci95'][1]:+.4f}]"
            lines.append(f"| {LABEL.get(m, m)} | {v['FZ0']:.4f} | {ci} | {fmt_p(v['FZ0_DM_p'])} | "
                         f"{'yes' if v.get('FZ0_in_MCS90') else 'no'} | {v['violation_rate']:.2%} | "
                         f"{v['assets_pass_cc']}/11 · {v['assets_pass_dq']}/11 | {v['es_ratio']:.2f} |")
    else:
        rows.sort(key=lambda mv: mv[1]["tick"])
        lines = [f"| Model | Quantile loss ×10³ ↓ | vs reference (95% CI) | DM p | In MCS | Violations (target {a:.1%}) "
                 "| Assets passing Kupiec / CC / DQ |", "|---|---|---|---|---|---|---|"]
        for m, v in rows:
            ci = "—" if v["tick_ci95_pct_vs_ref"] is None else \
                f"{v['tick_pct_vs_ref']:+.1%} [{v['tick_ci95_pct_vs_ref'][0]:+.1%}, {v['tick_ci95_pct_vs_ref'][1]:+.1%}]"
            lines.append(f"| {LABEL.get(m, m)} | {v['tick'] * 1e3:.3f} | {ci} | {fmt_p(v['tick_DM_p'])} | "
                         f"{'yes' if v.get('tick_in_MCS90') else 'no'} | {v['violation_rate']:.2%} | "
                         f"{v['assets_pass_kupiec']}/11 · {v['assets_pass_cc']}/11 · {v['assets_pass_dq']}/11 |")
    return "\n".join(lines)


def main():
    dev_only = "--dev-only" in sys.argv
    P = pd.read_parquet(RESULTS / "benchmark_tail_predictions.parquet")
    P["date"] = pd.to_datetime(P["date"])
    models, es_models = models_in(P)
    core = support(P[~P["breadth"]], models, es_models)
    dev, hold = core[core["date"] < SPLIT], core[core["date"] >= SPLIT]

    dev_t = tables_for(dev, models, es_models, run_mcs=not dev_only)
    sel = {m: dev_t[PRIMARY][m]["FZ0"] for m in MERIDIAN_CANDIDATES}
    selected = min(sel, key=sel.get)
    print(f"development {dev_t[PRIMARY]['_meta']['date_range']} FZ0@2.5%:")
    for m, v in sorted(((m, v) for m, v in dev_t[PRIMARY].items() if m != "_meta" and v["has_es"]),
                       key=lambda mv: mv[1]["FZ0"]):
        print(f"  {m:18s} FZ0={v['FZ0']:.4f}  diff={v['FZ0_diff_vs_ref']:+.4f}")
    print("selected Meridian tail variant:", selected)
    if dev_only:
        return

    hold_t = tables_for(hold, models, es_models)
    # head-to-head: the selected Meridian variant vs the HAR-filtered tails (Amendment 7 record)
    h2h_models = [selected, "HAR_FHS", "HAR_EVT"]
    h2h = {k: score_tail(hold, h2h_models, "HAR_FHS", k, a, h2h_models, run_mcs=False)
           for k, a in ALPHAS.items()}
    report = {"reference": REF, "split_date": str(SPLIT.date()), "primary": "FZ0 at 2.5%",
              "meridian_dev_FZ0": sel, "selected": selected,
              "tables": {"dev": dev_t, "holdout": hold_t, "holdout_vs_HAR_FHS": h2h}}
    br = support(P[P["breadth"]], models, es_models)
    brh = br[br["date"] >= SPLIT]
    if brh["date"].nunique() > 250:
        report["tables"]["breadth_holdout"] = {PRIMARY: score_tail(brh, models, REF, PRIMARY,
                                                                   ALPHAS[PRIMARY], es_models, run_mcs=False)}

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    report = clean(report)
    (RESULTS / "benchmark_tail.json").write_text(json.dumps(report, indent=2))

    m = report["tables"]["holdout"][PRIMARY]["_meta"]
    md = [
        "# Tail-risk benchmark (VaR / Expected Shortfall)",
        "",
        "Generated by `scripts/score_tail_benchmark.py` from `results/benchmark_tail_predictions.parquet` "
        "(PREREGISTRATION.md Amendment 7). Do not edit by hand.",
        "",
        f"Target: next-day close-to-close log return. Holdout {m['date_range'][0]} → {m['date_range'][1]} "
        f"({m['n_dates']} dates × {m['n_assets']} assets, {m['n_rows']} forecasts), never used for any choice. "
        f"Reference: {LABEL[REF]}. Significance on date-level losses (HAC DM, block bootstrap, MCS 90%). "
        f"CC = Christoffersen conditional coverage, DQ = Engle–Manganelli dynamic quantile; an asset passes at p > 0.05. "
        f"Meridian tail variant selected on the development period: **{LABEL[selected]}**.",
        "",
        "## Primary: joint VaR + ES at 2.5% (FZ0 loss, lower is better)",
        "", md_table(report["tables"]["holdout"]["025"], "025", es_only=True), "",
    ]
    h = report["tables"]["holdout_vs_HAR_FHS"]
    md += [f"## Head-to-head: {LABEL[selected]} vs HAR-RV + FHS (holdout)", "",
           "Both sides filter returns by a realized-volatility forecast and read the tail from past "
           "standardized returns, so this mostly isolates the value of Meridian's volatility forecast over HAR's.", "",
           "| Level | FZ0 Δ (95% CI) | FZ0 DM p | Quantile loss vs HAR-FHS (95% CI) | Quantile DM p |",
           "|---|---|---|---|---|"]
    for k in ALPHAS:
        v = h[k][selected]
        md.append(f"| {ALPHAS[k]:.1%} | {v['FZ0_diff_vs_ref']:+.4f} [{v['FZ0_diff_ci95'][0]:+.4f}, "
                  f"{v['FZ0_diff_ci95'][1]:+.4f}] | {fmt_p(v['FZ0_DM_p'])} | {v['tick_pct_vs_ref']:+.1%} "
                  f"[{v['tick_ci95_pct_vs_ref'][0]:+.1%}, {v['tick_ci95_pct_vs_ref'][1]:+.1%}] | "
                  f"{fmt_p(v['tick_DM_p'])} |")
    md.append("")
    for k in ALPHAS:
        md += [f"## VaR at {ALPHAS[k]:.1%} (quantile loss on scaled returns)", "",
               md_table(report["tables"]["holdout"][k], k), ""]
    md += ["## Development period (2.5%, used for the Meridian variant choice)", "",
           md_table(report["tables"]["dev"]["025"], "025", es_only=True), ""]
    if "breadth_holdout" in report["tables"]:
        md += ["## Breadth assets (TLT, GLD), holdout, 2.5% — pooled vol models never trained on them", "",
               md_table(report["tables"]["breadth_holdout"]["025"], "025", es_only=True).replace("/11", "/2"), ""]
    (RESULTS / "benchmark_tail.md").write_text("\n".join(md))

    # summary block for README.md / MODEL_CARD.md
    from scripts.render_benchmark_md import inject
    v1 = h["010"][selected]; v25 = h["025"][selected]
    block = "\n".join([
        f"Holdout {m['date_range'][0]} → {m['date_range'][1]} ({m['n_dates']} dates × {m['n_assets']} assets), "
        f"next-day return VaR and ES at 2.5%, reference {LABEL[REF]}. Lower FZ0 loss is better.", "",
        md_table(report["tables"]["holdout"]["025"], "025", es_only=True), "",
        f"Head-to-head against HAR-RV + FHS (both volatility-filtered): FZ0 Δ {v25['FZ0_diff_vs_ref']:+.4f} "
        f"[{v25['FZ0_diff_ci95'][0]:+.4f}, {v25['FZ0_diff_ci95'][1]:+.4f}], DM p {fmt_p(v25['FZ0_DM_p'])} at 2.5%; "
        f"1% VaR quantile loss {v1['tick_pct_vs_ref']:+.1%} [{v1['tick_ci95_pct_vs_ref'][0]:+.1%}, "
        f"{v1['tick_ci95_pct_vs_ref'][1]:+.1%}], DM p {fmt_p(v1['tick_DM_p'])}. Full tables: `results/benchmark_tail.md`."])
    for doc in ("README.md", "MODEL_CARD.md"):
        inject(RESULTS.parent / doc, block, "<!-- TAIL:START -->", "<!-- TAIL:END -->")
    print("saved -> results/benchmark_tail.json, results/benchmark_tail.md")


if __name__ == "__main__":
    main()
