"""Score the frozen-model deployment (scripts/deploy_frozen.py) and render DEPLOYMENT.md.

Every number in results/deploy/deploy_score.json and DEPLOYMENT.md comes from this file,
computed from results/deploy/predictions_{A,B}.parquet. Nothing is typed by hand.

Scored on the pre-registered CORE universe (8 equities/ETFs + 3 FX); TLT/GLD are reported
separately. All models are compared on ONE common set of rows (every model finite).
Significance uses the DATE-level loss series (cross-asset mean per date), Newey-West HAC,
and a moving-block bootstrap over dates. With a few weeks of data these tests have little
power; the JSON stores the CI width so that is visible, not hidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.evalproto import qlike
from meridian.scoring import hac_mean_test, block_bootstrap_ratio_ci, nw_lag

ROOT = Path(__file__).resolve().parent.parent
DEP = ROOT / "results" / "deploy"
MODELS = {
    "A": ["Meridian_M0", "Meridian_M0_rtvix", "HAR_frozen", "HAR", "HAR_pooled", "LHAR", "SHAR", "HARX", "Ridge", "RandomForest",
          "LightGBM", "TimeMixer", "RW", "MA22", "EWMA", "GARCH", "GJR_GARCH", "EGARCH"],
    "B": ["Meridian_legacy", "Meridian_legacy_raw", "Meridian_legacy_cal", "HAR", "HAR_pooled", "LHAR", "SHAR", "HARX", "Ridge",
          "RandomForest", "LightGBM", "TimeMixer", "RW", "MA22", "EWMA", "GARCH", "GJR_GARCH", "EGARCH"],
}
REF = "HAR"
LABEL = {"Meridian_M0": "Meridian M0 (frozen, 5 seeds)",
         "Meridian_M0_rtvix": "Meridian M0, real-time VIX vintage (previous-day VIX/yields)", "HAR_frozen": "HAR-RV per asset, frozen with M0",
         "Meridian_legacy": "Meridian legacy (frozen; Jensen term from pre-cutoff rows)",
         "Meridian_legacy_raw": "Meridian legacy, raw E[log RV] head (no Jensen term)",
         "Meridian_legacy_cal": "Meridian legacy + level offset fit on pre-cutoff rows (secondary)",
         "HAR": "HAR-RV per asset (reference)", "HAR_pooled": "HAR-RV pooled", "LHAR": "LHAR (leverage)",
         "SHAR": "SHAR (signed)", "HARX": "HAR-X (+VIX)", "Ridge": "Ridge", "RandomForest": "Random forest",
         "LightGBM": "LightGBM", "TimeMixer": "TimeMixer", "RW": "Random walk (log RV_t)", "MA22": "22-day mean",
         "EWMA": "EWMA RiskMetrics", "GARCH": "GARCH(1,1)", "GJR_GARCH": "GJR-GARCH(1,1)", "EGARCH": "EGARCH(1,1)"}


def score_window(P, models, ref=REF):
    P = P[np.isfinite(P[models + ["rv_next", "y"]]).all(axis=1)].copy()
    for m in models:
        P[f"q_{m}"] = qlike(P["rv_next"].to_numpy(), np.exp(P[m].to_numpy()))
    T = P["target_date"].nunique()
    qd = {m: P.groupby("target_date")[f"q_{m}"].mean().sort_index() for m in models}
    qs = {m: P.groupby("target_date")[f"q_{m}"].sum().sort_index() for m in models}
    ref_asset = P.groupby("asset")[f"q_{ref}"].mean()
    block = max(1, int(round(T ** (1 / 3))))
    out = {}
    for m in models:
        q = float(P[f"q_{m}"].mean())
        rec = {"QLIKE": q, "RMSE_log": float(np.sqrt(((P["y"] - P[m]) ** 2).mean())),
               "pct_vs_ref": q / float(P[f"q_{ref}"].mean()) - 1.0,
               "assets_beating_ref": int((P.groupby("asset")[f"q_{m}"].mean() < ref_asset).sum()),
               "dates_beating_ref": int((qd[m] < qd[ref]).sum()),
               "ci95_pct_vs_ref": None, "DM_p": None}
        if m != ref and T >= 20:
            _, p = hac_mean_test((qd[m] - qd[ref]).to_numpy(), nw_lag(T))
            lo, hi = block_bootstrap_ratio_ci(qs[m].to_numpy(), qs[ref].to_numpy(), block=block, reps=4000)
            rec.update(DM_p=p, ci95_pct_vs_ref=[lo, hi])
        out[m] = rec
    order = sorted(models, key=lambda m: out[m]["QLIKE"])
    for i, m in enumerate(order):
        out[m]["rank"] = i + 1
    meta = {"n_rows": int(len(P)), "n_dates": int(T), "n_assets": int(P["asset"].nunique()),
            "target_range": [str(P["target_date"].min().date()), str(P["target_date"].max().date())],
            "ref": ref, "bootstrap_block": block,
            "tests_run": T >= 20,
            "note": ("DM test and CI need >= 20 target dates; not computed" if T < 20 else
                     "few weeks of data: low power, wide intervals")}
    per_asset = {a: {m: float(g[f"q_{m}"].mean()) for m in models} for a, g in P.groupby("asset")}
    return {"models": out, "meta": meta, "per_asset": per_asset}


def main():
    meta = json.loads((DEP / "deploy_meta.json").read_text())
    res = {"deploy_meta": {k: meta[k] for k in ("run_at_utc", "git_commit", "data_last_target")}, "snapshots": {}}
    for snap, models in MODELS.items():
        P = pd.read_parquet(DEP / f"predictions_{snap}.parquet")
        cut = pd.Timestamp(meta["snapshots"][snap]["train_label_end_max"])
        assert (P["target_date"] > cut).all(), "a scored target is inside the training period"
        out = {"name": meta["snapshots"][snap]["name"], "train_label_end_max": str(cut.date()), "windows": {}}
        for w, (lo, hi) in meta["snapshots"][snap]["windows"].items():
            m = (P["target_date"] >= lo) & (P["target_date"] <= hi)
            core, br = P[m & ~P["breadth"]], P[m & P["breadth"]]
            if core.empty:
                continue
            out["windows"][w] = {"core": score_window(core, models),
                                 "breadth": score_window(br, models) if len(br) else None}
        res["snapshots"][snap] = out
    res["forward_log"] = score_forward_log()
    (DEP / "deploy_score.json").write_text(json.dumps(res, indent=2))
    render(res)
    print((ROOT / "DEPLOYMENT.md").read_text())


def score_forward_log():
    """The strictest evidence: forecasts written to results/forward/forecast_log.csv BEFORE their
    outcome existed (precommitted), scored against the realized variance in the deployment data."""
    log = pd.read_csv(ROOT / "results" / "forward" / "forecast_log.csv", parse_dates=["origin_date"])
    log = log[log["precommitted"] & ~log["breadth"]]
    P = pd.concat([pd.read_parquet(DEP / f"predictions_{s}.parquet") for s in ("A", "B")])
    truth = P.drop_duplicates(["asset", "date"]).set_index(["asset", "date"])[["rv_next", "target_date"]]
    S = log.join(truth, on=["asset", "origin_date"], how="inner")
    S = S[np.isfinite(S["rv_next"])]
    if S.empty:
        return None
    for m in ("Meridian_M0", "HAR"):
        S[f"q_{m}"] = qlike(S["rv_next"].to_numpy(), np.exp(S[m].to_numpy()))
    by = S.groupby("target_date")[["q_Meridian_M0", "q_HAR"]].mean()
    return {"n_rows": int(len(S)), "n_dates": int(len(by)),
            "target_dates": [str(x.date()) for x in by.index],
            "QLIKE_Meridian_M0": float(S["q_Meridian_M0"].mean()), "QLIKE_HAR": float(S["q_HAR"].mean()),
            "pct_vs_HAR": float(S["q_Meridian_M0"].mean() / S["q_HAR"].mean() - 1),
            "per_date_pct": {str(k.date()): float(v["q_Meridian_M0"] / v["q_HAR"] - 1) for k, v in by.iterrows()},
            "assets_beating_HAR": int((S.groupby("asset")["q_Meridian_M0"].mean() < S.groupby("asset")["q_HAR"].mean()).sum()),
            "n_assets": int(S["asset"].nunique())}


def fmt_ci(ci):
    return "—" if ci is None else f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]"


def fmt_p(p):
    return "—" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def table(sc):
    mods = sorted(sc["models"], key=lambda m: sc["models"][m]["QLIKE"])
    na = sc["meta"]["n_assets"]
    rows = ["| Rank | Model | QLIKE ↓ | vs HAR (95% CI) | DM p | Assets beating HAR | Dates beating HAR |",
            "|---|---|---|---|---|---|---|"]
    for m in mods:
        r = sc["models"][m]
        rows.append(f"| {r['rank']} | {LABEL.get(m, m)} | {r['QLIKE']:.4f} | {100*r['pct_vs_ref']:+.1f}% {fmt_ci(r['ci95_pct_vs_ref'])} "
                    f"| {fmt_p(r['DM_p'])} | {r['assets_beating_ref']}/{na} | {r['dates_beating_ref']}/{sc['meta']['n_dates']} |")
    return "\n".join(rows)


def render(res):
    L = ["<!-- generated by scripts/score_deploy.py from results/deploy/deploy_score.json — do not edit by hand -->",
         "# Frozen-model deployment: October and November 2026", "",
         "The already-trained Meridian models were loaded from their saved weights (sha256-checked), "
         "**not trained, fine-tuned or recalibrated**, and run forward on market data that arrived after their "
         "training period. Every comparison model was fit once on data ending at the same cut-off and then also "
         "run forward frozen. Target: next-day Garman–Klass realized variance; loss: QLIKE (lower is better). "
         "Reference: per-asset HAR-RV. Pipeline: `scripts/deploy_data.py` → `scripts/deploy_frozen.py` → "
         "`scripts/score_deploy.py`; independent checks: `scripts/audit_deploy.py` "
         "(results in `results/deploy/audit_report.md`).", "",
         "## What is the WM component, and what could be deployed", "",
         "**Meridian-WM** (`meridian/meridian_wm.py`, class `MeridianWM`) is the bridged world-model network: "
         "(1) a sticky switching-SSM belief core (`meridian/switching.py`, K=3 soft regimes, posterior α_t, regime = argmax α_t) "
         "with its own QLIKE head; (2) a decoupled plain diagonal-SSM volatility core with an EMA target encoder whose "
         "smoothed belief feeds the point log-variance head (QLIKE); (3) a small Student-t head over next-day returns "
         "(tail heaviness and scale) reading the stop-gradient belief; (4) a JEPA predictor + SIGReg auxiliary loss whose "
         "latent prediction error is the surprise score. The wider *world model* (`WORLD_MODEL.md`, `scripts/world_model.py`) "
         "is a bank of separately trained modules (volatility, tail, regime, factor, connectedness, propagation, covariance) "
         "linked by inspectable combiners into `results/world_state.json`.", "",
         "**No trained `MeridianWM` weights exist in the repository**: `scripts/run_wm.py`, `scripts/live.py` and "
         "`scripts/world_model.py` train it from scratch on every run and never save it. Deploying it would require training, "
         "so it is not deployed here. Its pre-audit numbers stay withdrawn (`WORLD_MODEL.md`). What *is* saved, and is deployed "
         "below without any training, is the world model's **volatility module** in two frozen snapshots:", "",
         "- **Snapshot A** — `results/forward/frozen/` Meridian M0 (the audited configuration: SSM belief core, HAR-anchored, "
         "QLIKE-trained, 5 seeds), trained on labels through 2026-10-30. October 2026 is inside its training data, so only "
         "November 2026 is scored for it.",
         "- **Snapshot B** — `results/meridian_final.pt`, the pre-audit single-seed Meridian (11 inputs, MSE head = E[log RV]). "
         "Its cut-off was not recorded; audit check T4 reconstructs it from the saved feature scaler: it was trained on the "
         "*uncleaned* pre-audit bars with origins through late July 2026. October and November 2026 are therefore more than "
         "two months out of sample. Its comparison models were fit through 2026-09-08 (the file's commit date), which can "
         "only favour them. Because it was trained on uncleaned FX bars (read as near-zero variance), its head is biased low "
         "in every year, including its own training years.", ""]
    for snap, s in res["snapshots"].items():
        L += [f"## Snapshot {snap}: {s['name']}", "",
              f"Training labels end **{s['train_label_end_max']}**; only targets after that date are scored.", ""]
        for w, sc in s["windows"].items():
            c = sc["core"]; mt = c["meta"]
            L += [f"### {w} — core universe ({mt['n_assets']} assets × {mt['n_dates']} target dates = {mt['n_rows']} forecasts, "
                  f"{mt['target_range'][0]} → {mt['target_range'][1]})", "", table(c), "",
                  f"_{mt['note']}._", ""]
            if sc["breadth"]:
                b = sc["breadth"]
                top = sorted(b["models"], key=lambda m: b["models"][m]["QLIKE"])
                pos = {m: i + 1 for i, m in enumerate(top)}
                mer = [m for m in top if m.startswith("Meridian")]
                L += [f"Breadth check (TLT, GLD; not in the pre-registered universe): "
                      + ", ".join(f"{LABEL.get(m, m)} rank {pos[m]}/{len(top)} ({100*b['models'][m]['pct_vs_ref']:+.1f}% vs HAR)" for m in mer) + ".", ""]
    fl = res.get("forward_log")
    if fl:
        L += ["## Pre-committed real-time log (strictest test)", "",
              f"Forecasts written to `results/forward/forecast_log.csv` before their outcome existed: {fl['n_rows']} core "
              f"forecasts over {fl['n_dates']} target dates ({', '.join(fl['target_dates'])}). Meridian M0 QLIKE "
              f"{fl['QLIKE_Meridian_M0']:.4f} vs frozen HAR {fl['QLIKE_HAR']:.4f} (**{100*fl['pct_vs_HAR']:+.1f}%**), "
              f"Meridian better on {fl['assets_beating_HAR']}/{fl['n_assets']} assets; per date: "
              + ", ".join(f"{k} {100*v:+.1f}%" for k, v in fl["per_date_pct"].items()) + ". "
              "These used the previous day's VIX (FRED delay), so they correspond to the real-time-vintage row in "
              "the Snapshot A table, not to the same-day row.", ""]
    open(ROOT / "DEPLOYMENT.md", "w").write("\n".join(L) + "\n")
    readme = ROOT / "README.md"
    txt = readme.read_text()
    a, b = "<!-- DEPLOY:START -->", "<!-- DEPLOY:END -->"
    if a in txt and b in txt:
        txt = txt[:txt.index(a) + len(a)] + "\n" + readme_block(res) + "\n" + txt[txt.index(b):]
        readme.write_text(txt)


def readme_block(res):
    out = []
    for snap, w in (("B", "Oct-2026"), ("B", "Nov-2026"), ("A", "Nov-2026")):
        sc = res["snapshots"][snap]["windows"].get(w)
        if not sc:
            continue
        c = sc["core"]; mods = c["models"]
        mer = [m for m in mods if m.startswith("Meridian")]
        best = min(mods, key=lambda m: mods[m]["QLIKE"])
        parts = [f"{LABEL[m]}: rank {mods[m]['rank']}/{len(mods)}, {100*mods[m]['pct_vs_ref']:+.1f}% vs HAR"
                 + (f" {fmt_ci(mods[m]['ci95_pct_vs_ref'])}" if mods[m]["ci95_pct_vs_ref"] else "") for m in mer]
        out.append(f"- **Snapshot {snap}, {w}** ({c['meta']['n_dates']} target dates × {c['meta']['n_assets']} assets): "
                   + "; ".join(parts) + f". Best model: {LABEL[best]} ({100*mods[best]['pct_vs_ref']:+.1f}%).")
    fl = res.get("forward_log")
    if fl:
        out.append(f"- **Pre-committed live log** ({fl['n_dates']} target dates): Meridian M0 {100*fl['pct_vs_HAR']:+.1f}% vs frozen HAR.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
