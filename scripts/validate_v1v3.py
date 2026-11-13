"""PREREGISTRATION.md Amendment 8, tests V1 and V3.

    python scripts/validate_v1v3.py run      # forecasts -> results/validation/v1v3_predictions.parquet
    python scripts/validate_v1v3.py score    # scores    -> results/validation/v1v3_score.json

V1  Snapshot B (results/meridian_final.pt, last training origin 2026-07-28 per audit T4) run
    forward untouched on targets 2026-08-03 .. last complete day. Comparison models fit ONCE on
    rows whose 1-day label ends <= 2026-07-29, then run forward frozen.
V3  Meridian-WM (default config, 5 seeds) trained ONCE on the same rows, saved + sha256-hashed
    in results/wm/frozen/, then run forward untouched on the same targets.

Volatility: QLIKE vs per-asset HAR (core universe), date-level DM (HAC), block-bootstrap CI,
MCS(10%), per-month split. Tail: next-day VaR/ES at 2.5% and 1% from each volatility model
through filtered historical simulation (and EVT for Meridian-B, the Amendment-7 selection),
GJR-GARCH + FHS reference, plus the WM's own Student-t head; FZ0 loss, Kupiec, Christoffersen.
Forecasts are also produced for the history before the cut-off (with frozen parameters) only
because filtered tails need a past of standardized returns; those rows are never scored.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.benchmark_vol as bv
import scripts.benchmark_wm as bw
import scripts.deploy_frozen as dfz
from meridian.evalproto import qlike
from meridian.features import realized_variance
from meridian.meridian_wm import MeridianWM, WMConfig
from meridian.riskscore import fz0_loss, kupiec, christoffersen_cc
from meridian.scoring import hac_mean_test, block_bootstrap_ratio_ci, nw_lag
from meridian.windows import FEATURES, train_scaler
from scripts.benchmark_tail import filtered_tail

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "results" / "validation"
WMF = ROOT / "results" / "wm" / "frozen"
CUT_ORIGIN, CUT_LABEL = pd.Timestamp("2026-07-28"), pd.Timestamp("2026-07-29")
EVAL_FROM = pd.Timestamp("2026-08-03")
HIST_FROM = pd.Timestamp("2021-06-01")      # forecasts from here (tails need ~1000 past z's)
N_WM = 5
ALPHAS = {"010": 0.010, "025": 0.025}
VOL_MODELS = ["Meridian_legacy", "Meridian_legacy_cal", "Meridian_legacy_raw", "MeridianWM_frozen",
              "HAR", "HAR_pooled", "LHAR", "SHAR", "HARX", "Ridge", "RandomForest", "LightGBM", "TimeMixer",
              "RW", "MA22", "EWMA", "GARCH", "GJR_GARCH", "EGARCH"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_wm(R, frames, asset_win):
    """Train MeridianWM once (if not already frozen) on rows with labels <= CUT_LABEL."""
    cutoff, test_start = CUT_LABEL, CUT_LABEL + pd.Timedelta(days=1)
    core = {a: v for a, v in asset_win.items() if a in bv.CORE_ASSETS}
    if (WMF / "manifest.json").exists():
        man = json.loads((WMF / "manifest.json").read_text())
        for n, h in man["sha256"].items():
            if sha(WMF / n) != h:
                sys.exit(f"frozen WM file {n} changed since freeze — refusing to deploy")
    else:
        WMF.mkdir(parents=True, exist_ok=True)
        mean, std = train_scaler(frames, core, cutoff)
        W = bw.wm_windows(asset_win, frames, mean, std)
        tr = {k: [] for k in ("ctx", "fut", "y", "r_next")}
        last_origin, last_label = [], []
        for a, w in W.items():
            if a not in bv.CORE_ASSETS:
                continue
            m = (w["dates"] < cutoff) & (w["end_1"] < np.datetime64(test_start))
            for k in tr:
                tr[k].append(w[k][m])
            last_origin.append(w["dates"][m].max()); last_label.append(pd.Timestamp(w["end_1"][m].max()))
        tr = {k: np.concatenate(v) for k, v in tr.items()}
        files = {}
        torch.set_num_threads(2)
        for s in range(N_WM):
            t0 = time.time()
            m = bw.train_wm(WMConfig(n_features=len(FEATURES), window=bv.L_NEURAL, seed=7000 + s), tr)
            p = WMF / f"wm_seed{s}.pt"
            torch.save(m.state_dict(), p)
            files[p.name] = sha(p)
            print(f"  frozen WM seed {s} trained ({time.time() - t0:.0f}s)", flush=True)
        np.savez(WMF / "scaler.npz", mean=mean, std=std)
        files["scaler.npz"] = sha(WMF / "scaler.npz")
        man = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model": "MeridianWM default WMConfig, 5 seeds (7000-7004), 8 epochs",
               "train_origin_max": str(max(last_origin).date()), "train_label_end_max": str(max(last_label).date()),
               "train_rows": int(len(tr["y"])), "sha256": files}
        (WMF / "manifest.json").write_text(json.dumps(man, indent=2))
    sc = np.load(WMF / "scaler.npz")
    models = []
    for s in range(N_WM):
        m = MeridianWM(WMConfig(n_features=len(FEATURES), window=bv.L_NEURAL, seed=7000 + s))
        m.load_state_dict(torch.load(WMF / f"wm_seed{s}.pt"))
        m.eval()
        models.append(m)
    return models, sc["mean"], sc["std"], json.loads((WMF / "manifest.json").read_text())


@torch.no_grad()
def wm_forecast(models, X):
    t = torch.tensor(X)
    outs = [m.forward(t) for m in models]
    return (np.mean([o["vol"].numpy() for o in outs], 0), np.mean([o["df"].numpy() for o in outs], 0),
            np.mean([o["scale"].numpy() for o in outs], 0))


def run():
    t0 = time.time()
    VAL.mkdir(parents=True, exist_ok=True)
    dfz.use_snapshot()
    R, Wlog, Wlev, d, frames, asset_win = bv.build()
    cutoff, test_start = CUT_LABEL, CUT_LABEL + pd.Timedelta(days=1)     # date < cutoff, end_1 < test_start
    te_m = (R["date"] >= HIST_FROM).to_numpy()
    te, base, fits, tinfo = dfz.fit_and_forecast_baselines(R, Wlog, Wlev, d, cutoff, test_start, te_m)
    print(f"  baselines done [{time.time() - t0:.0f}s]", flush=True)
    P = te[["asset", "date", "end_1", "y", "rv_next", "har_d"]].rename(columns={"end_1": "target_date"}).copy()
    for k, v in base.items():
        P[k] = v

    mB, meanB, stdB = dfz.load_B()
    wms, meanW, stdW, wman = frozen_wm(R, frames, asset_win)
    print(f"  frozen WM ready [{time.time() - t0:.0f}s]", flush=True)
    rows = []
    for a, f in frames.items():
        dts, X = dfz.windows_for(f, dfz.LEGACY_FEATURES, meanB, stdB)
        keep = dts >= HIST_FROM
        b = dfz.run([mB], X[keep])
        dts2, X2 = dfz.windows_for(f, FEATURES, meanW, stdW)
        keep2 = dts2 >= HIST_FROM
        v, df_, sc_ = wm_forecast(wms, X2[keep2])
        A = pd.DataFrame({"asset": a, "date": dts[keep], "Meridian_legacy_raw": b})
        B = pd.DataFrame({"asset": a, "date": dts2[keep2], "MeridianWM_frozen": v, "WM_df": df_, "WM_scale": sc_})
        rows.append(A.merge(B, on=["asset", "date"], how="outer"))
    P = P.merge(pd.concat(rows), on=["asset", "date"], how="left")
    pre = P[(P["target_date"] <= CUT_LABEL) & (P["date"] >= CUT_LABEL - pd.Timedelta(days=730)) & ~P["asset"].isin(bv.BREADTH_ASSETS)]
    res = pre["y"] - pre["Meridian_legacy_raw"]
    P["Meridian_legacy"] = P["Meridian_legacy_raw"] + 0.5 * float(res.var())
    P["Meridian_legacy_cal"] = P["Meridian_legacy_raw"] + float(res.mean()) + 0.5 * float(res.var())

    # tails
    P["r_next"] = np.nan
    P["GJR_sig"] = np.nan
    for a in P["asset"].unique():
        rf = realized_variance(d["prices"][a])
        ret = rf["ret"]
        ia = (P["asset"] == a).to_numpy()
        P.loc[ia, "r_next"] = ret.shift(-1).reindex(P.loc[ia, "date"]).to_numpy()
        nxt = bv.garch_next(ret.to_numpy(), np.asarray(ret.index < cutoff), "GJR_GARCH")
        if nxt is not None:
            P.loc[ia, "GJR_sig"] = np.sqrt(pd.Series(nxt, index=ret.index).reindex(P.loc[ia, "date"]).to_numpy())
    P = P.sort_values(["asset", "date"]).reset_index(drop=True)
    tail_specs = {"Meridian_B_EVT": ("Meridian_legacy_raw", "EVT"), "Meridian_B_FHS": ("Meridian_legacy_raw", "FHS"),
                  "MeridianWM_FHS": ("MeridianWM_frozen", "FHS"), "HAR_FHS": ("HAR", "FHS"),
                  "LightGBM_FHS": ("LightGBM", "FHS"), "GJR_FHS": ("GJR_sig", "FHS")}
    for name, (col, kind) in tail_specs.items():
        for k in ALPHAS:
            P[f"VaR_{name}_{k}"] = np.nan; P[f"ES_{name}_{k}"] = np.nan
        for a, g in P.groupby("asset"):
            sig = g[col].to_numpy(float) if col == "GJR_sig" else np.sqrt(np.exp(g[col].to_numpy(float)))
            out = filtered_tail(sig, g["r_next"].to_numpy(float), kind)
            for k in ALPHAS:
                P.loc[g.index, f"VaR_{name}_{k}"] = out[k][0]
                P.loc[g.index, f"ES_{name}_{k}"] = out[k][1]
    for k, al in ALPHAS.items():                          # WM's own Student-t head
        nu, s = P["WM_df"].to_numpy(), P["WM_scale"].to_numpy()
        q = stats.t.ppf(al, nu)
        P[f"VaR_WM_studentt_{k}"] = s * q
        P[f"ES_WM_studentt_{k}"] = -s * (nu + q ** 2) / (nu - 1) * stats.t.pdf(q, nu) / al
    P["eval"] = P["target_date"] >= EVAL_FROM
    P["breadth"] = P["asset"].isin(bv.BREADTH_ASSETS)
    P.to_parquet(VAL / "v1v3_predictions.parquet")
    meta = {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cut_origin": str(CUT_ORIGIN.date()), "cut_label": str(CUT_LABEL.date()), "eval_from": str(EVAL_FROM.date()),
            "baseline_fit": tinfo, "frozen_wm": wman, "legacy_bias": float(res.mean()), "legacy_jensen": 0.5 * float(res.var()),
            "legacy_sha256": sha(dfz.LEGACY_B), "eval_rows": int(P["eval"].sum()),
            "eval_target_range": [str(P.loc[P["eval"], "target_date"].min().date()), str(P.loc[P["eval"], "target_date"].max().date())]}
    (VAL / "v1v3_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))
    print(f"done [{time.time() - t0:.0f}s]")


# --------------------------------------------------------------------------- #
def vol_scores(P, models, ref="HAR"):
    P = P[np.isfinite(P[models + ["rv_next"]]).all(axis=1)].copy()
    for m in models:
        P[f"q_{m}"] = qlike(P["rv_next"].to_numpy(), np.exp(P[m].to_numpy()))
    qd = pd.DataFrame({m: P.groupby("target_date")[f"q_{m}"].mean() for m in models}).sort_index()
    qs = pd.DataFrame({m: P.groupby("target_date")[f"q_{m}"].sum() for m in models}).sort_index()
    T = len(qd)
    out = {}
    ref_asset = P.groupby("asset")[f"q_{ref}"].mean()
    for m in models:
        q = float(P[f"q_{m}"].mean())
        r = {"QLIKE": q, "pct_vs_ref": q / float(P[f"q_{ref}"].mean()) - 1,
             "assets_beating_ref": int((P.groupby("asset")[f"q_{m}"].mean() < ref_asset).sum()),
             "dates_beating_ref": int((qd[m] < qd[ref]).sum()), "DM_p": None, "ci95_pct_vs_ref": None}
        if m != ref and T >= 20:
            r["DM_p"] = hac_mean_test((qd[m] - qd[ref]).to_numpy(), nw_lag(T))[1]
            r["ci95_pct_vs_ref"] = list(block_bootstrap_ratio_ci(qs[m].to_numpy(), qs[ref].to_numpy(), block=5, reps=4000))
        out[m] = r
    if T >= 20:
        from arch.bootstrap import MCS
        mcs = MCS(qd, size=0.10, reps=2000, block_size=5, method="R", seed=0)
        mcs.compute()
        inc = set(mcs.included)
        for m in models:
            out[m]["in_MCS90"] = m in inc
    for i, m in enumerate(sorted(models, key=lambda x: out[x]["QLIKE"])):
        out[m]["rank"] = i + 1
    monthly = {str(p): {m: float(g[f"q_{m}"].mean() / g[f"q_{ref}"].mean() - 1) for m in models}
               for p, g in P.groupby(P["target_date"].dt.to_period("M"))}
    monthly_n = {str(p): int(g["target_date"].nunique()) for p, g in P.groupby(P["target_date"].dt.to_period("M"))}
    per_asset = {a: {m: float(g[f"q_{m}"].mean() / g[f"q_{ref}"].mean() - 1) for m in models} for a, g in P.groupby("asset")}
    return {"models": out, "monthly_pct_vs_ref": monthly, "monthly_n_dates": monthly_n, "per_asset_pct_vs_ref": per_asset,
            "meta": {"n_rows": int(len(P)), "n_dates": int(T), "n_assets": int(P["asset"].nunique()),
                     "target_range": [str(P["target_date"].min().date()), str(P["target_date"].max().date())], "ref": ref}}


def tail_scores(P, names, ref="GJR_FHS"):
    out = {}
    for k, al in ALPHAS.items():
        cols = [c for n in names for c in (f"VaR_{n}_{k}", f"ES_{n}_{k}")]
        Q = P[np.isfinite(P[cols + ["r_next"]]).all(axis=1)].copy()
        r = Q["r_next"].to_numpy()
        res = {}
        for n in names:
            Q[f"fz_{n}"] = fz0_loss(r, Q[f"VaR_{n}_{k}"].to_numpy(), Q[f"ES_{n}_{k}"].to_numpy(), al)
        fd = pd.DataFrame({n: Q.groupby("target_date")[f"fz_{n}"].mean() for n in names}).sort_index()
        T = len(fd)
        for n in names:
            hits = (r <= Q[f"VaR_{n}_{k}"].to_numpy())
            per = {}
            for a, g in Q.groupby("asset"):
                h = (g["r_next"].to_numpy() <= g[f"VaR_{n}_{k}"].to_numpy()).astype(float)
                per[a] = {"viol": int(h.sum()), "n": int(len(h)), "kupiec_p": float(kupiec(h, al)[0]),
                          "cc_p": float(christoffersen_cc(h, al))}
            rec = {"FZ0": float(Q[f"fz_{n}"].mean()), "violation_rate": float(hits.mean()), "n": int(len(Q)),
                   "assets_pass_kupiec": int(sum(v["kupiec_p"] > 0.05 for v in per.values())),
                   "assets_pass_cc": int(sum(v["cc_p"] > 0.05 for v in per.values())),
                   "n_assets": len(per), "per_asset": per, "delta_vs_ref": float(Q[f"fz_{n}"].mean() - Q[f"fz_{ref}"].mean()),
                   "DM_p": None}
            if n != ref and T >= 20:
                rec["DM_p"] = hac_mean_test((fd[n] - fd[ref]).to_numpy(), nw_lag(T))[1]
            # pooled Kupiec over all asset-days (more power, ignores cross-asset dependence)
            rec["pooled_kupiec_p"] = float(kupiec(hits.astype(float), al)[0])
            res[n] = rec
        out[k] = {"models": res, "meta": {"n_rows": int(len(Q)), "n_dates": int(T), "alpha": al, "ref": ref}}
    return out


def score():
    P = pd.read_parquet(VAL / "v1v3_predictions.parquet")
    meta = json.loads((VAL / "v1v3_meta.json").read_text())
    E = P[P["eval"]]
    assert (E["target_date"] > pd.Timestamp(meta["cut_label"])).all()
    assert (E["target_date"] > pd.Timestamp(meta["frozen_wm"]["train_label_end_max"])).all()
    # Date-level tests align assets by the trading day each bar covers: Yahoo FX bars are
    # labelled one day early in UK summer time (meridian.data.fx_true_day, audit D4).
    # The label-date version is kept as a sensitivity check.
    from meridian.data import FX, fx_true_day
    lab_core = E[~E["breadth"]]
    E = E.copy()
    fxm = E["asset"].isin(list(FX))
    E.loc[fxm, "target_date"] = fx_true_day(E.loc[fxm, "target_date"])
    core, br = E[~E["breadth"]], E[E["breadth"]]
    tails = ["Meridian_B_EVT", "Meridian_B_FHS", "MeridianWM_FHS", "WM_studentt", "HAR_FHS", "LightGBM_FHS", "GJR_FHS"]
    res = {"meta": meta, "date_alignment": "FX target dates mapped to the London trading day (fx_true_day)",
           "vol_core_label_dates": vol_scores(lab_core, VOL_MODELS),
           "vol_core": vol_scores(core, VOL_MODELS),
           "vol_breadth": vol_scores(br, VOL_MODELS) if len(br) else None,
           "tail_core": tail_scores(core, tails),
           "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (VAL / "v1v3_score.json").write_text(json.dumps(res, indent=2, default=float))
    v = res["vol_core"]
    for m in sorted(v["models"], key=lambda x: v["models"][x]["QLIKE"]):
        r = v["models"][m]
        print(f"{m:22} {r['QLIKE']:.4f} {100*r['pct_vs_ref']:+6.1f}% CI {r['ci95_pct_vs_ref']} p={r['DM_p']} MCS={r.get('in_MCS90')}")
    for k, t in res["tail_core"].items():
        print("tail", k, {n: (round(x["FZ0"], 4), round(x["violation_rate"], 4), x["assets_pass_kupiec"], x["DM_p"]) for n, x in t["models"].items()})


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
