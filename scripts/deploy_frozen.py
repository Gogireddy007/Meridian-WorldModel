"""Deploy the already-trained (frozen) Meridian models on data AFTER their training period,
with NO further training, and benchmark them against other models on the same rows.

    python scripts/deploy_data.py       # 1. data snapshot  -> results/deploy/data/
    python scripts/deploy_frozen.py     # 2. forecasts      -> results/deploy/predictions_{A,B}.parquet
    python scripts/score_deploy.py      # 3. scores         -> results/deploy/deploy_score.json, DEPLOYMENT.md
    python scripts/audit_deploy.py      # 4. audit          -> results/deploy/audit_report.json

Two frozen Meridian snapshots exist in the repository. Neither is trained, fine-tuned or
recalibrated here: their weights are loaded, their hashes checked, and they are only run
forward on new inputs.

  A  forward-test Meridian M0 (results/forward/frozen/, 5 seeds, the audited configuration).
     Trained on origins through 2026-10-29 (labels through 2026-10-30), frozen 2026-11-01.
     Out-of-sample targets: 2026-11-02 onward. October 2026 is INSIDE its training data,
     so it is never scored on October.
  B  legacy Meridian (results/meridian_final.pt, written by scripts/fit_final.py before the
     audit: 11 input features, MSE head = E[log RV], single seed). Its training cut-off is
     not recorded; the file was first committed on 2026-09-08, so it cannot contain data
     after that date. Out-of-sample targets used: October 2026 and November 2026.

Comparison models are fit ONCE, on data that ends at the same cut-off as the Meridian
snapshot they are compared with (for B: the 2026-09-08 upper bound, which can only favour
the comparison models), and are then also run forward with frozen parameters. GARCH/EWMA
filters and HAR features update with new returns as part of making a forecast; that is
deployment, not training. No model sees any row whose target is in the evaluation window.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import meridian.data as mdata
from meridian.features import realized_variance
from meridian.model import Meridian, MeridianConfig
from meridian.windows import FEATURES
import scripts.benchmark_vol as bv

ROOT = Path(__file__).resolve().parent.parent
DEP = ROOT / "results" / "deploy"
DATA = DEP / "data"
FROZEN_A = ROOT / "results" / "forward" / "frozen"
LEGACY_B = ROOT / "results" / "meridian_final.pt"
EPS = 1e-12

SNAPSHOTS = {
    "A": {"name": "Meridian M0 forward-frozen (audited config, 5 seeds)",
          "train_origin_max": "2026-10-29", "train_label_end_max": "2026-10-30",
          "eval_target_from": "2026-11-02",
          "windows": {"Nov-2026": ["2026-11-01", "2026-11-30"]}},
    "B": {"name": "Meridian legacy final (pre-audit config, 1 seed)",
          "train_origin_max": "2026-09-07", "train_label_end_max": "2026-09-08",
          "eval_target_from": "2026-10-01",
          "windows": {"Oct-2026": ["2026-10-01", "2026-10-31"], "Nov-2026": ["2026-11-01", "2026-11-30"],
                      "Oct+Nov-2026": ["2026-10-01", "2026-11-30"]}},
}
LEGACY_FEATURES = FEATURES[:11]          # checkpoint has 11 inputs; see audit_deploy.check_legacy_features


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def use_snapshot():
    mdata.DATA_DIR = DATA


# --------------------------------------------------------------------------- #
# Frozen Meridian inference
# --------------------------------------------------------------------------- #
def load_A():
    man = json.loads((FROZEN_A / "manifest.json").read_text())
    for n, h in man["sha256"].items():
        if sha(FROZEN_A / n) != h:
            sys.exit(f"frozen file {n} changed since freeze — refusing to deploy")
    sc = np.load(FROZEN_A / "scaler.npz")
    models = []
    for s in range(5):
        m = Meridian(bv.variant_cfg("M0", 9000 + s))
        m.load_state_dict(torch.load(FROZEN_A / f"meridian_M0_seed{s}.pt", map_location="cpu"))
        m.eval()
        models.append(m)
    har = json.loads((FROZEN_A / "har.json").read_text())
    return man, sc["mean"], sc["std"], har, models


def load_B():
    ck = torch.load(LEGACY_B, map_location="cpu", weights_only=False)
    cfg = MeridianConfig(**ck["cfg"])                 # n_features=11, loss_mode="mse"
    m = Meridian(cfg)
    m.load_state_dict(ck["state"], strict=True)
    m.eval()
    return m, np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32)


def windows_for(frame, feats, mean, std, L=bv.L_NEURAL):
    """Context windows ending at every origin t whose L-day feature history is finite.
    Uses rows <= t only."""
    F = frame[feats].to_numpy(np.float32)
    ok = np.isfinite(F).all(1)
    Xs = (F - mean) / std
    idx = [i for i in range(L - 1, len(F)) if ok[i - L + 1:i + 1].all()]
    if not idx:
        return pd.DatetimeIndex([]), np.zeros((0, L, len(feats)), np.float32)
    return frame.index[idx], np.stack([Xs[i - L + 1:i + 1] for i in idx]).astype(np.float32)


@torch.no_grad()
def run(models, X):
    t = torch.tensor(X)
    return np.mean([m.forward(t)["vol"].numpy() for m in models], 0)


# --------------------------------------------------------------------------- #
# Comparison models: fit once on pre-cutoff rows, run forward frozen
# --------------------------------------------------------------------------- #
def fit_and_forecast_baselines(R, Wlog, Wlev, d, cutoff, test_start, te_m):
    tr_core = bv.train_rows(R, cutoff, test_start, "y", bv.CORE_ASSETS)
    te = R[te_m].reset_index(drop=True)
    trc = R[tr_core]
    out = {}
    out["HAR_pooled"] = bv.ols_apply(te, bv.HAR3, bv.ols(trc, bv.HAR3))
    out["Ridge"] = bv.fit_ridge(trc, te)
    out["RandomForest"] = bv.fit_rf(trc, te)
    out["LightGBM"] = bv.fit_lgbm(trc, te)
    out["TimeMixer"] = bv.fit_timemixer(Wlog[tr_core], Wlev[tr_core], trc["y"].to_numpy(), Wlog[te_m], Wlev[te_m])
    cols = {k: np.full(len(te), np.nan) for k in ["RW", "MA22", "HAR", "LHAR", "SHAR", "HARX", "EWMA", *bv.GARCH_SPECS]}
    fits = {}
    for a in te["asset"].unique():
        ai = (te["asset"] == a).to_numpy()
        tea = te[ai]
        tra = R[bv.train_rows(R, cutoff, test_start, "y", [a])]
        c = bv.offset_model(tra, "har_d"); cols["RW"][ai] = tea["har_d"] + c if c is not None else np.nan
        c = bv.offset_model(tra, "har_m"); cols["MA22"][ai] = tea["har_m"] + c if c is not None else np.nan
        hf = bv.ols(tra, bv.HAR3); fits[a] = hf
        cols["HAR"][ai] = bv.ols_apply(tea, bv.HAR3, hf)
        lev = bv.HAR3 + ["lev_d", "lev_w", "lev_m"]
        cols["LHAR"][ai] = bv.ols_apply(tea, lev, bv.ols(tra, lev))
        shar = ["har_d_up", "har_d_dn", "down", "har_w", "har_m"]
        cols["SHAR"][ai] = bv.ols_apply(tea, shar, bv.ols(tra, shar))
        cols["HARX"][ai] = bv.ols_apply(tea, bv.HAR3 + ["vix"], bv.ols(tra, bv.HAR3 + ["vix"]))
        rvf = realized_variance(d["prices"][a])
        rv, ret = rvf["rv"], rvf["ret"]
        ew = pd.Series(bv.ewma_next(rv.to_numpy()), index=rv.index)
        trd = rv.index < cutoff
        rv_next_tr = rv.shift(-1)[trd].to_numpy()
        tdates = tea["date"].to_numpy()
        cE = np.nanmean(rv_next_tr) / np.nanmean(ew[trd].to_numpy())
        cols["EWMA"][ai] = np.log(np.clip(cE * ew.reindex(tdates).to_numpy(), EPS, None))
        for kind in bv.GARCH_SPECS:
            nxt = bv.garch_next(ret.to_numpy(), np.asarray(trd), kind)
            if nxt is None:
                continue
            s = pd.Series(nxt, index=ret.index)
            cG = np.nanmean(rv_next_tr) / np.nanmean(s[trd].to_numpy())
            cols[kind][ai] = np.log(np.clip(cG * s.reindex(tdates).to_numpy(), EPS, None))
    out.update(cols)
    train_info = {"n_train_rows_core": int(tr_core.sum()),
                  "train_origin_max": str(pd.Timestamp(R.loc[tr_core, "date"].max()).date()),
                  "train_label_end_max": str(pd.Timestamp(R.loc[tr_core, "end_1"].max()).date())}
    return te, out, fits, train_info


def build_eval_frame(R, snap):
    """Rows whose 1-day TARGET date falls in the evaluation span of a snapshot."""
    s = SNAPSHOTS[snap]
    lo = pd.Timestamp(s["eval_target_from"])
    hi = max(pd.Timestamp(w[1]) for w in s["windows"].values())
    return ((R["end_1"] >= lo) & (R["end_1"] <= hi)).to_numpy()


def main():
    t0 = time.time()
    use_snapshot()
    R, Wlog, Wlev, d, frames, _ = bv.build()
    meta = {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True, cwd=ROOT).stdout.strip(),
            "data_manifest_sha256": sha(DEP / "data_manifest.json"),
            "data_last_target": str(pd.Timestamp(R["end_1"].max()).date()), "snapshots": {}}

    manA, meanA, stdA, harA, modelsA = load_A()
    mB, meanB, stdB = load_B()
    meta["weights_sha256"] = {"A": manA["sha256"], "B": {LEGACY_B.name: sha(LEGACY_B)}}

    # Meridian forecasts for every possible origin (filtered to eval rows below)
    merA, merB = [], []
    for a, f in frames.items():
        dts, X = windows_for(f, FEATURES, meanA, stdA)
        if len(dts):
            merA.append(pd.DataFrame({"asset": a, "date": dts, "Meridian_M0": run(modelsA, X)}))
        dts, X = windows_for(f, LEGACY_FEATURES, meanB, stdB)
        if len(dts):
            merB.append(pd.DataFrame({"asset": a, "date": dts, "Meridian_legacy_raw": run([mB], X)}))
    merA, merB = pd.concat(merA), pd.concat(merB)

    for snap in ("A", "B"):
        s = SNAPSHOTS[snap]
        cutoff = pd.Timestamp(s["train_origin_max"]) + pd.Timedelta(days=1)       # date < cutoff
        test_start = pd.Timestamp(s["train_label_end_max"]) + pd.Timedelta(days=1)  # label end < test_start
        te_m = build_eval_frame(R, snap)
        te, base, fits, tinfo = fit_and_forecast_baselines(R, Wlog, Wlev, d, cutoff, test_start, te_m)
        P = te[["asset", "date", "end_1", "y", "rv_next", "har_d", "vix"]].rename(columns={"end_1": "target_date"}).copy()
        for k, v in base.items():
            P[k] = v
        if snap == "A":
            P = P.merge(merA, on=["asset", "date"], how="left")
            hf = np.full(len(P), np.nan)                                # the frozen per-asset HAR
            te_h = te.set_index(["asset", "date"])[bv.HAR3]
            X = te_h.reindex(pd.MultiIndex.from_frame(P[["asset", "date"]])).to_numpy()
            for a, h in harA.items():
                m = (P["asset"] == a).to_numpy()
                b = np.asarray(h["beta"])
                hf[m] = b[0] + X[m] @ b[1:] + h["jensen"]
            P["HAR_frozen"] = hf
            # Real-time vintage: FRED publishes the VIX / yield close with a delay, so a forecast
            # made right after the close sees macro inputs only through t-1 (carried forward).
            # This is what the live forward log (results/forward/forecast_log.csv) used.
            rt = np.full(len(P), np.nan)
            for i, (a, t) in enumerate(zip(P["asset"], P["date"])):
                o = d["prices"][a]
                f = bv.build_asset_frame(o[o.index <= t], d["macro"][d["macro"].index < t])
                _, X = windows_for(f, FEATURES, meanA, stdA)
                rt[i] = float(run(modelsA, X[-1:])[0])
            P["Meridian_M0_rtvix"] = rt
        else:
            P = P.merge(merB, on=["asset", "date"], how="left")
            # Jensen term for the MSE (E[log RV]) head: 0.5*Var(residual) on PRE-cutoff rows only.
            pre = R[(R["date"] < cutoff) & (R["end_1"] < test_start) & R["asset"].isin(bv.CORE_ASSETS)]
            pre = pre.merge(merB, on=["asset", "date"], how="inner")
            pre = pre[pre["date"] >= pre["date"].max() - pd.Timedelta(days=730)]
            res = pre["y"] - pre["Meridian_legacy_raw"]
            jb = 0.5 * float(np.var(res))
            P["Meridian_legacy"] = P["Meridian_legacy_raw"] + jb
            # Secondary: also remove the head's mean bias (it was trained on uncleaned FX bars,
            # see audit check legacy_provenance). One offset from pre-cutoff rows; weights untouched.
            P["Meridian_legacy_cal"] = P["Meridian_legacy_raw"] + float(res.mean()) + jb
            tinfo["legacy_bias"] = float(res.mean())
            tinfo["legacy_jensen"] = jb
            tinfo["legacy_jensen_rows"] = [str(pre["date"].min().date()), str(pre["date"].max().date()), int(len(pre))]
        P["snapshot"] = snap
        P["breadth"] = P["asset"].isin(bv.BREADTH_ASSETS)
        P.to_parquet(DEP / f"predictions_{snap}.parquet")
        meta["snapshots"][snap] = {**s, "baseline_fit": tinfo, "rows": int(len(P)),
                                   "target_range": [str(P["target_date"].min().date()), str(P["target_date"].max().date())]}
        print(f"  snapshot {snap}: {len(P)} rows, targets {meta['snapshots'][snap]['target_range']}  [{time.time()-t0:.0f}s]", flush=True)
    (DEP / "deploy_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["snapshots"], indent=2, default=str))


if __name__ == "__main__":
    main()
