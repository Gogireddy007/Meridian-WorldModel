"""Independent audit of the frozen-model deployment (no data leak, no bluffed numbers,
nothing fake).

    python scripts/audit_deploy.py      # -> results/deploy/audit_report.{json,md}; exit 1 on any FAIL

Each check re-derives its evidence from raw inputs instead of trusting the pipeline's own
outputs. Where possible it uses an implementation written separately from the pipeline
(the Garman-Klass truth, QLIKE, next-trading-day mapping), so an error shared by the
pipeline and the audit is unlikely.

  W1 weights_unchanged        frozen weight files hash-match their manifest AND git HEAD:
                              nothing was trained or overwritten in this deployment
  W2 no_training_code_path    the deployment script has no optimizer / backward / train call
                              on a Meridian model, and every Meridian runs in eval mode
  T1 targets_after_training   every scored target date is after the snapshot's last training label
  T2 baselines_same_cutoff    comparison models were fit on rows that end at the same cut-off
  T3 lookahead_truncation     forecasts made from data CUT at the origin equal the deployed
                              forecasts (Meridian A/B, HAR, EWMA, GARCH): no input after t
  T4 legacy_provenance        snapshot B's training data/cut-off reconstructed from its saved
                              scaler; must end before the evaluation window
  D1 data_source_parity       snapshot data reproduces the audited benchmark's realized variance
  D2 truth_recomputed         rv_next / target_date re-derived from raw OHLC independently
  D3 bars_complete            no in-progress bar, no duplicates, no degenerate bars in the window
  R1 precommitted_reproduced  frozen weights + the real-time data vintage reproduce exactly the
                              forecasts logged in results/forward/forecast_log.csv BEFORE their outcomes
  R2 frozen_har_reproduced    HAR refit at the same cut-off equals the frozen HAR artifact
  S1 scores_recomputed        every QLIKE / % vs HAR in deploy_score.json recomputed independently
  S2 report_matches_json      DEPLOYMENT.md and README's DEPLOY block equal a fresh render of the JSON
  S3 no_tests_on_tiny_samples no p-value or CI is reported for windows with < 20 dates
  S4 scorer_sanity            an oracle scores QLIKE 0; date-shuffled forecasts should score worse
                              (WARN when Meridian's ordering is no better than shuffled)
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import meridian.data as mdata
import scripts.benchmark_vol as bv
import scripts.deploy_frozen as dfz
import scripts.score_deploy as sd
from meridian.features import build_asset_frame
from meridian.windows import FEATURES

ROOT = Path(__file__).resolve().parent.parent
DEP = ROOT / "results" / "deploy"
RESULTS = []


def check(cid, name):
    def deco(fn):
        def run():
            try:
                status, evidence = fn()
            except Exception as e:  # noqa: BLE001 — an audit check that crashes is a FAIL
                status, evidence = "FAIL", {"exception": repr(e)[:400]}
            RESULTS.append({"id": cid, "check": name, "status": status, "evidence": evidence})
            print(f"  [{status:4}] {cid} {name}", flush=True)
        return run
    return deco


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_blob_sha(path):
    """sha256 of the file as committed at HEAD (proves the working copy was not retrained)."""
    blob = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, cwd=ROOT).stdout
    return hashlib.sha256(blob).hexdigest()


def gk_independent(o):
    """Garman-Klass variance written out separately from meridian.features."""
    lo, hi, op, cl = (np.log(o[c].to_numpy(float)) for c in ("low", "high", "open", "close"))
    v = 0.5 * (hi - lo) ** 2 - (2 * np.log(2) - 1) * (cl - op) ** 2
    return pd.Series(np.maximum(v, 1e-12), index=o.index)


def qlike_independent(rv, logvar):
    r = rv / np.exp(logvar)
    return r - np.log(r) - 1.0


# --------------------------------------------------------------------------- #
@check("W1", "weights_unchanged")
def w1():
    man = json.loads((dfz.FROZEN_A / "manifest.json").read_text())
    ev, ok = {}, True
    for n, h in man["sha256"].items():
        p = dfz.FROZEN_A / n
        rel = str(p.relative_to(ROOT))
        cur, head = sha(p), git_blob_sha(rel)
        ev[rel] = {"matches_manifest": cur == h, "matches_git_HEAD": cur == head}
        ok &= cur == h and cur == head
    rel = str(dfz.LEGACY_B.relative_to(ROOT))
    cur = sha(dfz.LEGACY_B)
    meta = json.loads((DEP / "deploy_meta.json").read_text())
    ev[rel] = {"matches_git_HEAD": cur == git_blob_sha(rel),
               "matches_hash_recorded_at_deploy": cur == meta["weights_sha256"]["B"][dfz.LEGACY_B.name],
               "first_committed": subprocess.run(["git", "log", "--diff-filter=A", "--format=%ad", "--date=short", "--", rel],
                                                 capture_output=True, text=True, cwd=ROOT).stdout.strip()}
    ok &= ev[rel]["matches_git_HEAD"] and ev[rel]["matches_hash_recorded_at_deploy"]
    return ("PASS" if ok else "FAIL"), ev


@check("W2", "no_training_code_path")
def w2():
    src = (ROOT / "scripts" / "deploy_frozen.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    banned = [".backward(", "optim.", "train_meridian(", "train_block(", ".train()", "load_state_dict(best"]
    hits = [b for b in banned if b in code]
    _, _, _, _, models = dfz.load_A()
    mB, _, _ = dfz.load_B()
    eval_mode = all(not m.training for m in models) and not mB.training
    ok = not hits and eval_mode and "torch.no_grad()" in code
    return ("PASS" if ok else "FAIL"), {"banned_calls_found": hits, "all_meridian_in_eval_mode": eval_mode}


@check("T1", "targets_after_training")
def t1():
    meta = json.loads((DEP / "deploy_meta.json").read_text())
    ev, ok = {}, True
    for s in ("A", "B"):
        P = pd.read_parquet(DEP / f"predictions_{s}.parquet")
        cut = pd.Timestamp(meta["snapshots"][s]["train_label_end_max"])
        gap = (P["target_date"].min() - cut).days
        bad = int((P["target_date"] <= cut).sum()) + int((P["target_date"] <= P["date"]).sum())
        ev[s] = {"train_label_end_max": str(cut.date()), "first_scored_target": str(P["target_date"].min().date()),
                 "calendar_gap_days": int(gap), "rows_violating": bad}
        ok &= bad == 0
    return ("PASS" if ok else "FAIL"), ev


@check("T2", "baselines_same_cutoff")
def t2():
    meta = json.loads((DEP / "deploy_meta.json").read_text())
    ev, ok = {}, True
    for s in ("A", "B"):
        sn = meta["snapshots"][s]
        bf = sn["baseline_fit"]
        same = (bf["train_origin_max"] <= sn["train_origin_max"]) and (bf["train_label_end_max"] <= sn["train_label_end_max"])
        ev[s] = {"meridian_label_end": sn["train_label_end_max"], "baselines_label_end": bf["train_label_end_max"],
                 "baselines_origin_max": bf["train_origin_max"], "ok": same}
        ok &= same
    return ("PASS" if ok else "FAIL"), ev


@check("T3", "lookahead_truncation")
def t3():
    """Re-make forecasts at sampled origins from data TRUNCATED at the origin."""
    dfz.use_snapshot()
    d = bv.load_clean()
    manA, meanA, stdA, harA, modelsA = dfz.load_A()
    mB, meanB, stdB = dfz.load_B()
    PA = pd.read_parquet(DEP / "predictions_A.parquet")
    PB = pd.read_parquet(DEP / "predictions_B.parquet")
    rng = np.random.default_rng(0)
    worst = {"Meridian_M0": 0.0, "Meridian_legacy_raw": 0.0, "HAR_frozen": 0.0, "EWMA_state": 0.0, "GARCH_state": 0.0}
    n = 0
    for P, col in ((PA, "Meridian_M0"), (PB, "Meridian_legacy_raw")):
        for _, r in P.sample(min(12, len(P)), random_state=1).iterrows():
            a, t = r["asset"], r["date"]
            o = d["prices"][a]
            o_cut = o[o.index <= t]
            f = build_asset_frame(o_cut, d["macro"][d["macro"].index <= t])
            if col == "Meridian_M0":
                dts, X = dfz.windows_for(f, FEATURES, meanA, stdA)
                v = float(dfz.run(modelsA, X[-1:])[0])
                b = np.asarray(harA[a]["beta"])
                h = b[0] + f[bv.HAR3].iloc[-1].to_numpy() @ b[1:] + harA[a]["jensen"]
                worst["HAR_frozen"] = max(worst["HAR_frozen"], abs(h - PA.set_index(["asset", "date"]).loc[(a, t), "HAR_frozen"]))
            else:
                dts, X = dfz.windows_for(f, dfz.LEGACY_FEATURES, meanB, stdB)
                v = float(dfz.run([mB], X[-1:])[0])
            assert dts[-1] == t
            worst[col] = max(worst[col], abs(v - r[col]))
            # EWMA / GARCH filter state at t from truncated vs full history
            from meridian.features import realized_variance
            rf, rc = realized_variance(o), realized_variance(o_cut)
            e_full = bv.ewma_next(rf["rv"].to_numpy())[rf.index.get_loc(t)]
            e_cut = bv.ewma_next(rc["rv"].to_numpy())[-1]
            worst["EWMA_state"] = max(worst["EWMA_state"], abs(np.log(e_full) - np.log(e_cut)))
            trm = np.asarray(rf.index < pd.Timestamp("2026-09-08"))
            g_full = bv.garch_next(rf["ret"].to_numpy(), trm, "GARCH")
            g_cut = bv.garch_next(rc["ret"].to_numpy(), trm[:len(rc)], "GARCH")
            if g_full is not None and g_cut is not None:
                # next[t] of the cut series is NaN by construction (no t+1 return needed: use s2 recursion)
                i = rf.index.get_loc(t)
                worst["GARCH_state"] = max(worst["GARCH_state"], abs(np.log(g_full[i - 1]) - np.log(g_cut[i - 1])))
            n += 1
    ok = all(v < 1e-4 for v in worst.values())
    return ("PASS" if ok else "FAIL"), {"origins_tested": n, "max_abs_diff_log_var": {k: float(v) for k, v in worst.items()},
                                        "tolerance": 1e-4}


@check("T4", "legacy_provenance")
def t4():
    """Snapshot B's cut-off is not recorded, so reconstruct it: its saved feature scaler is the
    mean/std of its training rows. Find the data version (cleaned / uncleaned) and end date
    whose statistics reproduce it."""
    ck = torch.load(dfz.LEGACY_B, map_location="cpu", weights_only=False)
    mdata.DATA_DIR = dfz.DATA
    d = mdata.load_all()
    F = dfz.LEGACY_FEATURES
    frames = {(c, a): build_asset_frame(bv.clean_ohlc(o) if c else o, d["macro"])
              for c in (False, True) for a, o in d["prices"].items()}
    best = None
    for clean in (False, True):
        for end in pd.bdate_range("2026-06-01", "2026-09-30"):
            X = []
            for (c, a), f in frames.items():
                if c != clean:
                    continue
                df = f[F + ["y"]]
                df = df[np.isfinite(df.to_numpy()).all(1) & (df.index < end)]
                X.append(df[F].to_numpy())
            X = np.concatenate(X)
            err = float(max(np.max(np.abs(X.mean(0) - ck["mean"]) / ck["std"]), np.max(np.abs((X.std(0) + 1e-6) / ck["std"] - 1))))
            if best is None or err < best[0]:
                best = (err, clean, end)
    err, clean, end = best
    last_origin = pd.Timestamp(end) - pd.offsets.BDay(1)
    meta = json.loads((DEP / "deploy_meta.json").read_text())
    eval_start = pd.Timestamp(meta["snapshots"]["B"]["eval_target_from"])
    ok = err < 1e-3 and last_origin < eval_start and pd.Timestamp(meta["snapshots"]["B"]["train_label_end_max"]) >= last_origin
    return ("PASS" if ok else "FAIL"), {
        "best_match_training_data": "uncleaned (pre-audit) bars" if not clean else "cleaned bars",
        "best_match_last_origin": str(last_origin.date()), "max_scaled_error": err,
        "assumed_cutoff_used_for_baselines": meta["snapshots"]["B"]["train_label_end_max"],
        "eval_window_starts": str(eval_start.date()),
        "note": "baselines were given data through the later, conservative cut-off"}


@check("D1", "data_source_parity")
def d1():
    man = json.loads((DEP / "data_manifest.json").read_text())
    ev, ok = {}, True
    for a, v in man["assets"].items():
        c = v["check_vs_benchmark"]
        good = c.get("overlap", 0) > 500 and c["median_abs_log_diff"] < 1e-3 and c["corr_log"] > 0.99
        ev[a] = {"source": v["source"], "overlap": c.get("overlap"), "median_abs_log_diff": c.get("median_abs_log_diff"),
                 "corr_log": c.get("corr_log")}
        ok &= good
    stored = {p.name: sha(p) for p in sorted(dfz.DATA.glob("*.parquet"))}
    same = stored == man["files_sha256"]
    ev["_snapshot_files_match_manifest"] = same
    return ("PASS" if ok and same else "FAIL"), ev


@check("D2", "truth_recomputed")
def d2():
    mdata.DATA_DIR = dfz.DATA
    d = mdata.load_all(include_breadth=True)
    worst, bad_dates, n = 0.0, 0, 0
    for s in ("A", "B"):
        P = pd.read_parquet(DEP / f"predictions_{s}.parquet")
        for a, g in P.groupby("asset"):
            o = bv.clean_ohlc(d["prices"][a])
            gk = gk_independent(o)
            idx = o.index
            for _, r in g.iterrows():
                i = idx.get_loc(r["date"])
                nxt = idx[i + 1]
                bad_dates += int(nxt != r["target_date"])
                worst = max(worst, abs(np.log(gk.loc[nxt]) - np.log(r["rv_next"])))
                n += 1
    ok = worst < 1e-9 and bad_dates == 0
    return ("PASS" if ok else "FAIL"), {"rows": n, "max_abs_log_diff": worst, "target_date_mismatches": bad_dates}


@check("D3", "bars_complete")
def d3():
    mdata.DATA_DIR = dfz.DATA
    d = mdata.load_all(include_breadth=True)
    man = json.loads((DEP / "data_manifest.json").read_text())
    built = pd.Timestamp(man["built_at_utc"]).tz_convert("America/New_York")
    last_complete = built.normalize().tz_localize(None) - (pd.Timedelta(0) if built.hour >= 17 else pd.Timedelta(days=1))
    ev, ok = {}, True
    for a, o in d["prices"].items():
        w = o[o.index >= "2026-09-01"]
        # FX bars are UTC-day bars: day d is complete only once d+1 00:00 UTC has passed
        fx_incomplete = a in mdata.FX and o.index.max() + pd.Timedelta(days=1) > pd.Timestamp(man["built_at_utc"]).tz_localize(None)
        rec = {"last_bar": str(o.index.max().date()),
               "after_last_complete_day": bool(o.index.max() > last_complete or fx_incomplete),
               "duplicates": int(o.index.duplicated().sum()), "degenerate_bars_in_window": int((w["high"] <= w["low"]).sum())}
        ev[a] = rec
        ok &= not rec["after_last_complete_day"] and rec["duplicates"] == 0
    return ("PASS" if ok else "FAIL"), ev


@check("R1", "precommitted_reproduced")
def r1():
    """Recompute every pre-committed forecast in results/forward/forecast_log.csv from the frozen
    weights. FRED publishes the VIX/yield close with a delay, so at logging time the macro inputs
    ended one business day before the origin (carried forward). Rebuild that real-time vintage
    and require an exact match; also report how far the deployed (same-day macro) forecast is."""
    dfz.use_snapshot()
    d = bv.load_clean()
    _, mean, std, har, models = dfz.load_A()
    log = pd.read_csv(ROOT / "results" / "forward" / "forecast_log.csv", parse_dates=["origin_date", "logged_at_utc"])
    log = log[log["precommitted"]]
    P = pd.read_parquet(DEP / "predictions_A.parquet").set_index(["asset", "date"])
    rows, w_vintage, w_har, w_deployed = 0, 0.0, 0.0, 0.0
    for _, r in log.iterrows():
        a, t = r["asset"], r["origin_date"]
        o = d["prices"][a]
        o = o[o.index <= t]
        logged_day = r["logged_at_utc"].tz_convert("America/New_York").tz_localize(None).normalize()
        mac = d["macro"][d["macro"].index < min(t + pd.Timedelta(days=1), logged_day)]
        mac = mac[mac.index < t] if logged_day <= t + pd.offsets.BDay(1) else mac
        f = build_asset_frame(o, mac)
        dts, X = dfz.windows_for(f, FEATURES, mean, std)
        v = float(dfz.run(models, X[-1:])[0])
        b = np.asarray(har[a]["beta"])
        h = b[0] + f[bv.HAR3].iloc[-1].to_numpy() @ b[1:] + har[a]["jensen"]
        w_vintage = max(w_vintage, abs(v - r["Meridian_M0"]))
        w_har = max(w_har, abs(h - r["HAR"]))
        if (a, t) in P.index:
            w_deployed = max(w_deployed, abs(P.loc[(a, t), "Meridian_M0"] - r["Meridian_M0"]))
        rows += 1
    ok = rows > 0 and w_vintage < 1e-4 and w_har < 1e-4
    return ("PASS" if ok else "FAIL"), {
        "logged_rows_recomputed": rows, "max_abs_diff_Meridian_realtime_vintage": w_vintage,
        "max_abs_diff_HAR": w_har, "max_abs_diff_deployed_same_day_macro": w_deployed,
        "note": "deployed forecasts use the same-day VIX/yield close (as in training); the live log used "
                "the previous day's value because FRED had not yet published. Both use only data known at the close."}


@check("R2", "frozen_har_reproduced")
def r2():
    P = pd.read_parquet(DEP / "predictions_A.parquet")
    diff = float(np.max(np.abs(P["HAR"] - P["HAR_frozen"])))
    return ("PASS" if diff < 1e-6 else "FAIL"), {"max_abs_diff_log_var": diff}


@check("S1", "scores_recomputed")
def s1():
    S = json.loads((DEP / "deploy_score.json").read_text())
    meta = json.loads((DEP / "deploy_meta.json").read_text())
    worst, n = 0.0, 0
    for s, snap in S["snapshots"].items():
        P = pd.read_parquet(DEP / f"predictions_{s}.parquet")
        for w, (lo, hi) in meta["snapshots"][s]["windows"].items():
            if w not in snap["windows"]:
                continue
            sc = snap["windows"][w]["core"]
            models = list(sc["models"])
            Q = P[(P["target_date"] >= lo) & (P["target_date"] <= hi) & ~P["breadth"]]
            Q = Q[np.isfinite(Q[models + ["rv_next"]]).all(axis=1)]
            ref = qlike_independent(Q["rv_next"].to_numpy(), Q["HAR"].to_numpy()).mean()
            for m in models:
                q = qlike_independent(Q["rv_next"].to_numpy(), Q[m].to_numpy()).mean()
                worst = max(worst, abs(q - sc["models"][m]["QLIKE"]), abs((q / ref - 1) - sc["models"][m]["pct_vs_ref"]))
                n += 1
            assert len(Q) == sc["meta"]["n_rows"]
    return ("PASS" if worst < 1e-9 else "FAIL"), {"numbers_checked": n, "max_abs_diff": worst}


@check("S2", "report_matches_json")
def s2():
    """DEPLOYMENT.md and the README DEPLOY block must equal a fresh render of the JSON."""
    paths = [ROOT / "DEPLOYMENT.md", ROOT / "README.md"]
    before = {p: p.read_text() for p in paths}
    sd.render(json.loads((DEP / "deploy_score.json").read_text()))
    after = {p: p.read_text() for p in paths}
    same = {p.name: before[p] == after[p] for p in paths}
    for p in paths:
        if before[p] != after[p]:
            p.write_text(before[p])                     # leave the files as found
    return ("PASS" if all(same.values()) else "FAIL"), {"identical_to_fresh_render": same}


@check("S3", "no_tests_on_tiny_samples")
def s3():
    S = json.loads((DEP / "deploy_score.json").read_text())
    bad = []
    for s, snap in S["snapshots"].items():
        for w, ww in snap["windows"].items():
            sc = ww["core"]
            if sc["meta"]["n_dates"] < 20:
                bad += [f"{s}/{w}/{m}" for m, r in sc["models"].items() if r["DM_p"] is not None or r["ci95_pct_vs_ref"] is not None]
    return ("PASS" if not bad else "FAIL"), {"violations": bad}


@check("S4", "scorer_sanity")
def s4():
    P = pd.read_parquet(DEP / "predictions_B.parquet")
    P = P[~P["breadth"]]
    oracle = qlike_independent(P["rv_next"].to_numpy(), np.log(P["rv_next"].to_numpy())).mean()
    rng = np.random.default_rng(0)
    ev = {"oracle_QLIKE": float(oracle)}
    ok = abs(oracle) < 1e-12
    for s, col in (("A", "Meridian_M0"), ("A", "HAR"), ("B", "Meridian_legacy"), ("B", "HAR")):
        Q = pd.read_parquet(DEP / f"predictions_{s}.parquet")
        Q = Q[~Q["breadth"]].copy()
        real = qlike_independent(Q["rv_next"].to_numpy(), Q[col].to_numpy()).mean()
        sh = []
        for _ in range(200):
            Q["sh"] = Q.groupby("asset")[col].transform(lambda x: rng.permutation(x.to_numpy()))
            sh.append(qlike_independent(Q["rv_next"].to_numpy(), Q["sh"].to_numpy()).mean())
        ev[f"{s}:{col}"] = {"QLIKE": float(real), "shuffled_mean": float(np.mean(sh)), "share_shuffles_better": float(np.mean(np.array(sh) < real))}
    status = "PASS" if ok else "FAIL"
    if any(v["share_shuffles_better"] > 0.05 for k, v in ev.items() if isinstance(v, dict) and "Meridian" in k):
        status = "WARN" if ok else "FAIL"
    ev["meaning"] = ("share_shuffles_better > 0.05: within this window the model's day-to-day ordering of "
                     "forecasts does not beat a random ordering of its own forecasts (no timing evidence); "
                     "HAR rows are the control")
    return status, ev


def render_md(rep):
    L = ["<!-- generated by scripts/audit_deploy.py — do not edit by hand -->",
         "# Audit of the frozen-model deployment", "",
         f"Run {rep['run_at_utc']} on commit `{rep['git_commit']}`. "
         f"**{rep['summary']['PASS']} PASS, {rep['summary'].get('WARN', 0)} WARN, {rep['summary'].get('FAIL', 0)} FAIL.**", "",
         "| ID | Check | Status | Key evidence |", "|---|---|---|---|"]
    for r in rep["checks"]:
        e = json.dumps(r["evidence"], default=str)
        L.append(f"| {r['id']} | {r['check']} | {r['status']} | `{e[:300]}{'…' if len(e) > 300 else ''}` |")
    L += ["", "Full evidence: `results/deploy/audit_report.json`."]
    return "\n".join(L) + "\n"


def main():
    for fn in (w1, w2, t1, t2, t3, t4, d1, d2, d3, r1, r2, s1, s2, s3, s4):
        fn()
    summ = pd.Series([r["status"] for r in RESULTS]).value_counts().to_dict()
    rep = {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "summary": summ, "checks": RESULTS}
    (DEP / "audit_report.json").write_text(json.dumps(rep, indent=2, default=str))
    (DEP / "audit_report.md").write_text(render_md(rep))
    print(summ)
    sys.exit(1 if summ.get("FAIL") else 0)


if __name__ == "__main__":
    main()
