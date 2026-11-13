"""Meridian-WM inside the audited walk-forward harness (PREREGISTRATION.md Amendment 8, V2).

    python scripts/benchmark_wm.py <seed_index>        # one ensemble member -> results/wm/bench_seed<k>.parquet
    python scripts/benchmark_wm.py combine             # -> results/wm/bench_wm.parquet (+ stored models)

Same protocol as scripts/benchmark_vol.py: identical fold grid (date-index based, 1260-day
minimum train, 378-day test blocks, 22-day embargo), training rows purged by 1-day label end,
scaler fit on the fold's train rows only, training on the CORE assets only. The data are the
deployment snapshot truncated to the audited benchmark's data end, so the fold boundaries
match results/benchmark_vol_predictions.parquet (checked in `combine`).

MeridianWM is used exactly as defined in meridian/meridian_wm.py (default WMConfig), trained
with the loop of scripts/run_wm.py (8 epochs, batch 256, AdamW 1e-3, grad clip 1).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import meridian.data as mdata
import scripts.benchmark_vol as bv
from meridian.meridian_wm import MeridianWM, WMConfig
from meridian.windows import FEATURES, build_windows, train_scaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "wm"
SNAP = ROOT / "results" / "deploy" / "data"
BENCH_DATA_END = pd.Timestamp("2026-07-30")          # last row of the audited benchmark's data
EPOCHS, BATCH, LR = 8, 256, 1e-3
N_SEEDS = 3


def load_truncated(end=BENCH_DATA_END):
    """The deployment data snapshot (verified against the benchmark by audit check D1), cut at `end`."""
    mdata.DATA_DIR = SNAP
    raw = mdata.load_all(include_breadth=True)
    raw["prices"] = {a: o[o.index <= end] for a, o in raw["prices"].items()}
    raw["macro"] = raw["macro"][raw["macro"].index <= end]
    return raw


def build(end=BENCH_DATA_END):
    raw = load_truncated(end)
    bv.load_all = lambda **k: raw                    # bv.build() -> load_clean() -> load_all()
    return bv.build()


def fold_grid(R):
    dates = np.array(sorted(R["date"].unique()))
    folds, i = [], bv.MIN_TRAIN_D
    while i < len(dates):
        test_dates = dates[i:i + bv.TEST_D]
        test_start, test_end = dates[i], dates[min(i + bv.TEST_D, len(dates) - 1)]
        cutoff = dates[i - bv.EMBARGO_D]
        te_m = R["date"].isin(set(test_dates)).to_numpy()
        tr_core = bv.train_rows(R, cutoff, test_start, "y", bv.CORE_ASSETS)
        if tr_core.sum() >= 2000 and te_m.sum() >= 100:
            folds.append({"fold": len(folds) + 1, "cutoff": cutoff, "test_start": test_start,
                          "test_end": test_end, "test_dates": set(test_dates)})
        i += bv.TEST_D
    return folds


def wm_windows(asset_win, frames, mean, std):
    out = {}
    for a, (dates, X, y, aux) in asset_win.items():
        w = build_windows(dates, X, y, bv.L_NEURAL, mean, std)
        if w is None:
            continue
        dts = pd.DatetimeIndex(w["dates"])
        w["r_next"] = np.nan_to_num(frames[a]["r_next"].reindex(dts).to_numpy(np.float32))
        w["end_1"] = frames[a].loc[dts, "end_1"].to_numpy()
        w["dates"] = dts
        out[a] = w
    return out


SKIP_NONFINITE = True        # Amendment 8a: skip optimizer steps whose loss / grad norm is non-finite


def train_wm(cfg, tr, threads=None, stats=None):
    if threads:
        torch.set_num_threads(threads)
    m = MeridianWM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    T = {k: torch.tensor(tr[k]) for k in ("ctx", "fut", "y", "r_next")}
    n = len(T["y"])
    g = torch.Generator().manual_seed(cfg.seed)
    m.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss, _ = m.loss({"x_ctx": T["ctx"][b], "x_fut": T["fut"][b], "y": T["y"][b], "r_next": T["r_next"][b]})
            if SKIP_NONFINITE and not torch.isfinite(loss):
                if stats is not None:
                    stats["skipped"] = stats.get("skipped", 0) + 1
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            if SKIP_NONFINITE and not torch.isfinite(gn):
                opt.zero_grad()
                if stats is not None:
                    stats["skipped"] = stats.get("skipped", 0) + 1
                continue
            opt.step()
            m.update_target()
            if stats is not None:
                stats["steps"] = stats.get("steps", 0) + 1
    m.eval()
    return m


@torch.no_grad()
def predict(m, ctx, chunk=4096):
    outs = {"vol": [], "df": [], "scale": [], "regime": []}
    for i in range(0, len(ctx), chunk):
        o = m.forward(torch.tensor(ctx[i:i + chunk]))
        for k in outs:
            outs[k].append(o[k].numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def run_seed(k):
    torch.set_num_threads(1)
    t0 = time.time()
    R, _, _, _, frames, asset_win = build()
    recs, skips = [], {}
    ck = OUT / f"partial_seed{k}"
    ck.mkdir(parents=True, exist_ok=True)
    for f in fold_grid(R):
        fp = ck / f"fold{f['fold']:02d}.parquet"
        if fp.exists():                                   # resume after a restart
            recs.append(pd.read_parquet(fp))
            skips[f["fold"]] = json.loads((ck / f"fold{f['fold']:02d}.json").read_text())
            continue
        core = {a: v for a, v in asset_win.items() if a in bv.CORE_ASSETS}
        mean, std = train_scaler(frames, core, f["cutoff"])
        W = wm_windows(asset_win, frames, mean, std)
        tr = {kk: [] for kk in ("ctx", "fut", "y", "r_next")}
        for a, w in W.items():
            if a not in bv.CORE_ASSETS:
                continue
            m = (w["dates"] < f["cutoff"]) & (w["end_1"] < np.datetime64(f["test_start"]))
            for kk in tr:
                tr[kk].append(w[kk][m])
        tr = {kk: np.concatenate(v) for kk, v in tr.items()}
        cfg = WMConfig(n_features=len(FEATURES), window=bv.L_NEURAL, seed=f["fold"] * 100 + k)
        st = {}
        model = train_wm(cfg, tr, stats=st)
        fold_recs = []
        for a, w in W.items():
            tm = np.array([d in f["test_dates"] for d in w["dates"].values])
            if not tm.any():
                continue
            p = predict(model, w["ctx"][tm])
            fold_recs.append(pd.DataFrame({"asset": a, "date": w["dates"][tm], "fold": f["fold"],
                                           f"WM_s{k}": p["vol"], f"WMdf_s{k}": p["df"],
                                           f"WMscale_s{k}": p["scale"], f"WMreg_s{k}": p["regime"]}))
        fr = pd.concat(fold_recs, ignore_index=True)
        fr.to_parquet(fp)
        (ck / f"fold{f['fold']:02d}.json").write_text(json.dumps(st))
        recs.append(fr)
        skips[f["fold"]] = st
        print(f"  seed {k} fold {f['fold']:2d} {pd.Timestamp(f['test_start']).date()} train={len(tr['y'])} skipped={st.get('skipped', 0)} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(recs, ignore_index=True).to_parquet(OUT / f"bench_seed{k}.parquet")
    (OUT / f"bench_seed{k}_steps.json").write_text(json.dumps({str(a): b for a, b in skips.items()}, indent=1))
    print(f"seed {k} done in {time.time() - t0:.0f}s", flush=True)


def combine():
    P = pd.read_parquet(ROOT / "results" / "benchmark_vol_predictions.parquet")
    S = [pd.read_parquet(OUT / f"bench_seed{k}.parquet") for k in range(N_SEEDS)]
    W = S[0]
    for s in S[1:]:
        W = W.merge(s.drop(columns="fold"), on=["asset", "date"], how="inner")
    W["MeridianWM"] = W[[f"WM_s{k}" for k in range(N_SEEDS)]].mean(axis=1)
    W["WM_df"] = W[[f"WMdf_s{k}" for k in range(N_SEEDS)]].mean(axis=1)
    W["WM_scale"] = W[[f"WMscale_s{k}" for k in range(N_SEEDS)]].mean(axis=1)
    W["WM_regime"] = W["WMreg_s0"]
    # fold grid identity with the audited benchmark
    chk = P[["asset", "date", "fold"]].merge(W[["asset", "date", "fold"]], on=["asset", "date"],
                                             how="outer", suffixes=("_bench", "_wm"), indicator=True)
    both = chk["_merge"] == "both"
    grid = {"rows_bench": int(len(P)), "rows_wm": int(len(W)), "rows_matched": int(both.sum()),
            "fold_mismatch": int((chk.loc[both, "fold_bench"] != chk.loc[both, "fold_wm"]).sum()),
            "only_in_bench": int((chk["_merge"] == "left_only").sum()),
            "only_in_wm": int((chk["_merge"] == "right_only").sum())}
    M = P.merge(W[["asset", "date", "MeridianWM", "WM_df", "WM_scale", "WM_regime"]
                  + [f"WM_s{k}" for k in range(N_SEEDS)]], on=["asset", "date"], how="left")
    M.to_parquet(OUT / "bench_wm.parquet")
    steps = {f"seed{k}": json.loads((OUT / f"bench_seed{k}_steps.json").read_text()) for k in range(N_SEEDS)}
    grid["nonfinite_forecasts"] = int((~np.isfinite(W["MeridianWM"])).sum())
    from datetime import datetime, timezone
    (OUT / "bench_wm_meta.json").write_text(json.dumps({"seeds": N_SEEDS, "epochs": EPOCHS, "fold_grid_check": grid,
                                                        "combined_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                        "amendment": "8a: optimizer steps with non-finite loss/grad skipped",
                                                        "steps_per_fold": steps,
                                                        "data_end": str(BENCH_DATA_END.date())}, indent=2))
    print(json.dumps(grid, indent=2))


if __name__ == "__main__":
    a = sys.argv[1]
    combine() if a == "combine" else run_seed(int(a))
