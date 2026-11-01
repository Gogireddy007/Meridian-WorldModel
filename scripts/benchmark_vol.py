"""Volatility benchmark -- PREDICTION GENERATOR (PREREGISTRATION.md Amendment 6).

Fits every model per walk-forward fold on train-only, purged data and writes each
model's raw out-of-sample log-variance forecast per (asset, date) to
results/<BENCH_OUT> (parquet). It computes NO metrics: all scoring is done from that
file by scripts/score_vol_benchmark.py, so every reported number can be recomputed in
seconds without retraining.

Protocol
  * Date-based expanding walk-forward: first test block after ~5y, 378-day blocks.
  * Embargo: train rows must be dated before test_start minus 22 union-calendar days.
  * Purge by label end: a row used to fit a horizon-h target must have its h-day label
    window end (on that asset's own calendar) strictly before test_start.
  * All scalers/normalizers are fit on the fold's train rows only.
  * Pooled models (HAR_pooled, Ridge, RandomForest, LightGBM, TimeMixer, Meridian) train
    on the pre-registered CORE universe only; TLT/GLD rows are scored separately as an
    out-of-sample generalization check. Per-asset models train on each asset's own history.

Models (all output a log-variance forecast of RV_{t+1} made at the close of day t)
  Naive     RW (log RV_t), MA22 (log of 22-day mean RV), EWMA (RiskMetrics, lambda=0.94)
  GARCH     GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1) on returns, per asset
  HAR       HAR_pooled (one pooled regression), HAR (per asset, the reference), LHAR
            (Corsi-Reno leverage terms), SHAR (daily-sign analogue of the Patton-Sheppard
            semivariance split), HARX (+ log VIX)
  ML        Ridge (asset fixed effects), RandomForest, LightGBM
  Deep      TimeMixer (compact faithful reimplementation)
  Meridian  variants selected by BENCH_VARIANTS:
            M0  current core: pooled HAR-anchored init (Amendment 5 config)
            M1  per-asset HAR offset: forecast = per-asset HAR + zero-init NN correction
            M2  M1 + learned asset embedding
            M3  M1 + early stopping on a purged, time-ordered validation slice
  Level models (EWMA, GARCH family) are scaled by a train-only mean(RV)/mean(forecast)
  factor; log models carry a train-only Jensen term 0.5*Var(residual).

HARQ, HAR-CJ and Realized GARCH need intraday data and are omitted (not approximated).
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import load_all, EQUITIES, FX, BREADTH
from meridian.features import build_asset_frame, realized_variance
from meridian.model import Meridian, MeridianConfig
from meridian.windows import FEATURES, AUX_HORIZONS, asset_matrix, build_windows, train_scaler

EPS = 1e-12
L = 40                        # TimeMixer input window
L_NEURAL = 32                 # Meridian input window
MIN_TRAIN_D = 1260
TEST_D = 378
EMBARGO_D = 22
RESULTS = Path(__file__).resolve().parent.parent / "results"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

CORE_ASSETS = sorted(set(EQUITIES) | set(FX.keys()))
BREADTH_ASSETS = sorted(BREADTH.keys())
ASSET_INDEX = {a: i for i, a in enumerate(CORE_ASSETS)}      # unseen assets -> len(CORE_ASSETS)

VARIANTS = [v for v in os.environ.get("BENCH_VARIANTS", "M1").split(",") if v]
ENSEMBLE = int(os.environ.get("BENCH_ENSEMBLE", "5"))
EPOCHS = int(os.environ.get("BENCH_EPOCHS", "8"))
MAX_EPOCHS_ES = int(os.environ.get("BENCH_MAX_EPOCHS_ES", "20"))
BATCH = 256
FOLD_LIMIT = int(os.environ.get("BENCH_FOLD_LIMIT", "0")) or None
OUT_NAME = os.environ.get("BENCH_OUT", "benchmark_vol_predictions.parquet")

HAR3 = ["har_d", "har_w", "har_m"]
RF_FEATS = ["har_d", "har_w", "har_m", "log_rv", "ret_abs", "ret_5", "vix",
            "nsv_d", "nsv_w", "nsv_m", "psv_d"]
ML_FEATS = ["har_d", "har_w", "har_m", "lev_d", "lev_w", "lev_m", "ret_abs", "vix",
            "down", "har_d_dn"]
PANEL_COLS = (["asset", "date", "rv", "y", "y_5", "y_22", "rv_next", "rv_next_5", "rv_next_22",
               "end_1", "end_5", "end_22", "log_rv", "ret", "ret_abs", "ret_5", "vix"]
              + HAR3 + ["nsv_d", "nsv_w", "nsv_m", "psv_d", "lev_d", "lev_w", "lev_m",
                        "down", "har_d_up", "har_d_dn"])
NEED_FINITE = HAR3 + ["nsv_d", "lev_d", "lev_w", "lev_m", "vix", "y", "rv_next"]
HAR_IDX = tuple(FEATURES.index(c) for c in HAR3)
TARGET_END = {"y": "end_1", "y_5": "end_5", "y_22": "end_22"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def clean_ohlc(o):
    """Repair Yahoo OHLC bars (mostly FX):
      * drop stale bars with high <= low (O=H=L=C, volume 0: no real quote that day);
      * clip open/close into [low, high] where they fall outside (a snapshot-timing quirk,
        ~3% of FX days, usually by a fraction of the day's range).
    Left as-is, the range-based RV estimator reads these bars as zero or negative variance
    (clipped to 1e-12, log ~ -27.6), which dominates QLIKE."""
    o = o[o["high"] > o["low"]].copy()
    for c in ("open", "close"):
        o[c] = o[c].clip(lower=o["low"], upper=o["high"])
    return o


def load_clean():
    d = load_all(include_breadth=True)
    d["prices"] = {a: clean_ohlc(o) for a, o in d["prices"].items()}
    return d


def build():
    d = load_clean()
    macro = d["macro"]
    panels, WlogA, WlevA, frames, asset_win = [], [], [], {}, {}
    for a, ohlc in d["prices"].items():
        f = build_asset_frame(ohlc, macro)
        f["lev_d"] = f["ret"].clip(upper=0)
        f["lev_w"] = f["lev_d"].rolling(5).mean()
        f["lev_m"] = f["lev_d"].rolling(22).mean()
        f["down"] = (f["ret"] < 0).astype(float)
        f["har_d_up"] = f["har_d"] * (1 - f["down"])
        f["har_d_dn"] = f["har_d"] * f["down"]
        n = len(f)
        for h in (1, 5, 22):                    # label-end date on this asset's own calendar
            e = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
            e[:n - h] = f.index.values[h:]
            f[f"end_{h}"] = e
        f["asset"] = a
        f["date"] = f.index
        frames[a] = f

        lrv = f["log_rv"].to_numpy(np.float32)
        lev = np.nan_to_num(f["lev_d"].to_numpy(np.float32))
        Wl, Wv = sliding_window_view(lrv, L), sliding_window_view(lev, L)
        cand = np.arange(L, n - 1)
        fin = np.isfinite(f[NEED_FINITE].to_numpy(float)).all(1)
        keep = cand[fin[cand]]
        panels.append(f.iloc[keep][PANEL_COLS].reset_index(drop=True))
        WlogA.append(Wl[keep - L + 1]); WlevA.append(Wv[keep - L + 1])
        asset_win[a] = asset_matrix(f)
    R = pd.concat(panels, ignore_index=True)
    return R, np.concatenate(WlogA), np.concatenate(WlevA), d, frames, asset_win


def train_rows(R, cutoff, test_start, target="y", assets=None):
    m = (R["date"] < cutoff) & (R[TARGET_END[target]] < test_start)
    if assets is not None:
        m &= R["asset"].isin(assets)
    return m.to_numpy()


# --------------------------------------------------------------------------- #
# Linear / ML models
# --------------------------------------------------------------------------- #
def ols(tr, cols, target="y"):
    t = tr[cols + [target]].dropna()
    if len(t) < 100:
        return None
    A = np.column_stack([np.ones(len(t))] + [t[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(A, t[target].to_numpy(), rcond=None)
    jb = 0.5 * float(np.var(t[target].to_numpy() - A @ beta))
    return beta, jb


def ols_apply(te, cols, fit):
    if fit is None:
        return np.full(len(te), np.nan)
    beta, jb = fit
    B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    return B @ beta + jb


def offset_model(tr, col, target="y"):
    """Fixed-coefficient model y = col + c; returns the Jensen-corrected offset c."""
    t = tr[[col, target]].dropna()
    if len(t) < 100:
        return None
    r = t[target] - t[col]
    return float(r.mean()) + 0.5 * float(r.var())


def ml_matrix(df, scaler=None):
    X = df[ML_FEATS].to_numpy(float)
    if scaler is None:
        scaler = (X.mean(0), X.std(0) + 1e-9)
    Xs = (X - scaler[0]) / scaler[1]
    D = np.column_stack([(df["asset"] == a).to_numpy(float) for a in CORE_ASSETS])
    return np.column_stack([Xs, D]), scaler


def fit_ridge(tr, te):
    Xtr, sc = ml_matrix(tr)
    m = Ridge(alpha=1.0).fit(Xtr, tr["y"].to_numpy())
    jb = 0.5 * float(np.var(tr["y"].to_numpy() - m.predict(Xtr)))
    return m.predict(ml_matrix(te, sc)[0]) + jb


def fit_rf(tr, te):
    t = tr[RF_FEATS + ["y"]].replace([np.inf, -np.inf], np.nan).dropna()
    m = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=15, n_jobs=-1,
                              random_state=0, oob_score=True).fit(t[RF_FEATS].to_numpy(), t["y"].to_numpy())
    jb = 0.5 * float(np.var(t["y"].to_numpy() - m.oob_prediction_))
    return m.predict(te[RF_FEATS].to_numpy()) + jb


def fit_lgbm(tr, te):
    """LightGBM in a subprocess (meridian/lgbm_worker.py): it cannot share a process with
    PyTorch-MPS on macOS (conflicting OpenMP runtimes crash)."""
    cols = ML_FEATS + ["asset", "date", "end_1", "y"]
    with tempfile.TemporaryDirectory() as tmp:
        tp, ep, op = (str(Path(tmp) / n) for n in ("tr.parquet", "te.parquet", "out.npy"))
        tr[cols].to_parquet(tp)
        te[ML_FEATS + ["asset"]].to_parquet(ep)
        subprocess.run([sys.executable, "-m", "meridian.lgbm_worker", tp, ep, op, ",".join(CORE_ASSETS)],
                       check=True, cwd=str(RESULTS.parent))
        return np.load(op)


# --------------------------------------------------------------------------- #
# Level models (EWMA, GARCH family): one-step-ahead variance for day t+1 at close t
# --------------------------------------------------------------------------- #
def ewma_next(rv, lam=0.94):
    """next[t] = forecast of rv[t+1] using rv[0..t]."""
    nxt = np.empty(len(rv)); s = rv[0]
    for t in range(len(rv)):
        s = lam * s + (1 - lam) * rv[t]
        nxt[t] = s
    return nxt


GARCH_SPECS = {"GARCH": dict(vol="GARCH", p=1, o=0, q=1),
               "GJR_GARCH": dict(vol="GARCH", p=1, o=1, q=1),
               "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1)}


def garch_next(ret, train_mask, kind):
    """next[t] = conditional variance of day t+1's return given returns through day t."""
    from arch import arch_model
    r = np.nan_to_num(ret * 100.0, nan=0.0)
    spec = GARCH_SPECS[kind]
    try:
        res = arch_model(r[train_mask], mean="Zero", rescale=False, **spec).fit(disp="off")
        fixed = arch_model(r, mean="Zero", rescale=False, **spec).fix(res.params)
        s2 = np.asarray(fixed.conditional_volatility) ** 2
    except Exception:
        return None
    nxt = np.full(len(r), np.nan)
    nxt[:-1] = s2[1:]
    if not np.isfinite(nxt[:-1]).all() or (nxt[:-1] <= 0).any():
        return None
    return nxt / 1e4


# --------------------------------------------------------------------------- #
# TimeMixer (compact, faithful): multiscale downsample + decomposition + mixing
# --------------------------------------------------------------------------- #
def moving_avg(x, k):
    pad = k // 2
    xp = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    return torch.nn.functional.avg_pool1d(xp, k, stride=1)[..., :x.shape[-1]]


class TimeMixer(nn.Module):
    def __init__(self, L=L, C=2, scales=(1, 2, 4), hid=48):
        super().__init__()
        self.scales = scales
        self.enc = nn.ModuleList()
        for f in scales:
            Ls = max(L // f, 4)
            self.enc.append(nn.Sequential(nn.Linear(2 * C * Ls, hid), nn.GELU(), nn.Linear(hid, hid)))
        self.mix = nn.Sequential(nn.Linear(hid * len(scales), hid), nn.GELU(), nn.Linear(hid, hid))
        self.heads = nn.ModuleList([nn.Linear(hid, 1) for _ in scales])
        self.head_mix = nn.Linear(hid, 1)

    def forward(self, x):
        feats = []
        for f, enc in zip(self.scales, self.enc):
            xs = x if f == 1 else torch.nn.functional.avg_pool1d(x, f)
            trend = moving_avg(xs, 5); seas = xs - trend
            feats.append(enc(torch.cat([seas.flatten(1), trend.flatten(1)], 1)))
        out = self.head_mix(self.mix(torch.cat(feats, 1)))
        for h, fe in zip(self.heads, feats):
            out = out + h(fe)
        return out.squeeze(-1)


def fit_timemixer(Wl_tr, Wv_tr, y_tr, Wl_te, Wv_te, epochs=35):
    ml, sl = Wl_tr.mean(), Wl_tr.std() + 1e-6           # train-only statistics
    mv, sv = Wv_tr.mean(), Wv_tr.std() + 1e-6

    def stack(Wl, Wv):
        return torch.tensor(np.stack([(Wl - ml) / sl, (Wv - mv) / sv], 1), dtype=torch.float32).to(DEV)
    torch.manual_seed(0)
    net = TimeMixer().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    X, y = stack(Wl_tr, Wv_tr), torch.tensor((y_tr - ml) / sl, dtype=torch.float32).to(DEV)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = ((net(X[idx]) - y[idx]) ** 2).mean()
            loss.backward(); opt.step()
    with torch.no_grad():
        p_tr = net(X).cpu().numpy() * sl + ml
        p_te = net(stack(Wl_te, Wv_te)).cpu().numpy() * sl + ml
    return p_te + 0.5 * float(np.var(y_tr - p_tr))


# --------------------------------------------------------------------------- #
# Meridian
# --------------------------------------------------------------------------- #
def variant_cfg(variant, seed):
    base = dict(n_features=len(FEATURES), window=L_NEURAL, d_model=64, d_state=64, n_layers=2,
                loss_mode="qlike", aux_horizons=tuple(AUX_HORIZONS), lambda_aux=0.3, seed=seed)
    if variant == "M0":
        return MeridianConfig(**base, har_anchor=True, har_idx=HAR_IDX)
    if variant in ("M1", "M3"):
        return MeridianConfig(**base, offset_mode=True)
    if variant == "M2":
        return MeridianConfig(**base, offset_mode=True, n_assets=len(CORE_ASSETS), asset_emb_dim=8)
    raise ValueError(variant)


KEYS = ("ctx", "fut", "y", *AUX_HORIZONS, "off_y", *(f"off_{h}" for h in AUX_HORIZONS), "asset_id")


def _to_batch(T, idx):
    b = {"x_ctx": T["ctx"][idx], "x_fut": T["fut"][idx], "y": T["y"][idx],
         "off_y": T["off_y"][idx], "asset_id": T["asset_id"][idx]}
    for h in AUX_HORIZONS:
        b[h] = T[h][idx]; b[f"off_{h}"] = T[f"off_{h}"][idx]
    return b


def _tensors(W):
    return {k: torch.tensor(W[k], device=DEV) for k in KEYS}


@torch.no_grad()
def _forecast(m, T, use_offset):
    m.eval()
    aux_off = {h: T[f"off_{h}"] for h in AUX_HORIZONS} if use_offset else None
    o = m.forward(T["ctx"], None, offset=T["off_y"] if use_offset else None,
                  aux_offset=aux_off, asset_id=T["asset_id"])
    return o["vol"], o["aux"]


def train_meridian(variant, cfg, TR, VA, mean, std, har_beta, aux_beta):
    use_offset = variant != "M0"
    m = Meridian(cfg).to(DEV)
    if variant == "M0":
        m.init_har_anchor(har_beta, mean[list(HAR_IDX)], std[list(HAR_IDX)], aux_beta=aux_beta)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(cfg.seed)
    n = TR["y"].shape[0]
    epochs = MAX_EPOCHS_ES if VA is not None else EPOCHS
    best, best_loss, bad = None, np.inf, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = _to_batch(TR, perm[i:i + BATCH].to(DEV))
            if not use_offset:
                b.pop("off_y"); [b.pop(f"off_{h}") for h in AUX_HORIZONS]
            opt.zero_grad()
            loss, _ = m.loss(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            m.update_target()
        if VA is not None:
            f, _ = _forecast(m, VA, use_offset)
            vl = float((torch.exp(VA["y"] - f) - (VA["y"] - f) - 1).mean())
            if vl < best_loss - 1e-6:
                best_loss, bad = vl, 0
                best = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 3:
                    break
    if best is not None:
        m.load_state_dict(best)
    return m


def meridian_windows(asset_win, frames, mean, std, per_asset_off):
    """Scaled windows for every asset, plus per-window offsets, asset ids and label ends."""
    out = {}
    for a, (dates, X, y, aux) in asset_win.items():
        w = build_windows(dates, X, y, L_NEURAL, mean, std, aux=aux)
        if w is None:
            continue
        raw = X[L_NEURAL - 1:len(X) - 1]                       # unscaled anchor-row features
        dts = pd.DatetimeIndex(w["dates"])
        w["end_22"] = frames[a].loc[dts, "end_22"].to_numpy()
        w["asset_id"] = np.full(len(dts), ASSET_INDEX.get(a, len(CORE_ASSETS)), dtype=np.int64)
        for tgt in ("y", *AUX_HORIZONS):
            fit = per_asset_off.get((a, tgt))
            if fit is None:
                w[f"off_{tgt}"] = np.full(len(dts), np.nan, np.float32)
            else:
                beta, jb = fit
                w[f"off_{tgt}"] = (beta[0] + raw[:, list(HAR_IDX)] @ beta[1:] + jb).astype(np.float32)
        w["dates"] = dts
        out[a] = w
    return out


def run_meridian_fold(variants, frames, asset_win, cutoff, test_start, test_end, per_asset_off,
                      har_beta, aux_beta, seed_base):
    core_win = {a: v for a, v in asset_win.items() if a in CORE_ASSETS}
    mean, std = train_scaler(frames, core_win, cutoff)
    W = meridian_windows(asset_win, frames, mean, std, per_asset_off)

    def cat(parts):
        return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
    tr_parts, tr_dates, tr_ends, te = [], [], [], []
    for a, w in W.items():
        ok = np.isfinite(w["off_y"]) & np.isfinite(w["off_y_5"]) & np.isfinite(w["off_y_22"])
        trm = (w["dates"] < cutoff) & (w["end_22"] < np.datetime64(test_start)) & ok
        if a in CORE_ASSETS and trm.any():
            tr_parts.append({k: w[k][trm] for k in KEYS})
            tr_dates.append(w["dates"].values[trm])
            tr_ends.append(w["end_22"][trm])
        tem = (w["dates"] >= test_start) & (w["dates"] < test_end) & ok
        if tem.any():
            te.append((a, w["dates"][tem], {k: w[k][tem] for k in KEYS}))
    if not tr_parts or not te:
        return None
    TR_all = cat(tr_parts)
    all_dates = np.concatenate(tr_dates)
    ends = np.concatenate(tr_ends)
    # early-stopping split: last 10% of train dates is validation; fit rows must have
    # their 22-day label end before validation starts (purged)
    uniq = np.unique(all_dates)
    val_start = uniq[int(len(uniq) * 0.9)]
    fit_m = ends < val_start
    val_m = all_dates >= val_start

    out = []
    for variant in variants:
        if variant == "M3":
            TR = _tensors({k: v[fit_m] for k, v in TR_all.items()})
            VA = _tensors({k: v[val_m] for k, v in TR_all.items()})
        else:
            TR, VA = _tensors(TR_all), None
        models = [train_meridian(variant, variant_cfg(variant, seed_base + s), TR, VA, mean, std,
                                 har_beta, aux_beta) for s in range(ENSEMBLE)]
        for a, dts, w in te:
            T = _tensors(w)
            vols, auxs = [], {h: [] for h in AUX_HORIZONS}
            for m in models:
                v, ax = _forecast(m, T, variant != "M0")
                vols.append(v.cpu().numpy())
                for h in AUX_HORIZONS:
                    auxs[h].append(ax[h].cpu().numpy())
            rec = pd.DataFrame({"asset": a, "date": dts, f"Meridian_{variant}": np.mean(vols, 0)})
            for h in AUX_HORIZONS:
                rec[f"Meridian_{variant}_{h}"] = np.mean(auxs[h], 0)
            out.append(rec)
    df = pd.concat(out, ignore_index=True)
    return df.groupby(["asset", "date"], as_index=False).first()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"device={DEV} variants={VARIANTS} ensemble={ENSEMBLE} epochs={EPOCHS} out={OUT_NAME}", flush=True)
    R, Wlog, Wlev, d, frames, asset_win = build()
    level = {}
    for a, o in d["prices"].items():
        rvf = realized_variance(o)
        level[a] = {"rv": rvf["rv"], "ret": rvf["ret"],
                    "ewma": pd.Series(ewma_next(rvf["rv"].to_numpy()), index=rvf.index)}
    dates = np.array(sorted(R["date"].unique()))
    preds, fold, i = [], 0, MIN_TRAIN_D
    while i < len(dates):
        if FOLD_LIMIT is not None and fold >= FOLD_LIMIT:
            break
        test_dates = dates[i:i + TEST_D]
        test_start, test_end = dates[i], dates[min(i + TEST_D, len(dates) - 1)]
        cutoff = dates[i - EMBARGO_D]
        te_m = R["date"].isin(set(test_dates)).to_numpy()
        tr_core = train_rows(R, cutoff, test_start, "y", CORE_ASSETS)
        if tr_core.sum() < 2000 or te_m.sum() < 100:
            i += TEST_D; continue
        fold += 1
        te = R[te_m].reset_index(drop=True)
        out = te[["asset", "date", "y", "y_5", "y_22", "rv_next", "rv_next_5", "rv_next_22",
                  "har_d", "vix"]].copy()
        out["fold"] = fold
        trc = R[tr_core]

        # pooled models (core-trained)
        out["HAR_pooled"] = ols_apply(te, HAR3, ols(trc, HAR3))
        out["Ridge"] = fit_ridge(trc, te)
        out["RandomForest"] = fit_rf(trc, te)
        out["LightGBM"] = fit_lgbm(trc, te)
        out["TimeMixer"] = fit_timemixer(Wlog[tr_core], Wlev[tr_core], trc["y"].to_numpy(),
                                         Wlog[te_m], Wlev[te_m])
        pooled_beta = ols(trc, HAR3)[0]
        pooled_aux = {h: ols(R[train_rows(R, cutoff, test_start, h, CORE_ASSETS)], HAR3, h)[0]
                      for h in AUX_HORIZONS}

        # per-asset models
        per_asset_off = {}
        cols = {k: np.full(len(te), np.nan) for k in
                ["RW", "MA22", "HAR", "LHAR", "SHAR", "HARX", "HAR_y_5", "HAR_y_22",
                 "EWMA", *GARCH_SPECS]}
        for a in te["asset"].unique():
            ai = (te["asset"] == a).to_numpy()
            tea = te[ai]
            tra = R[train_rows(R, cutoff, test_start, "y", [a])]
            c = offset_model(tra, "har_d"); cols["RW"][ai] = tea["har_d"] + c if c is not None else np.nan
            c = offset_model(tra, "har_m"); cols["MA22"][ai] = tea["har_m"] + c if c is not None else np.nan
            har_fit = ols(tra, HAR3)
            cols["HAR"][ai] = ols_apply(tea, HAR3, har_fit)
            per_asset_off[(a, "y")] = har_fit
            cols["LHAR"][ai] = ols_apply(tea, HAR3 + ["lev_d", "lev_w", "lev_m"],
                                         ols(tra, HAR3 + ["lev_d", "lev_w", "lev_m"]))
            shar = ["har_d_up", "har_d_dn", "down", "har_w", "har_m"]
            cols["SHAR"][ai] = ols_apply(tea, shar, ols(tra, shar))
            cols["HARX"][ai] = ols_apply(tea, HAR3 + ["vix"], ols(tra, HAR3 + ["vix"]))
            for h in AUX_HORIZONS:
                fh = ols(R[train_rows(R, cutoff, test_start, h, [a])], HAR3, h)
                cols[f"HAR_{h}"][ai] = ols_apply(tea, HAR3, fh)
                per_asset_off[(a, h)] = fh

            lv = level[a]
            tdates = tea["date"].to_numpy()
            trd = (lv["rv"].index < cutoff)
            rv_next_tr = lv["rv"].shift(-1)[trd].to_numpy()
            ew = lv["ewma"]
            cE = np.nanmean(rv_next_tr) / np.nanmean(ew[trd].to_numpy())
            cols["EWMA"][ai] = np.log(np.clip(cE * ew.reindex(tdates).to_numpy(), EPS, None))
            for kind in GARCH_SPECS:
                nxt = garch_next(lv["ret"].to_numpy(), np.asarray(trd), kind)
                if nxt is None:
                    continue
                s = pd.Series(nxt, index=lv["ret"].index)
                cG = np.nanmean(rv_next_tr) / np.nanmean(s[trd].to_numpy())
                cols[kind][ai] = np.log(np.clip(cG * s.reindex(tdates).to_numpy(), EPS, None))
        for k, v in cols.items():
            out[k] = v

        # Meridian variants
        if VARIANTS:
            mf = run_meridian_fold(VARIANTS, frames, asset_win, cutoff, test_start, test_end,
                                   per_asset_off, pooled_beta, pooled_aux, seed_base=fold * 100)
            if mf is not None:
                out = out.merge(mf, on=["asset", "date"], how="left")
        preds.append(out)
        print(f"  fold {fold:2d} {pd.Timestamp(test_start).date()}→{pd.Timestamp(test_end).date()} "
              f"train_core={tr_core.sum()} test={te_m.sum()}  [{time.time() - t0:.0f}s]", flush=True)
        i += TEST_D

    P = pd.concat(preds, ignore_index=True)
    P["breadth"] = P["asset"].isin(BREADTH_ASSETS)
    RESULTS.mkdir(exist_ok=True)
    P.to_parquet(RESULTS / OUT_NAME)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=RESULTS.parent).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {"out": OUT_NAME, "variants": VARIANTS, "ensemble": ENSEMBLE, "epochs": EPOCHS,
            "max_epochs_early_stop": MAX_EPOCHS_ES, "folds": fold, "rows": int(len(P)),
            "min_train_d": MIN_TRAIN_D, "test_d": TEST_D, "embargo_d": EMBARGO_D,
            "git_commit_at_run": commit, "runtime_s": round(time.time() - t0)}
    (RESULTS / (Path(OUT_NAME).stem + "_meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"saved {len(P)} rows -> results/{OUT_NAME}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
