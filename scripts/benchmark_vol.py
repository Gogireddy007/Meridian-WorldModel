"""Volatility-forecast BENCHMARK — Meridian vs GARCH, EWMA, HAR, and TimeMixer.

Protocol (as requested):
  * date-based WALK-FORWARD with PURGE + EMBARGO (no leakage across the target-overlap edge),
  * metrics: QLIKE (variance), RMSE (log-vol), IC (Spearman rank corr of forecast vs realized),
  * Diebold-Mariano tests vs HAR, and a Model Confidence Set (Hansen-Lunde-Nason) at 90%.

Models (all fit only on each fold's train, evaluated on its embargoed test block):
  * EWMA (RiskMetrics, lambda=0.94 on RV) — recursive.
  * GARCH(1,1) — arch MLE per fold/asset; leak-free variance recursion, train-mean calibrated.
  * HAR-RV (Corsi 2009) — OLS on the daily/weekly/monthly log-RV cascade.
  * HAR+Lev — HAR plus a single down-move (leverage) feature; kept for continuity with
    earlier reports, but this is NOT the production Meridian model (see below).
  * RF+SemiVar — lean RandomForest on HAR + good/bad realized-semivariance features
    (Patton-Sheppard decomposition; scripts/run_semivar_ml.py), pooled across assets.
  * Meridian — the ACTUAL production core (meridian/model.py): a QLIKE-trained,
    5-seed CF-JEPA diagonal-SSM ensemble with semivariance features and auxiliary
    multi-horizon (h=5, h=22) heads, trained per fold on that fold's train windows.
    This replaces the old 4-coefficient OLS placeholder that used to be reported here
    (see PREREGISTRATION.md Amendment 4) — this script now evaluates the SAME model
    class that produced the MODEL_CARD headline numbers, inside this rigorous harness.
  * Meridian+ — equal-weight log-space blend of Meridian and RF+SemiVar (the
    heterogeneous-ensembling lever flagged as the best result in RESULTS.md).
  * TimeMixer (compact) — faithful reimplementation: multiscale downsampling + series
    decomposition + cross-scale (past-decomposable) mixing + multi-predictor head, trained pooled.

Level forecasts (EWMA/GARCH) and log forecasts (HAR/HAR+Lev/RF+SemiVar/Meridian/Meridian+/
TimeMixer) are put on a common footing by a train-only calibration (multiplicative for level
models; Jensen for log models), so QLIKE compares DYNAMICS, not a variance-proxy offset.
Honest: whatever wins, wins.

Extra reporting cuts (see PREREGISTRATION.md Amendment 4):
  * BREADTH — all pooled models (HAR/RF+SemiVar/Meridian/TimeMixer) are trained ONLY on
    the pre-registered CORE universe (8 equities + 3 FX), never on TLT/GLD, so the CORE
    table stays comparable to the pre-amendment protocol. The CORE+BREADTH table applies
    those same core-trained models to TLT/GLD as an out-of-sample cross-asset-class
    GENERALIZATION test, not an expanded-training-data benchmark.
  * HORIZON — QLIKE/RMSE/IC at h=1 (daily), h=5 (weekly), h=22 (monthly) for the models
    that support multi-horizon targets natively (HAR, RF+SemiVar, Meridian); EWMA/GARCH/
    TimeMixer remain h=1-only in this script (not silently dropped — reported as "n/a").
  * REGIME  — QLIKE by VIX tercile (calm / normal / stress) on the pooled OOS test rows.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import load_all, EQUITIES, FX, BREADTH
from meridian.features import build_asset_frame
from meridian.evalproto import qlike, diebold_mariano
from meridian.model import Meridian, MeridianConfig
from meridian.windows import FEATURES, AUX_HORIZONS, asset_matrix, build_windows, train_scaler

EPS = 1e-12
L = 40                        # input window for TimeMixer (unchanged)
L_NEURAL = 32                 # input window for the neural Meridian core (matches run_meridian.py)
MIN_TRAIN_D = 1260            # ~5y initial train (calendar days of data)
TEST_D = 378                  # ~1.5y test block
EMBARGO_D = 22
RESULTS = Path(__file__).resolve().parent.parent / "results"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

CORE_ASSETS = set(EQUITIES) | set(FX.keys())
BREADTH_ASSETS = set(BREADTH.keys())

# Dry-run knobs (defaults = full pre-registered-style config; override for a fast smoke test)
ENSEMBLE = int(os.environ.get("BENCH_ENSEMBLE", "5"))
EPOCHS = int(os.environ.get("BENCH_EPOCHS", "8"))
BATCH = int(os.environ.get("BENCH_BATCH", "256"))
FOLD_LIMIT = int(os.environ.get("BENCH_FOLD_LIMIT", "0")) or None   # 0/unset = no limit
INCLUDE_BREADTH = os.environ.get("BENCH_NO_BREADTH", "") != "1"

RF_FEATS = ["har_d", "har_w", "har_m", "log_rv", "ret_abs", "ret_5", "vix",
            "nsv_d", "nsv_w", "nsv_m", "psv_d"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build(include_breadth: bool = INCLUDE_BREADTH):
    d = load_all(include_breadth=include_breadth)
    macro = d["macro"]
    rows, WlogA, WlevA = [], [], []
    frames, asset_win = {}, {}
    cols = ["asset", "date", "rv", "y", "har_d", "har_w", "har_m",
            "nsv_d", "nsv_w", "nsv_m", "psv_d", "log_rv", "ret_abs", "ret_5", "vix",
            "y_5", "y_22", "rv_next", "rv_next_5", "rv_next_22"]
    need_finite = ["har_d", "har_w", "har_m", "nsv_d", "y", "rv_next"]
    for a, ohlc in d["prices"].items():
        f = build_asset_frame(ohlc, macro)
        frames[a] = f
        lrv = f["log_rv"].to_numpy()
        neg = f["nsv_d"].to_numpy()          # log bad (downside) semivariance at day
        n = len(f)
        for t in range(L, n - 1):
            row = f.iloc[t]
            if not np.isfinite(row[need_finite].to_numpy(dtype=float)).all():
                continue
            rows.append((a, f.index[t], row["rv"], row["y"], row["har_d"], row["har_w"], row["har_m"],
                         row["nsv_d"], row["nsv_w"], row["nsv_m"], row["psv_d"],
                         row["log_rv"], row["ret_abs"], row["ret_5"], row.get("vix", np.nan),
                         row["y_5"], row["y_22"], row["rv_next"], row["rv_next_5"], row["rv_next_22"]))
            WlogA.append(lrv[t - L + 1:t + 1]); WlevA.append(neg[t - L + 1:t + 1])
        asset_win[a] = asset_matrix(f)        # (dates, X[15], y, aux={y_5,y_22}) for the neural core
    R = pd.DataFrame(rows, columns=cols)
    return R, np.asarray(WlogA, np.float32), np.asarray(WlevA, np.float32), d, frames, asset_win


# --------------------------------------------------------------------------- #
# TimeMixer (compact, faithful): multiscale downsample + decomposition + mixing
# --------------------------------------------------------------------------- #
def moving_avg(x, k):                                   # x [B,C,Lc]
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

    def forward(self, x):                                # x [B,C,L]
        feats = []
        for f, enc in zip(self.scales, self.enc):
            xs = x if f == 1 else torch.nn.functional.avg_pool1d(x, f)
            trend = moving_avg(xs, 5); seas = xs - trend       # series decomposition
            z = torch.cat([seas.flatten(1), trend.flatten(1)], 1)
            feats.append(enc(z))
        mixed = self.mix(torch.cat(feats, 1))               # cross-scale (past-decomposable) mixing
        out = self.head_mix(mixed)
        for h, fe in zip(self.heads, feats):                # future multi-predictor mixing
            out = out + h(fe)
        return out.squeeze(-1)


def train_timemixer(Xtr, ytr, mu, sd, epochs=35):
    torch.manual_seed(0)
    net = TimeMixer().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    X = torch.tensor((Xtr - mu) / sd).to(DEV); y = torch.tensor((ytr - mu) / sd).to(DEV)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = ((net(X[idx]) - y[idx]) ** 2).mean()
            loss.backward(); opt.step()
    return net


# --------------------------------------------------------------------------- #
# Classical / lean-ML models
# --------------------------------------------------------------------------- #
def har_predict(tr, te, cols, target="y", return_beta=False):
    trf = tr[cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    A = np.column_stack([np.ones(len(trf))] + [trf[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(A, trf[target].to_numpy(), rcond=None)
    jb = 0.5 * np.var(trf[target].to_numpy() - A @ beta)
    B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    pred = B @ beta + jb                                    # log-var forecast (Jensen-corrected)
    return (pred, beta) if return_beta else pred


def rf_semivar_predict(tr, te, feats=RF_FEATS, target="y"):
    m = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=15,
                               n_jobs=-1, random_state=0)
    trf = tr[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    m.fit(trf[feats].to_numpy(), trf[target].to_numpy())
    teX = te[feats].to_numpy()
    return m.predict(teX)


def ewma_series(rv, lam=0.94):
    f = np.empty(len(rv)); f[0] = rv[0]
    for t in range(1, len(rv)):
        f[t] = lam * f[t - 1] + (1 - lam) * rv[t - 1]        # forecast of rv[t] from info < t
    return f


def garch_sigma2(ret, train_mask):
    from arch import arch_model
    r = np.nan_to_num(ret * 100.0, nan=0.0)             # arch rejects NaN (first .diff() is NaN)
    try:
        res = arch_model(r[train_mask], mean="Zero", vol="Garch", p=1, q=1, rescale=False).fit(disp="off")
        w, al, be = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    except Exception:
        return None
    s2 = np.empty(len(r)); s2[0] = np.nanvar(r[train_mask])
    for t in range(1, len(r)):
        s2[t] = w + al * r[t - 1] ** 2 + be * s2[t - 1]
    return s2 / (100.0 ** 2)                                 # back to return-variance units


# --------------------------------------------------------------------------- #
# Meridian — the real, QLIKE-trained, multi-horizon neural core (per fold)
# --------------------------------------------------------------------------- #
HAR_IDX = tuple(FEATURES.index(c) for c in ("har_d", "har_w", "har_m"))


def make_meridian_cfg(seed=0):
    return MeridianConfig(n_features=len(FEATURES), window=L_NEURAL, d_model=64, d_state=64,
                           n_layers=2, loss_mode="qlike", aux_horizons=tuple(AUX_HORIZONS),
                           lambda_aux=0.3, har_anchor=True, har_idx=HAR_IDX, seed=seed)


def train_meridian_ensemble(cfg, tr, k, mean=None, std=None, har_beta=None, aux_beta=None,
                             epochs=EPOCHS, batch=BATCH, lr=1e-3):
    models = []
    n = len(tr["y"])
    ctx = torch.tensor(tr["ctx"], device=DEV); fut = torch.tensor(tr["fut"], device=DEV)
    y = torch.tensor(tr["y"], device=DEV)
    aux_t = {h: torch.tensor(tr[h], device=DEV) for h in cfg.aux_horizons if h in tr}
    for s in range(k):
        m = Meridian(dataclasses.replace(cfg, seed=cfg.seed + s)).to(DEV)
        if cfg.har_anchor and har_beta is not None:
            m.init_har_anchor(har_beta, mean[list(cfg.har_idx)], std[list(cfg.har_idx)], aux_beta=aux_beta)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(cfg.seed + s)
        m.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch):
                b = perm[i:i + batch]
                bt = {"x_ctx": ctx[b], "x_fut": fut[b], "y": y[b]}
                for h, av in aux_t.items():
                    bt[h] = av[b]
                opt.zero_grad()
                loss, _ = m.loss(bt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                m.update_target()
        models.append(m)
    return models


@torch.no_grad()
def predict_meridian(models, w):
    ctx, fut = torch.tensor(w["ctx"], device=DEV), torch.tensor(w["fut"], device=DEV)
    vols, aux_preds = [], {h: [] for h in models[0].cfg.aux_horizons}
    for m in models:
        m.eval()
        out = m.forward(ctx, fut)
        vols.append(out["vol"].cpu().numpy())
        for h in aux_preds:
            aux_preds[h].append(out["aux"][h].cpu().numpy())
    vol = np.mean(vols, axis=0)
    aux = {h: np.mean(v, axis=0) for h, v in aux_preds.items()}
    return vol, aux


def run_meridian_fold(frames, asset_win, cutoff, test_start, test_end, train_assets, seed=0,
                       har_beta=None, aux_beta=None):
    """Train the 5-seed QLIKE ensemble on this fold's train windows (train_assets only —
    the pre-registered CORE universe), predict on ALL assets' test windows (core+breadth),
    so breadth rows test cross-asset-class generalization rather than polluting training.
    """
    core_win = {a: v for a, v in asset_win.items() if a in train_assets}
    mean, std = train_scaler(frames, core_win, cutoff)
    tr_parts, te_rows = [], []
    for a, (dates, X, y, aux) in asset_win.items():
        w = build_windows(dates, X, y, L_NEURAL, mean, std, aux=aux)
        if w is None:
            continue
        dt = pd.DatetimeIndex(w["dates"])
        tr_m = dt < cutoff
        te_m = (dt >= test_start) & (dt < test_end)
        if tr_m.any() and a in train_assets:
            tr_parts.append({k: w[k][tr_m] for k in ("ctx", "fut", "y", *AUX_HORIZONS)})
        if te_m.any():
            te_rows.append((a, dt[te_m], {k: w[k][te_m] for k in ("ctx", "fut", "y", *AUX_HORIZONS)}))
    if not tr_parts or not te_rows:
        return None
    tr = {k: np.concatenate([p[k] for p in tr_parts]) for k in ("ctx", "fut", "y", *AUX_HORIZONS)}
    cfg = make_meridian_cfg(seed=seed)
    models = train_meridian_ensemble(cfg, tr, ENSEMBLE, mean=mean, std=std,
                                      har_beta=har_beta, aux_beta=aux_beta)
    recs = []
    for a, dts, w in te_rows:
        vol, aux = predict_meridian(models, w)
        rec = pd.DataFrame({"asset": a, "date": dts, "Meridian": vol})
        for h, v in aux.items():
            rec[f"Meridian_{h}"] = v
        recs.append(rec)
    return pd.concat(recs, ignore_index=True)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def main():
    print(f"Building panel … (device={DEV}, breadth={INCLUDE_BREADTH}, "
          f"ensemble={ENSEMBLE}, epochs={EPOCHS})")
    R, Wlog, Wlev, d, frames, asset_win = build()
    R = R.reset_index(drop=True)
    per = {}
    for a, o in d["prices"].items():
        from meridian.features import realized_variance
        rvf = realized_variance(o)
        per[a] = {"rv": rvf["rv"], "ret": rvf["ret"], "ewma": pd.Series(ewma_series(rvf["rv"].to_numpy()), index=rvf.index)}
    dates = np.array(sorted(R["date"].unique()))
    mu = Wlog.mean(); sd = Wlog.std() + 1e-6

    preds = []
    i = MIN_TRAIN_D; fold = 0
    while i < len(dates):
        if FOLD_LIMIT is not None and fold >= FOLD_LIMIT:
            break
        test_dates = set(dates[i:i + TEST_D])
        test_start, test_end = dates[i], dates[min(i + TEST_D, len(dates) - 1)]
        cutoff = dates[i - EMBARGO_D]
        # Pooled-model TRAINING is always restricted to the pre-registered CORE universe
        # (§5 win bar), regardless of INCLUDE_BREADTH — breadth assets only ever appear
        # in evaluation (a cross-asset-class generalization test), never in training, so
        # the CORE-table HAR/RF/Meridian/TimeMixer numbers stay comparable to the
        # pre-amendment protocol (see PREREGISTRATION.md Amendment 4).
        tr_mask = (R["date"] < cutoff).to_numpy() & R["asset"].isin(CORE_ASSETS).to_numpy()
        te_mask = R["date"].isin(test_dates).to_numpy()
        if tr_mask.sum() < 2000 or te_mask.sum() < 100:
            i += TEST_D; continue
        fold += 1
        tr, te = R[tr_mask], R[te_mask]
        out = te[["asset", "date", "y", "y_5", "y_22", "rv_next", "rv_next_5", "rv_next_22", "vix"]].copy()

        # HAR & HAR+Lev (continuity baseline; NOT the production Meridian anymore)
        out["HAR"], har_beta = har_predict(tr, te, ["har_d", "har_w", "har_m"], return_beta=True)
        out["HAR+Lev"] = har_predict(tr, te, ["har_d", "har_w", "har_m", "nsv_d"])
        out["HAR_y_5"], har_beta_5 = har_predict(tr, te, ["har_d", "har_w", "har_m"], target="y_5", return_beta=True)
        out["HAR_y_22"], har_beta_22 = har_predict(tr, te, ["har_d", "har_w", "har_m"], target="y_22", return_beta=True)

        # RF+SemiVar (pooled lean RF on HAR+semivariance features)
        out["RF_SemiVar"] = rf_semivar_predict(tr, te)
        out["RF_SemiVar_y_5"] = rf_semivar_predict(tr, te, target="y_5")
        out["RF_SemiVar_y_22"] = rf_semivar_predict(tr, te, target="y_22")

        # EWMA & GARCH — level forecasts, train-mean calibrated, per asset
        ew, ga = np.full(len(te), np.nan), np.full(len(te), np.nan)
        for a in te["asset"].unique():
            aidx = te["asset"].to_numpy() == a
            rv = per[a]["rv"]; ewf = per[a]["ewma"]
            tdates = te["date"].to_numpy()[aidx]
            trd = per[a]["rv"].index < cutoff
            cE = np.nanmean(rv[trd].to_numpy()) / (np.nanmean(ewf[trd].to_numpy()) + EPS)
            ew[aidx] = cE * ewf.reindex(tdates).to_numpy()
            s2 = garch_sigma2(per[a]["ret"].to_numpy(), np.asarray(per[a]["ret"].index < cutoff))
            if s2 is not None:
                s2s = pd.Series(s2, index=per[a]["ret"].index)
                cG = np.nanmean(rv[trd].to_numpy()) / (np.nanmean(s2s[per[a]["ret"].index < cutoff].to_numpy()) + EPS)
                ga[aidx] = cG * s2s.reindex(tdates).to_numpy()
        out["EWMA"] = np.log(np.clip(ew, EPS, None))
        out["GARCH"] = np.log(np.clip(ga, EPS, None))

        # TimeMixer — pooled, trained on this fold's train windows
        Xtr = np.stack([Wlog[tr_mask], Wlev[tr_mask]], 1)
        net = train_timemixer(Xtr, tr["y"].to_numpy().astype(np.float32), mu, sd)
        Xte = torch.tensor((np.stack([Wlog[te_mask], Wlev[te_mask]], 1) - mu) / sd).to(DEV)
        with torch.no_grad():
            out["TimeMixer"] = (net(Xte).cpu().numpy() * sd + mu)

        # Meridian — real QLIKE-trained 5-seed neural ensemble (this fold's train windows)
        mfold = run_meridian_fold(frames, asset_win, cutoff, test_start, test_end,
                                   train_assets=CORE_ASSETS, seed=fold * 100,
                                   har_beta=har_beta, aux_beta={"y_5": har_beta_5, "y_22": har_beta_22})
        if mfold is not None:
            out = out.merge(mfold, on=["asset", "date"], how="left")
        else:
            out["Meridian"] = np.nan

        # Meridian+ — equal-weight log-space blend with RF+SemiVar
        if "Meridian" in out.columns:
            out["Meridian+"] = 0.5 * out["Meridian"] + 0.5 * out["RF_SemiVar"]
            if "Meridian_y_5" in out.columns:
                out["Meridian+_y_5"] = 0.5 * out["Meridian_y_5"] + 0.5 * out["RF_SemiVar_y_5"]
                out["Meridian+_y_22"] = 0.5 * out["Meridian_y_22"] + 0.5 * out["RF_SemiVar_y_22"]

        preds.append(out)
        n_mer = int(np.isfinite(out.get("Meridian", pd.Series(dtype=float)).to_numpy()).sum())
        print(f"  fold {fold}: train {tr_mask.sum()}  test {te_mask.sum()}  meridian_cov={n_mer}  "
              f"({min(test_dates)}→{max(test_dates)})", flush=True)
        i += TEST_D

    P = pd.concat(preds, ignore_index=True)
    P["breadth"] = P["asset"].isin(BREADTH_ASSETS)
    h1_models = ["EWMA", "GARCH", "HAR", "HAR+Lev", "RF_SemiVar", "Meridian", "Meridian+", "TimeMixer"]
    h1_models = [m for m in h1_models if m in P.columns]
    for m in h1_models:
        P[m] = P[m].replace([np.inf, -np.inf], np.nan)

    # Post-hoc leakage-safe recalibration + optimal combination (meridian.evalproto):
    # the SAME mechanism that produced the MODEL_CARD headline win (per-model,
    # per-asset walk-forward affine recalibration), applied uniformly here so
    # every model gets the same chance to have its LEVEL fixed — QLIKE punishes a
    # miscalibrated level even when the model's ranking/dynamics are good.
    from meridian.evalproto import walk_forward_calibrate, walk_forward_combine
    print("\nApplying post-hoc walk-forward calibration/combination …")
    cal_models = []
    for m in h1_models:
        col = f"{m}_cal"
        out_var = np.full(len(P), np.nan)
        for a, sub in P.groupby("asset"):
            var, mask = walk_forward_calibrate(sub["date"].to_numpy(), sub["y"].to_numpy(), sub[m].to_numpy())
            idx = sub.index.to_numpy()
            out_var[idx[mask]] = var[mask]
        P[col] = np.log(np.clip(out_var, EPS, None))
        P.loc[~np.isfinite(out_var), col] = np.nan
        cal_models.append(col)

    # Ensemble-GR: walk-forward Granger-Ramanathan optimal combination of the three
    # strongest heterogeneous forecasters (HAR, Meridian, RF+SemiVar), fit weights
    # per asset on past data only — replaces the naive fixed 0.5/0.5 Meridian+ blend
    # with data-driven weights.
    gr_var = np.full(len(P), np.nan)
    for a, sub in P.groupby("asset"):
        preds_log = {"HAR": sub["HAR"].to_numpy(), "Meridian": sub["Meridian"].to_numpy(),
                     "RF_SemiVar": sub["RF_SemiVar"].to_numpy()}
        var, mask = walk_forward_combine(sub["date"].to_numpy(), sub["y"].to_numpy(), preds_log)
        idx = sub.index.to_numpy()
        gr_var[idx[mask]] = var[mask]
    P["Ensemble_GR"] = np.log(np.clip(gr_var, EPS, None))
    P.loc[~np.isfinite(gr_var), "Ensemble_GR"] = np.nan
    cal_models.append("Ensemble_GR")

    def score_table(df, models, common=False, dm_ref="HAR"):
        """common=True: force ALL models in this table onto the SAME row support
        (intersection of finite predictions) before scoring anything — required
        whenever the model list mixes raw and calibrated columns, since the
        walk-forward calibration step drops a ~252-row warm-up per asset and
        would otherwise silently compare models on different (and differently
        hard) date ranges. Without this, a model evaluated on an easier subset
        of dates looks better for a reason that has nothing to do with skill.
        """
        rv_true = df["rv_next"].to_numpy(); y_log = df["y"].to_numpy()
        base_mk = np.isfinite(rv_true)
        if common:
            for m in models:
                base_mk = base_mk & np.isfinite(df[m].to_numpy())
        loss = {m: qlike(rv_true, np.exp(df[m].to_numpy())) for m in models}
        hmask = np.isfinite(loss.get(dm_ref, np.full(len(df), np.nan)))
        summary, qmeans = {}, {}
        print(f"  {'model':>14} {'QLIKE':>8} {'RMSE(log)':>10} {'IC':>7} {f'DM vs {dm_ref} (p)':>18} {'n':>7}")
        for m in models:
            mk = base_mk & np.isfinite(df[m].to_numpy())
            if mk.sum() < 30:
                print(f"  {m:>14}  -- insufficient coverage ({mk.sum()}) --")
                continue
            q = float(np.nanmean(loss[m][mk]))
            rmse = float(np.sqrt(np.nanmean((y_log[mk] - df[m].to_numpy()[mk]) ** 2)))
            ic = float(spearmanr(np.exp(0.5 * df[m].to_numpy()[mk]), np.sqrt(rv_true[mk]))[0])
            qmeans[m] = q
            if m == dm_ref:
                dmp, p = "—", None
            else:
                cm = mk & hmask
                _, p = diebold_mariano(loss[m][cm], loss[dm_ref][cm]); dmp = f"{p:.3f}"
            summary[m] = {"QLIKE": q, "RMSE_log": rmse, "IC": ic,
                          "DM_vs_ref_p": None if p is None else float(p), "n": int(mk.sum())}
            print(f"  {m:>14} {q:>8.4f} {rmse:>10.4f} {ic:>7.3f} {dmp:>18} {int(mk.sum()):>7}")
        if qmeans:
            print(f"  best (lowest QLIKE): {min(qmeans, key=qmeans.get)}")
        return summary

    print(f"\nBenchmark — pooled OOS ({len(P)} rows, {fold} walk-forward folds, purge+embargo)")
    print("\n== CORE universe, RAW forecasts (pre-registered: 8 equities + 3 FX) ==")
    core = P[~P["breadth"]]
    summary_core = score_table(core, h1_models)

    print("\n== CORE universe, CALIBRATED forecasts (walk-forward affine recalibration, "
          "+ Ensemble_GR combination) — ALL models forced onto the SAME common row support "
          "(the calibration warm-up drops early rows; comparing on a shared support keeps "
          "this apples-to-apples, see PREREGISTRATION.md Amendment 5) ==")
    summary_core_cal = score_table(core, cal_models, common=True, dm_ref="HAR_cal")

    summary_breadth = summary_breadth_cal = None
    if INCLUDE_BREADTH and P["breadth"].any():
        print(f"\n== CORE-trained models evaluated on CORE+BREADTH (+{sorted(BREADTH_ASSETS)}), RAW; "
              f"breadth rows are a generalization test, not extra training data ==")
        summary_breadth = score_table(P, h1_models)
        print(f"\n== CORE-trained models on CORE+BREADTH, CALIBRATED (common support) ==")
        summary_breadth_cal = score_table(P, cal_models, common=True, dm_ref="HAR_cal")

    # Horizon cuts (h=5, h=22) — HAR, RF+SemiVar, Meridian(+) only
    horizon_summary = {}
    for h, ycol, rvcol in [("h5", "y_5", "rv_next_5"), ("h22", "y_22", "rv_next_22")]:
        hmods = {"HAR": f"HAR_y_5" if h == "h5" else "HAR_y_22",
                 "RF_SemiVar": f"RF_SemiVar_y_5" if h == "h5" else "RF_SemiVar_y_22"}
        if "Meridian_y_5" in P.columns:
            hmods["Meridian"] = f"Meridian_{ycol}"
        if "Meridian+_y_5" in P.columns:
            hmods["Meridian+"] = f"Meridian+_{ycol}"
        print(f"\n== HORIZON {h} (target={ycol}; EWMA/GARCH/TimeMixer are h=1-only, n/a here) ==")
        rv_true = core[rvcol].to_numpy(); y_log = core[ycol].to_numpy()
        rows = {}
        for name, col in hmods.items():
            if col not in core.columns:
                continue
            v = core[col].replace([np.inf, -np.inf], np.nan).to_numpy()
            mk = np.isfinite(v) & np.isfinite(rv_true) & np.isfinite(y_log)
            if mk.sum() < 30:
                continue
            q = float(np.nanmean(qlike(rv_true[mk], np.exp(v[mk]))))
            rmse = float(np.sqrt(np.nanmean((y_log[mk] - v[mk]) ** 2)))
            ic = float(spearmanr(np.exp(0.5 * v[mk]), np.sqrt(rv_true[mk]))[0])
            rows[name] = {"QLIKE": q, "RMSE_log": rmse, "IC": ic, "n": int(mk.sum())}
            print(f"  {name:>12} QLIKE={q:.4f}  RMSE(log)={rmse:.4f}  IC={ic:.3f}  n={int(mk.sum())}")
        horizon_summary[h] = rows

    # Regime cut: VIX terciles on pooled core OOS rows
    regime_summary = {}
    if "vix" in core.columns and core["vix"].notna().sum() > 100:
        terc = core["vix"].quantile([1 / 3, 2 / 3]).to_numpy()
        labels = np.where(core["vix"] <= terc[0], "calm",
                  np.where(core["vix"] <= terc[1], "normal", "stress"))
        core = core.assign(_regime=labels)
        print(f"\n== REGIME cut (VIX terciles: calm ≤{terc[0]:.2f} < normal ≤{terc[1]:.2f} < stress) ==")
        for rg in ["calm", "normal", "stress"]:
            sub = core[core["_regime"] == rg]
            rv_true = sub["rv_next"].to_numpy()
            rows = {}
            for m in h1_models:
                if m not in sub.columns:
                    continue
                v = sub[m].to_numpy()
                mk = np.isfinite(v) & np.isfinite(rv_true)
                if mk.sum() < 30:
                    continue
                q = float(np.nanmean(qlike(rv_true[mk], np.exp(v[mk]))))
                rows[m] = {"QLIKE": q, "n": int(mk.sum())}
            regime_summary[rg] = rows
            print(f"  {rg:>7} (n={len(sub)}): " + ", ".join(f"{m}={v['QLIKE']:.3f}" for m, v in rows.items()))

    # Model Confidence Set (Hansen-Lunde-Nason) on QLIKE losses — CORE universe, common finite rows
    from arch.bootstrap import MCS

    def run_mcs(models, label):
        loss_d = {m: qlike(core["rv_next"].to_numpy(), np.exp(core[m].to_numpy())) for m in models if m in core.columns}
        LM = pd.DataFrame(loss_d).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"\n  MCS input rows (all models finite, {label}): {len(LM)}")
        mcs = MCS(LM, size=0.10, reps=1000, block_size=22, method="R", seed=0); mcs.compute()
        inc = list(mcs.included); pv = mcs.pvalues["Pvalue"].to_dict()
        print(f"  Model Confidence Set (90%, {label}): {{{', '.join(inc)}}}")
        print(f"  MCS p-values: " + ", ".join(f"{m}={pv.get(m, float('nan')):.3f}" for m in LM.columns))
        return inc, {m: float(pv.get(m, float("nan"))) for m in LM.columns}

    included, pvals = run_mcs(h1_models, "core, raw")
    included_cal, pvals_cal = run_mcs(cal_models, "core, calibrated")
    print("\n  QLIKE/RMSE lower = better; IC higher = better. MCS = models statistically")
    print("  indistinguishable from the best at 90% (survivors of the elimination test).")

    RESULTS.mkdir(exist_ok=True)
    payload = {"n_forecasts": int(len(P)), "folds": fold,
               "protocol": f"date-based walk-forward, min_train={MIN_TRAIN_D}d, test={TEST_D}d, "
                           f"embargo={EMBARGO_D}d, meridian_ensemble={ENSEMBLE}, meridian_epochs={EPOCHS}, "
                           f"har_anchor=True (HARNet-style init), post-hoc walk-forward calibration+GR combo",
               "metrics_core_raw": summary_core, "metrics_core_plus_breadth_raw": summary_breadth,
               "metrics_core_calibrated": summary_core_cal, "metrics_core_plus_breadth_calibrated": summary_breadth_cal,
               "horizon_cuts": horizon_summary, "regime_cuts": regime_summary,
               "mcs_90_included_core_raw": included, "mcs_pvalues_core_raw": pvals,
               "mcs_90_included_core_calibrated": included_cal, "mcs_pvalues_core_calibrated": pvals_cal}
    (RESULTS / "benchmark_vol.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(summary_core_cal).T.to_csv(RESULTS / "benchmark_vol.csv")
    print(f"\n  saved → results/benchmark_vol.json, results/benchmark_vol.csv")


if __name__ == "__main__":
    main()
