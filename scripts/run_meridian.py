"""Train + evaluate the Meridian core under the pre-registered walk-forward.

Pooled across assets, expanding window, refit each test block, purge+embargo
at every boundary. Outputs results/meridian_predictions.parquet with OOS log-RV
forecasts, JEPA surprise energy, and regime assignments.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.model import Meridian, MeridianConfig
from meridian.windows import FEATURES, asset_matrix, build_windows, train_scaler

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

L = 32
TEST_BLOCK = 252
EMBARGO = 22
FIRST_TEST = "2012-01-01"
EPOCHS = 8
BATCH = 256
LR = 1e-3

LOSS_MODE = os.environ.get("MERIDIAN_LOSS", "mse")            # "mse" | "qlike"
OUT_NAME = os.environ.get("MERIDIAN_OUT", "meridian_predictions.parquet")
BELIEF_NAME = os.environ.get("MERIDIAN_BELIEF", "meridian_belief.npy")
ENSEMBLE = int(os.environ.get("MERIDIAN_ENSEMBLE", "1"))      # # seeds to average
HOLDOUT = set(a for a in os.environ.get("MERIDIAN_HOLDOUT", "").split(",") if a)  # assets excluded from TRAIN; only these predicted


def to_dev(a):
    return torch.tensor(a, device=DEVICE)


def train_block(cfg, tr, epochs=EPOCHS):
    model = Meridian(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n = len(tr["y"])
    ctx, fut, y = to_dev(tr["ctx"]), to_dev(tr["fut"]), to_dev(tr["y"])
    g = torch.Generator().manual_seed(cfg.seed)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            batch = {"x_ctx": ctx[b], "x_fut": fut[b], "y": y[b]}
            opt.zero_grad()
            loss, _ = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
    return model


def train_ensemble(cfg, tr, k):
    """Train k independently-seeded models (seeds cfg.seed .. cfg.seed+k-1)."""
    import dataclasses
    models = []
    for s in range(k):
        models.append(train_block(dataclasses.replace(cfg, seed=cfg.seed + s), tr))
    return models


@torch.no_grad()
def predict_block(models, w):
    """Average log-variance / energy across an ensemble; belief from seed 0.

    Returns (vol, energy, belief, vol_tgt) where vol_tgt is the EMA-target-encoder
    read-off (CF-JEPA test) or None.
    """
    if not isinstance(models, list):
        models = [models]
    ctx, fut = to_dev(w["ctx"]), to_dev(w["fut"])
    vols, ens, tgts, belief = [], [], [], None
    for i, m in enumerate(models):
        m.eval()
        out = m.forward(ctx, fut)
        vols.append(out["vol"].cpu().numpy())
        ens.append(out["energy"].cpu().numpy())
        if "vol_tgt" in out:
            tgts.append(out["vol_tgt"].cpu().numpy())
        if i == 0:
            belief = out["h"].cpu().numpy().astype(np.float32)
    vol_tgt = np.mean(tgts, axis=0) if tgts else None
    return (np.mean(vols, axis=0), np.mean(ens, axis=0), belief, vol_tgt)


def main():
    t0 = time.time()
    d = load_all()
    macro = d["macro"]
    frames = {a: build_asset_frame(ohlc, macro) for a, ohlc in d["prices"].items()}
    asset_data = {a: asset_matrix(f) for a, f in frames.items()}

    # global timeline of test-block boundaries
    all_dates = np.sort(np.unique(np.concatenate(
        [dts.values for dts, _, _ in asset_data.values()])))
    all_dates = pd.DatetimeIndex(all_dates)
    start = all_dates.searchsorted(pd.Timestamp(FIRST_TEST))

    cfg = MeridianConfig(n_features=len(FEATURES), window=L, seed=0, loss_mode=LOSS_MODE)
    if "MERIDIAN_LJEPA" in os.environ:
        cfg.lambda_jepa = float(os.environ["MERIDIAN_LJEPA"])
    if "MERIDIAN_LSIG" in os.environ:
        cfg.lambda_sig = float(os.environ["MERIDIAN_LSIG"])
    cfg.dual_vol = os.environ.get("MERIDIAN_DUALVOL", "") == "1"
    cfg.core_type = os.environ.get("MERIDIAN_CORE", "ssm")
    print(f"loss_mode={LOSS_MODE}  ensemble={ENSEMBLE}  holdout={sorted(HOLDOUT) or 'none'}  "
          f"lambda_jepa={cfg.lambda_jepa}  lambda_sig={cfg.lambda_sig}  out={OUT_NAME}", flush=True)
    records = []
    belief_store: list = []
    block_i = 0
    pos = start
    while pos < len(all_dates):
        test_start = all_dates[pos]
        end_pos = min(pos + TEST_BLOCK, len(all_dates) - 1)
        test_end = all_dates[end_pos]
        train_end = all_dates[max(pos - EMBARGO, 0)]      # purge+embargo boundary

        mean, std = train_scaler(frames, asset_data, train_end)

        tr_parts, te_parts_meta = [], []
        te_windows = []
        for a, (dates, X, y, _aux) in asset_data.items():
            w = build_windows(dates, X, y, L, mean, std)
            if w is None:
                continue
            dt = pd.DatetimeIndex(w["dates"])
            tr_m = dt < train_end
            te_m = (dt >= test_start) & (dt < test_end)
            if tr_m.any() and a not in HOLDOUT:            # held-out assets never train
                tr_parts.append({k: w[k][tr_m] for k in ("ctx", "fut", "y")})
            predict_this = (a in HOLDOUT) if HOLDOUT else True
            if te_m.any() and predict_this:
                te_windows.append((a, dt[te_m],
                                   {k: w[k][te_m] for k in ("ctx", "fut", "y")}))
        if not tr_parts or not te_windows:
            pos = end_pos + 1 if end_pos + 1 > pos else pos + TEST_BLOCK
            continue

        tr = {k: np.concatenate([p[k] for p in tr_parts]) for k in ("ctx", "fut", "y")}
        models = train_ensemble(cfg, tr, ENSEMBLE)

        # Jensen bias for variance conversion (MSE mode only; qlike head already
        # outputs log-variance so no conversion bias is needed).
        if LOSS_MODE == "qlike":
            bias = 0.0
        else:
            with torch.no_grad():
                preds = [m.forward(to_dev(tr["ctx"]))["vol"].cpu().numpy() for m in models]
            bias = 0.5 * float(np.var(tr["y"] - np.mean(preds, axis=0)))

        for a, dts, w in te_windows:
            vol, energy, belief, vol_tgt = predict_block(models, w)
            rec = pd.DataFrame({
                "date": dts, "asset": a, "model": "Meridian",
                "y_true_log": w["y"], "y_pred_log": vol,
                "logvar_bias": bias, "energy": energy,
            })
            if vol_tgt is not None:
                rec["y_pred_tgt"] = vol_tgt
            rec["_belief_row"] = np.arange(len(belief_store), len(belief_store) + len(rec))
            belief_store.extend(belief)
            records.append(rec)
        block_i += 1
        print(f"  block {block_i:2d}  {test_start.date()}->{test_end.date()}  "
              f"train_n={len(tr['y']):6d}  bias={bias:.3f}  "
              f"[{time.time()-t0:5.0f}s]", flush=True)
        pos = end_pos + 1 if end_pos + 1 > pos else pos + TEST_BLOCK

    preds = pd.concat(records, ignore_index=True)
    preds.to_parquet(RESULTS / OUT_NAME)
    np.save(RESULTS / BELIEF_NAME, np.asarray(belief_store, np.float32))
    print(f"\nsaved {len(preds)} OOS rows -> {OUT_NAME} "
          f"belief {np.asarray(belief_store).shape} ({time.time()-t0:.0f}s, device={DEVICE})")


if __name__ == "__main__":
    main()
