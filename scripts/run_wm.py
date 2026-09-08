"""Train + evaluate Meridian-WM under the pre-registered walk-forward.

Outputs results/wm_predictions.parquet with OOS log-RV forecasts (QLIKE),
switching-regime labels (persistent by construction), and the Student-t log-df.
Regime label comes straight from the switching posterior — no post-hoc HMM.
"""
from __future__ import annotations

import dataclasses
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
from meridian.meridian_wm import MeridianWM, WMConfig
from meridian.windows import FEATURES, asset_matrix, build_windows, train_scaler

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
L, TEST_BLOCK, EMBARGO, FIRST_TEST = 32, 252, 22, "2012-01-01"
EPOCHS, BATCH, LR = 8, 256, 1e-3
ENSEMBLE = int(os.environ.get("MERIDIAN_ENSEMBLE", "1"))
OUT_NAME = os.environ.get("MERIDIAN_OUT", "wm_predictions.parquet")


def to_dev(a):
    return torch.tensor(a, device=DEVICE)


def train_block(cfg, tr, epochs=EPOCHS):
    model = MeridianWM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ctx, fut, y = to_dev(tr["ctx"]), to_dev(tr["fut"]), to_dev(tr["y"])
    r = to_dev(tr["r_next"])
    n = len(y); g = torch.Generator().manual_seed(cfg.seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss, _ = model.loss({"x_ctx": ctx[b], "x_fut": fut[b], "y": y[b], "r_next": r[b]})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); model.update_target()
    return model


@torch.no_grad()
def predict_block(models, w):
    ctx, fut = to_dev(w["ctx"]), to_dev(w["fut"])
    vols, dfs, scales, regs = [], [], [], []
    for m in models:
        m.eval(); out = m.forward(ctx, fut)
        vols.append(out["vol"].cpu().numpy())
        dfs.append(out["df"].cpu().numpy())
        scales.append(out["scale"].cpu().numpy())
        regs.append(out["regime"].cpu().numpy())
    return np.mean(vols, 0), np.mean(dfs, 0), np.mean(scales, 0), regs[0]


def main():
    t0 = time.time()
    d = load_all(); macro = d["macro"]
    frames = {a: build_asset_frame(o, macro) for a, o in d["prices"].items()}
    asset_data = {a: asset_matrix(f) for a, f in frames.items()}
    all_dates = pd.DatetimeIndex(np.sort(np.unique(np.concatenate(
        [dt.values for dt, _, _ in asset_data.values()]))))
    pos = all_dates.searchsorted(pd.Timestamp(FIRST_TEST))

    cfg = WMConfig(n_features=len(FEATURES), window=L, seed=0)
    print(f"Meridian-WM  ensemble={ENSEMBLE}  device={DEVICE}", flush=True)
    records, bi = [], 0
    while pos < len(all_dates):
        test_start = all_dates[pos]
        end_pos = min(pos + TEST_BLOCK, len(all_dates) - 1)
        test_end = all_dates[end_pos]
        train_end = all_dates[max(pos - EMBARGO, 0)]
        mean, std = train_scaler(frames, asset_data, train_end)

        keys = ("ctx", "fut", "y", "r_next")
        tr_parts, te_windows = [], []
        for a, (dates, X, y, _aux) in asset_data.items():
            w = build_windows(dates, X, y, L, mean, std)
            if w is None:
                continue
            dt = pd.DatetimeIndex(w["dates"])
            # next-day return aligned to each anchor (for the distributional/VaR head)
            w["r_next"] = np.nan_to_num(
                frames[a]["r_next"].reindex(dt).to_numpy(np.float32))
            trm = dt < train_end; tem = (dt >= test_start) & (dt < test_end)
            if trm.any():
                tr_parts.append({k: w[k][trm] for k in keys})
            if tem.any():
                te_windows.append((a, dt[tem], {k: w[k][tem] for k in keys}))
        if not tr_parts or not te_windows:
            pos = end_pos + 1 if end_pos + 1 > pos else pos + TEST_BLOCK
            continue
        tr = {k: np.concatenate([p[k] for p in tr_parts]) for k in keys}
        models = [train_block(dataclasses.replace(cfg, seed=cfg.seed + s), tr)
                  for s in range(ENSEMBLE)]
        for a, dts, w in te_windows:
            vol, df, scale, reg = predict_block(models, w)
            records.append(pd.DataFrame({
                "date": dts, "asset": a, "model": "Meridian-WM",
                "y_true_log": w["y"], "y_pred_log": vol, "logvar_bias": 0.0,
                "regime": reg, "df": df, "scale": scale, "r_next": w["r_next"]}))
        bi += 1
        print(f"  block {bi:2d} {test_start.date()}->{test_end.date()} "
              f"n={len(tr['y']):6d} [{time.time()-t0:5.0f}s]", flush=True)
        pos = end_pos + 1 if end_pos + 1 > pos else pos + TEST_BLOCK

    preds = pd.concat(records, ignore_index=True)
    preds.to_parquet(RESULTS / OUT_NAME)
    print(f"\nsaved {len(preds)} rows -> {OUT_NAME} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
