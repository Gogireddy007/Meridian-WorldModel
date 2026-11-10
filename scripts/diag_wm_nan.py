"""Reproduce the V2 primary failure (PREREGISTRATION.md Amendment 8a): MeridianWM as written,
trained on walk-forward fold 7 (seed 700), produces a NaN SIGReg loss and NaN weights.

    python scripts/diag_wm_nan.py      # prints per-step losses until the first non-finite output
"""
import sys, time, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import scripts.benchmark_wm as bw, scripts.benchmark_vol as bv
from meridian.meridian_wm import MeridianWM, WMConfig
from meridian.windows import FEATURES, train_scaler
torch.set_num_threads(4)
R, _, _, _, frames, asset_win = bw.build()
f = bw.fold_grid(R)[6]
core = {a: v for a, v in asset_win.items() if a in bv.CORE_ASSETS}
mean, std = train_scaler(frames, core, f["cutoff"])
W = bw.wm_windows(asset_win, frames, mean, std)
tr = {k: [] for k in ("ctx", "fut", "y", "r_next")}
for a, w in W.items():
    if a in bv.CORE_ASSETS:
        m = (w["dates"] < f["cutoff"]) & (w["end_1"] < np.datetime64(f["test_start"]))
        for k in tr: tr[k].append(w[k][m])
tr = {k: np.concatenate(v) for k, v in tr.items()}
print("fold", f["fold"], "rows", len(tr["y"]), "finite inputs:", {k: bool(np.isfinite(v).all()) for k, v in tr.items()},
      "abs max ctx", float(np.abs(tr["ctx"]).max()), "y range", float(tr["y"].min()), float(tr["y"].max()), flush=True)
cfg = WMConfig(n_features=len(FEATURES), window=bv.L_NEURAL, seed=f["fold"] * 100 + 0)
m = MeridianWM(cfg); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
T = {k: torch.tensor(tr[k]) for k in tr}; n = len(T["y"]); g = torch.Generator().manual_seed(cfg.seed)
step = 0
for ep in range(8):
    perm = torch.randperm(n, generator=g)
    for i in range(0, n, 256):
        b = perm[i:i + 256]
        out = m.forward(T["ctx"][b], T["fut"][b])
        bad = {k: int((~torch.isfinite(v.float())).sum()) for k, v in out.items() if torch.is_tensor(v) and v.dtype.is_floating_point}
        if any(bad.values()):
            pn = {n_: float(p.abs().max()) for n_, p in m.named_parameters() if not torch.isfinite(p).all() or p.abs().max() > 1e3}
            print(f"NON-FINITE at epoch {ep} step {step}: {bad}; params non-finite/huge: {list(pn.items())[:8]}", flush=True)
            raise SystemExit
        opt.zero_grad(); loss, logs = m.loss({"x_ctx": T["ctx"][b], "x_fut": T["fut"][b], "y": T["y"][b], "r_next": T["r_next"][b]})
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if step % 200 == 0 or not torch.isfinite(gn):
            print(ep, step, {k: round(v, 3) for k, v in logs.items()}, "gradnorm", float(gn), flush=True)
        opt.step(); m.update_target(); step += 1
print("finished without NaN")
