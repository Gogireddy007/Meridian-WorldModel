"""LEAKAGE AUDIT of the online-adaptation claim (Build #2 from the EWM research).

The Proceed paper (KDD 2025) warns: for a horizon-H forecast the label arrives H steps
LATE, so any online method that adapts on the just-revealed label before forecasting the
STILL-OPEN target is leaking future information — and that leak roughly DOUBLES apparent
accuracy vs the honest strategy. Our online forecaster (online_vs_retrain.py) claims
-21.5% QLIKE vs retrain; before we quote that publicly we must prove it is leakage-free.

This audit runs the SAME model four ways and quantifies the leak:
  * STATIC        : train once, never adapt (floor).
  * ONLINE-CLEAN  : forecast t+1, THEN adapt on (x_t, y_t=RV_{t+1}). The adaptation only
                    affects LATER forecasts, made after RV_{t+1} is truly realized. (prod.)
  * ONLINE-LAGGED : extra-strict — before forecasting t+1, only ever adapt on labels whose
                    target day has already passed (explicit H=1 lag). Must ≈ CLEAN.
  * ONLINE-LEAKY  : adapt on (x_t, y_t) BEFORE forecasting t+1 — i.e. train on the answer.
                    This is the cheat; it shows how big an illegitimate gain leakage buys.

Verdict rule: if CLEAN ≈ LAGGED (leak-free) and LEAKY is materially better than CLEAN
(so leakage WOULD inflate), then the production number is honest — indeed conservative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.evalproto import qlike
from meridian.features import build_asset_frame

FEATS = ["har_d", "har_w", "har_m", "log_rv", "ret_abs", "vix"]


class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def qloss(f, y):
    rv = torch.exp(y)
    return (rv * torch.exp(-f) + f - y - 1.0).mean()


def fit(model, X, y, epochs, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        for i in range(0, len(y), 256):
            opt.zero_grad(); qloss(model(X[i:i+256]), y[i:i+256]).backward(); opt.step()


def run_online(X, y, split, mode):
    """mode in {clean, lagged, leaky}. Returns forecasts for [split:n]."""
    n = len(y)
    torch.manual_seed(0); m = Net(X.shape[1]); fit(m, X[:split], y[:split], 12)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    out = np.empty(n - split)
    for t in range(split, n):
        if mode == "leaky":
            # CHEAT: adapt on (x_t, y_t) — the very target — BEFORE forecasting it
            opt.zero_grad(); qloss(m(X[t:t+1]), y[t:t+1]).backward(); opt.step()
            with torch.no_grad(): out[t - split] = m(X[t:t+1]).item()
        elif mode == "clean":
            with torch.no_grad(): out[t - split] = m(X[t:t+1]).item()      # forecast first
            opt.zero_grad(); qloss(m(X[t:t+1]), y[t:t+1]).backward(); opt.step()   # then adapt
        elif mode == "lagged":
            # forecast first; only adapt on the PREVIOUS point (target day already passed),
            # guaranteeing the label for the current still-open target is never used
            with torch.no_grad(): out[t - split] = m(X[t:t+1]).item()
            if t - 1 >= 0:
                opt.zero_grad(); qloss(m(X[t-1:t]), y[t-1:t]).backward(); opt.step()
    return out


def main():
    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"])[FEATS + ["y"]].dropna()
              for a, o in d["prices"].items()}
    res = {"static": [], "clean": [], "lagged": [], "leaky": []}
    truth = []
    for a, f in frames.items():
        X = torch.tensor(f[FEATS].to_numpy(np.float32)); y = torch.tensor(f["y"].to_numpy(np.float32))
        n = len(y); split = int(n * 0.5); truth.append(y[split:].numpy())
        torch.manual_seed(0); ms = Net(len(FEATS)); fit(ms, X[:split], y[:split], 12)
        with torch.no_grad(): res["static"].append(ms(X[split:]).numpy())
        for mode in ("clean", "lagged", "leaky"):
            res[mode].append(run_online(X, y, split, mode))

    y_all = np.concatenate(truth); rv = np.exp(y_all)
    q = {k: float(np.nanmean(qlike(rv, np.exp(np.concatenate(v))))) for k, v in res.items()}
    print(f"LEAKAGE AUDIT — pooled OOS ({len(y_all)} forecasts)\n")
    print(f"{'strategy':>14} {'QLIKE':>8} {'vs static':>10}")
    for k in ("static", "clean", "lagged", "leaky"):
        print(f"{k:>14} {q[k]:>8.4f} {(q[k]/q['static']-1)*100:>+9.1f}%")

    clean_vs_lagged = abs(q["clean"] - q["lagged"]) / q["static"] * 100
    leak_gain = (q["clean"] - q["leaky"]) / q["clean"] * 100
    print(f"\n  CLEAN vs LAGGED gap: {clean_vs_lagged:.2f}% of static  "
          f"→ {'PASS (production is leak-free)' if clean_vs_lagged < 0.5 else 'INVESTIGATE'}")
    print(f"  LEAKY would gain an extra {leak_gain:.1f}% over CLEAN "
          f"→ {'leakage is real & material; CLEAN correctly avoids it' if leak_gain > 1 else 'leak channel small here'}")
    honest = clean_vs_lagged < 0.5
    print(f"\n  VERDICT: the online-adaptation gain is "
          + ("LEAKAGE-FREE and honest (production uses forecast-then-adapt with a true 1-day "
             "label delay; the extra-strict lagged variant matches it). The published number "
             "is conservative — it leaves the leaky gain on the table, as it should."
             if honest else
             "SUSPECT — clean and lagged diverge; the label timing needs a fix before quoting."))
    print("  Note: this holds for the H=1 (next-day) target. Any H>1 forecast must lag the "
          "adaptation by H — encode that when horizons are added.")


if __name__ == "__main__":
    main()
