"""Does ONLINE adaptation improve live forecasting vs periodic retraining? (The one
continual-learning claim not yet verified on our data.) Three strategies, same small
model, same purged-forward test, pooled across assets:
  * STATIC        : train once on the initial window, never update.
  * PERIODIC      : retrain from scratch on an expanding window each block (our default).
  * ONLINE (TTA)  : start from static, then after observing each (x_t, y_t) take one SGD
                    step before forecasting t+1 (test-time adaptation / online learning).
Metrics: OOS QLIKE + log-MSE. Honest: whichever wins, wins.
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


def main():
    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"])[FEATS + ["y"]].dropna() for a, o in d["prices"].items()}
    # build one time-sorted pooled panel with a per-row asset id (for per-asset online state we keep it simple: pooled)
    results = {"static": [], "periodic": [], "online": []}
    truth = []
    for a, f in frames.items():
        X = torch.tensor(f[FEATS].to_numpy(np.float32)); y = torch.tensor(f["y"].to_numpy(np.float32))
        n = len(y); split = int(n * 0.5); block = 252
        yt = y[split:].numpy(); truth.append(yt)

        # STATIC
        torch.manual_seed(0); ms = Net(len(FEATS)); fit(ms, X[:split], y[:split], 12)
        with torch.no_grad(): results["static"].append(ms(X[split:]).numpy())

        # PERIODIC (expanding-window retrain each block)
        pp = np.empty(n - split)
        for b0 in range(split, n, block):
            b1 = min(b0 + block, n)
            torch.manual_seed(0); mp = Net(len(FEATS)); fit(mp, X[:b0], y[:b0], 12)
            with torch.no_grad(): pp[b0 - split:b1 - split] = mp(X[b0:b1]).numpy()
        results["periodic"].append(pp)

        # ONLINE (start from static, one SGD step after each observed point)
        torch.manual_seed(0); mo = Net(len(FEATS)); fit(mo, X[:split], y[:split], 12)
        opt = torch.optim.Adam(mo.parameters(), lr=1e-3)
        po = np.empty(n - split)
        for t in range(split, n):
            with torch.no_grad(): po[t - split] = mo(X[t:t+1]).item()      # forecast BEFORE seeing y_t
            opt.zero_grad(); qloss(mo(X[t:t+1]), y[t:t+1]).backward(); opt.step()  # then adapt
        results["online"].append(po)

    y_all = np.concatenate(truth); rv = np.exp(y_all)
    print(f"online vs retrain, pooled OOS ({len(y_all)} forecasts)\n")
    print(f"{'strategy':>10} {'QLIKE':>8} {'log-MSE':>9}")
    for k in ("static", "periodic", "online"):
        p = np.concatenate(results[k])
        print(f"{k:>10} {np.nanmean(qlike(rv, np.exp(p))):>8.4f} {np.nanmean((p - y_all)**2):>9.4f}")
    print("\n  (online < periodic on QLIKE ⇒ live test-time adaptation genuinely helps; else periodic retrain suffices.)")


if __name__ == "__main__":
    main()
