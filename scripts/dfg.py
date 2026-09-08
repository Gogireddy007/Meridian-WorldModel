"""Dynamic Factor Graph (Mirowski & LeCun 2009) — minimal faithful implementation,
tested as the interpretable 'common latent market-state' module of the world model.

Energy(Z,W,f) = ||Y - Z W^T||^2  +  a*||Z_t - f(Z_{t-1})||^2  +  b*||Z_t - Z_{t-1}||^2
  observation factor (decoder)      dynamical factor            smoothing regularizer
Latent Z is INFERRED by energy minimization (gradient descent); parameters W,f trained
jointly (deterministic EM relaxation). Applied to the cross-asset log-RV panel to extract
K interpretable common factors. We check: (1) interpretable loadings by asset class,
(2) energy = systemic-surprise signal spiking in crises, (3) does the latent state
improve vol prediction (feed Z into a forecaster).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_broad import load_broad, ASSET_CLASS

K = 3           # number of latent factors
A_DYN, B_SMOOTH = 1.0, 0.5


def rv_panel(d, names):
    out = {}
    for n in names:
        r = np.log(d[n]["adjclose"]).diff()
        out[n] = np.log((r ** 2).rolling(5).mean().clip(lower=1e-10))
    return pd.DataFrame(out).dropna()


class Dynamics(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, 16), nn.Tanh(), nn.Linear(16, k))

    def forward(self, z):                                  # z: (T-1,k) -> predict next
        return self.net(z)


def fit_dfg(Y, k=K, iters=1500):
    T, N = Y.shape
    Yt = torch.tensor(Y, dtype=torch.float32)
    Z = nn.Parameter(torch.zeros(T, k))                    # latent state (inferred)
    W = nn.Parameter(torch.randn(N, k) * 0.1)              # observation loadings
    f = Dynamics(k)
    opt = torch.optim.Adam([Z, W] + list(f.parameters()), lr=1e-2)
    for _ in range(iters):
        opt.zero_grad()
        recon = ((Yt - Z @ W.T) ** 2).mean()
        dyn = ((Z[1:] - f(Z[:-1])) ** 2).mean()
        smooth = ((Z[1:] - Z[:-1]) ** 2).mean()
        energy = recon + A_DYN * dyn + B_SMOOTH * smooth
        energy.backward()
        opt.step()
    with torch.no_grad():
        per_t_energy = ((Yt - Z @ W.T) ** 2).mean(1) + A_DYN * torch.cat(
            [torch.zeros(1), ((Z[1:] - f(Z[:-1])) ** 2).mean(1)])
    return Z.detach().numpy(), W.detach().numpy(), per_t_energy.numpy(), f


def main():
    d = load_broad()
    names = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
             "GLD", "USO", "DBC", "EURUSD", "USDJPY", "AUDUSD"]
    names = [n for n in names if n in d]
    rv = rv_panel(d, names)
    Y = ((rv - rv.mean()) / rv.std()).to_numpy(np.float32)
    print(f"DFG on {len(names)}-asset log-RV panel, {rv.index.min().date()}→{rv.index.max().date()}, K={K}\n")

    Z, W, energy, f = fit_dfg(Y)
    recon_r2 = 1 - ((Y - Z @ W.T) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f"  reconstruction R2 (K={K} factors explain the {len(names)}-asset panel): {recon_r2:.3f}")

    # (1) interpretability: factor-1 loadings by asset class
    print("\n  Factor-1 loadings (the dominant common factor) by asset:")
    order = np.argsort(-np.abs(W[:, 0]))
    for i in order:
        print(f"    {names[i]:>7} {ASSET_CLASS.get(names[i],'?'):>10} {W[i,0]:+.2f}")

    # (2) energy = systemic surprise: does it spike in crises?
    en = pd.Series(energy, index=rv.index)
    enz = (en - en.rolling(252).mean()) / (en.rolling(252).std() + 1e-9)
    top = enz.nlargest(6)
    print("\n  Top systemic-surprise spikes (DFG energy z-score) — should be real crises:")
    for dt, v in top.items():
        print(f"    {dt.date()}  {v:+.1f}sigma")

    # (3) does the latent state improve vol prediction? add Z as features to predict SPY next-day RV
    from sklearn.ensemble import RandomForestRegressor
    from meridian.evalproto import qlike
    spy_rv = rv["SPY"]; y = spy_rv.shift(-1).to_numpy()
    base_X = np.column_stack([spy_rv.to_numpy(), spy_rv.rolling(5).mean().to_numpy(), spy_rv.rolling(22).mean().to_numpy()])
    full_X = np.column_stack([base_X, Z])
    m = np.isfinite(y) & np.isfinite(base_X).all(1) & np.isfinite(full_X).all(1)
    n = m.sum(); tr = slice(0, int(n*0.6))
    yb = y[m]; Xb = base_X[m]; Xf = full_X[m]
    def oos_qlike(X):
        rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=20, n_jobs=-1, random_state=0)
        rf.fit(X[tr], yb[tr]); p = rf.predict(X[int(n*0.6):])
        return float(np.nanmean(qlike(np.exp(yb[int(n*0.6):]), np.exp(p))))
    qb, qf = oos_qlike(Xb), oos_qlike(Xf)
    print(f"\n  SPY next-day RV forecast (HAR features): QLIKE {qb:.4f}")
    print(f"  + DFG latent factors as features:        QLIKE {qf:.4f}  (lift {(qb-qf)/qb*100:+.1f}%)")


if __name__ == "__main__":
    main()
