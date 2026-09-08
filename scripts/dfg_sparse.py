"""Sparse DFG — adds L1 sparsity to the factor loadings so each latent factor loads on
(mostly) ONE asset class → named, interpretable factors ('rates-vol', 'equity-vol', ...).
Tests whether sparsity yields cleaner class-aligned factors while keeping reconstruction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_broad import load_broad, ASSET_CLASS
from scripts.dfg import rv_panel

K = 4
A_DYN, B_SMOOTH, L1 = 1.0, 0.5, 0.02


class Dyn(nn.Module):
    def __init__(self, k):
        super().__init__(); self.net = nn.Sequential(nn.Linear(k, 16), nn.Tanh(), nn.Linear(16, k))

    def forward(self, z):
        return self.net(z)


def fit_sparse(Y, k=K, iters=2500):
    T, N = Y.shape
    Yt = torch.tensor(Y, dtype=torch.float32)
    Z = nn.Parameter(torch.randn(T, k) * 0.1); W = nn.Parameter(torch.randn(N, k) * 0.1); f = Dyn(k)
    opt = torch.optim.Adam([Z, W] + list(f.parameters()), lr=1e-2)
    for _ in range(iters):
        opt.zero_grad()
        recon = ((Yt - Z @ W.T) ** 2).mean()
        dyn = ((Z[1:] - f(Z[:-1])) ** 2).mean()
        smooth = ((Z[1:] - Z[:-1]) ** 2).mean()
        (recon + A_DYN * dyn + B_SMOOTH * smooth + L1 * W.abs().mean()).backward()
        opt.step()
    return Z.detach().numpy(), W.detach().numpy()


def main():
    d = load_broad()
    names = [n for n in ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
                         "GLD", "USO", "DBC", "EURUSD", "USDJPY", "AUDUSD"] if n in d]
    rv = rv_panel(d, names)
    Y = ((rv - rv.mean()) / rv.std()).to_numpy(np.float32)
    Z, W = fit_sparse(Y)
    r2 = 1 - ((Y - Z @ W.T) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f"sparse DFG (K={K}, L1={L1}) reconstruction R2: {r2:.3f}\n")

    classes = sorted(set(ASSET_CLASS.get(n, "?") for n in names))
    print("  factor → asset-class alignment (|loading| share per class):")
    for j in range(K):
        w = np.abs(W[:, j]); w = w / (w.sum() + 1e-9)
        share = {c: round(float(sum(w[i] for i, n in enumerate(names) if ASSET_CLASS.get(n) == c)), 2) for c in classes}
        dom = max(share, key=share.get)
        top = [names[i] for i in np.argsort(-np.abs(W[:, j]))[:3]]
        print(f"    Factor {j+1}: dominant class = {dom.upper():>9} ({share[dom]:.0%})  | top: {', '.join(top)}  | shares {share}")
    print("\n  (High single-class share = a clean, NAMED interpretable factor.)")


if __name__ == "__main__":
    main()
