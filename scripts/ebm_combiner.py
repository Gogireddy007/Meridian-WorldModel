"""Build #1 — a GLASS-BOX additive combiner/predictor (GA1M), replacing the opaque MLP.

The frontier research ranked this #1: the combiner / online-predictor is the only seam where
a black box can hide. An Explainable Boosting Machine / GA2M is interpretable BY CONSTRUCTION —
the forecast is a sum of per-feature shape functions f_j(x_j), each a readable curve — and on
tabular data at our scale it matches full-complexity models with ZERO accuracy premium lost
(Grinsztajn 2022). We implement a dependency-free additive GAM via backfitting with per-feature
MONOTONIC shape functions (isotonic), where volatility theory fixes the sign (persistence &
implied-vol → next vol is non-decreasing in every feature).

Acceptance (pre-registered):
  (a) glass-box GAM matches the black-box GBDT OOS QLIKE within noise;
  (b) every shape function is monotone as theory dictates and human-readable;
  (c) ADEBAYO SANITY CHECK — shuffle the labels, refit: the shape functions must COLLAPSE
      to ~flat (attribution moves). A shape that survives label-shuffling is an artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.evalproto import qlike

FEATS = ["har_d", "har_w", "har_m", "log_rv", "ret_abs", "vix"]
NBINS = 20


class Shape:
    """A monotone (increasing) piecewise-constant shape function f_j(x_j)."""
    def __init__(self, edges, vals):
        self.edges, self.vals = edges, vals
    def __call__(self, x):
        idx = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.vals) - 1)
        return self.vals[idx]


def fit_shape(x, r):
    """bin x by quantiles, isotonic-increasing mean partial-residual per bin."""
    qs = np.quantile(x, np.linspace(0, 1, NBINS + 1))
    edges = np.unique(qs)
    if len(edges) < 3:
        return Shape(np.array([x.min(), x.max()]), np.array([r.mean(), r.mean()]))
    binidx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    centers, means = [], []
    for b in range(len(edges) - 1):
        m = binidx == b
        if m.sum() > 0:
            centers.append(0.5 * (edges[b] + edges[b + 1])); means.append(r[m].mean())
    centers, means = np.array(centers), np.array(means)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(centers, means)
    vals = iso.predict(0.5 * (edges[:-1] + edges[1:]))
    return Shape(edges[:-1], vals)


class GAM:
    def __init__(self, feats): self.feats = feats
    def fit(self, X, y, passes=10):
        self.b0 = y.mean(); self.shapes = {j: None for j in range(X.shape[1])}
        f = {j: np.zeros(len(y)) for j in range(X.shape[1])}
        for _ in range(passes):
            for j in range(X.shape[1]):
                partial = y - self.b0 - sum(f[k] for k in f if k != j)
                s = fit_shape(X[:, j], partial); self.shapes[j] = s
                f[j] = s(X[:, j]) - s(X[:, j]).mean()      # center each shape
        return self
    def predict(self, X):
        return self.b0 + sum(self.shapes[j](X[:, j]) - self.shapes[j](X[:, j]).mean()
                             for j in range(X.shape[1]))
    def shape_range(self, X, j):                            # readable effect size
        v = self.shapes[j](X[:, j]); return float(v.max() - v.min())


def oos_qlike(pred, y):
    jb = 0.0
    return float(np.nanmean(qlike(np.exp(y), np.exp(pred + jb))))


def main():
    d = load_all()
    frames = {a: build_asset_frame(o, d["macro"])[FEATS + ["y"]].dropna()
              for a, o in d["prices"].items()}
    F = pd.concat(frames.values())
    X = F[FEATS].to_numpy(); y = F["y"].to_numpy()
    n = len(y); k = int(n * 0.6)
    Xtr, ytr, Xte, yte = X[:k], y[:k], X[k:], y[k:]

    # black-box baseline (monotone GBDT) vs glass-box GAM vs linear
    lin = np.linalg.lstsq(np.c_[np.ones(k), Xtr], ytr, rcond=None)[0]
    p_lin = np.c_[np.ones(len(yte)), Xte] @ lin
    gbdt = HistGradientBoostingRegressor(max_iter=300, monotonic_cst=[1] * len(FEATS),
                                         learning_rate=0.05).fit(Xtr, ytr)
    p_gb = gbdt.predict(Xte)
    gam = GAM(FEATS).fit(Xtr, ytr)
    p_gam = gam.predict(Xte)

    print("Build #1 — glass-box additive combiner (GAM) vs black box\n")
    print(f"  {'model':>16} {'OOS QLIKE':>10}")
    for nm, p in [("linear", p_lin), ("GBDT (monotone)", p_gb), ("GAM (glass-box)", p_gam)]:
        print(f"  {nm:>16} {oos_qlike(p, yte):>10.4f}")

    print("\n  (b) shape functions — effect size (monotone increasing by construction):")
    for j, f in enumerate(FEATS):
        print(f"      {f:>8}: Δforecast over its range = {gam.shape_range(Xtr, j):+.3f} (log-var)")

    # (c) Adebayo label-shuffle sanity check
    rng = np.random.RandomState(0)
    gam_sh = GAM(FEATS).fit(Xtr, rng.permutation(ytr))
    real_eff = np.mean([gam.shape_range(Xtr, j) for j in range(len(FEATS))])
    shuf_eff = np.mean([gam_sh.shape_range(Xtr, j) for j in range(len(FEATS))])
    print(f"\n  (c) ADEBAYO sanity check: mean shape effect  real {real_eff:.3f}  vs  "
          f"label-shuffled {shuf_eff:.3f}")
    print(f"      → {'PASS — shapes collapse under shuffling (reflect real structure)' if shuf_eff < real_eff*0.25 else 'FAIL — shapes survive shuffling (artifact)'}")
    print("\n  Verdict: if GAM QLIKE ≈ GBDT and shapes are monotone + collapse under shuffle,")
    print("  the glass-box combiner is adopted — same accuracy, fully interpretable from foundations.")


if __name__ == "__main__":
    main()
