"""INTERPRETABILITY AUDIT of the switching-regime module (Meridian-WM).

"Interpretable from foundations" is a claim that must be TESTED, not asserted. Using the
already-saved regime assignments (results/wm3_predictions.parquet — no recompute), this
audits the four things that make a latent regime genuinely interpretable rather than a
relabeling artifact:

  1. ECONOMIC MEANING  — does each latent state map to a real market condition (vol, return)?
  2. FAITHFULNESS      — do the states actually SEPARATE the volatility distribution
                         (eta^2 = between-state variance share)? A state that doesn't move
                         the vol distribution explains nothing.
  3. PERSISTENCE       — is the transition matrix sticky (regimes persist), as real market
                         regimes do — not flickering noise?
  4. IDENTIFIABILITY   — the label-switching test: is the vol-ORDERING of the states stable
                         ACROSS assets? If "state 2 = high vol" everywhere, the labels carry
                         a consistent, auditable meaning (not per-asset relabeling).

This is the rigorous bar for calling the regime module interpretable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
RESULTS = Path(__file__).resolve().parent.parent / "results"
ANN = 252


def dwell(seq):
    """mean consecutive run length per state."""
    seq = np.asarray(seq); runs = {}
    if len(seq) == 0:
        return {}
    cur, ln = seq[0], 1
    lens = {}
    for s in seq[1:]:
        if s == cur:
            ln += 1
        else:
            lens.setdefault(cur, []).append(ln); cur, ln = s, 1
    lens.setdefault(cur, []).append(ln)
    return {k: float(np.mean(v)) for k, v in lens.items()}


def main():
    d = pd.read_parquet(RESULTS / "wm3_predictions.parquet")
    d = d.sort_values(["asset", "date"])
    d["ann_vol"] = np.sqrt(np.exp(d["y_true_log"]) * ANN) * 100    # realized annualized vol %
    states = sorted(d["regime"].unique())
    print(f"Regime interpretability audit — {d['asset'].nunique()} assets, "
          f"{d['date'].min().date()}→{d['date'].max().date()}, {len(states)} latent states\n")

    # ---- 1. ECONOMIC MEANING ----
    print("[1] ECONOMIC MEANING (pooled) — does each latent state map to a market condition?")
    print(f"    {'state':>6} {'share':>7} {'ann vol':>9} {'mean ret':>9} {'|ret|':>7} {'dwell(d)':>9}")
    for s in states:
        sub = d[d["regime"] == s]
        # dwell per asset then averaged
        dv = np.mean([dwell(g["regime"].tolist()).get(s, np.nan)
                      for _, g in d.groupby("asset") if s in g["regime"].values])
        print(f"    {s:>6} {len(sub)/len(d)*100:>6.1f}% {sub['ann_vol'].mean():>8.1f}% "
              f"{sub['r_next'].mean()*100:>+8.2f}% {sub['r_next'].abs().mean()*100:>6.2f}% {dv:>9.1f}")

    # ---- 2. FAITHFULNESS: eta^2 (between-state share of log-vol variance) ----
    y = d["y_true_log"].to_numpy(); grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_bet = sum(len(sub) * (sub["y_true_log"].mean() - grand) ** 2 for _, sub in d.groupby("regime"))
    eta2 = ss_bet / ss_tot
    print(f"\n[2] FAITHFULNESS: regimes explain eta^2 = {eta2*100:.1f}% of log-vol variance "
          f"→ {'STRONG (states genuinely separate vol)' if eta2 > 0.2 else 'WEAK (states barely move vol)'}")

    # ---- 3. PERSISTENCE: pooled transition matrix ----
    print("\n[3] PERSISTENCE — transition matrix P(next | current), pooled across assets:")
    T = np.zeros((len(states), len(states)))
    idx = {s: i for i, s in enumerate(states)}
    for _, g in d.groupby("asset"):
        r = g["regime"].to_numpy()
        for a, b in zip(r[:-1], r[1:]):
            T[idx[a], idx[b]] += 1
    Tn = T / T.sum(1, keepdims=True)
    hdr = "        " + " ".join(f"→{s:>5}" for s in states)
    print(hdr)
    for s in states:
        print(f"    {s:>3} " + " ".join(f"{Tn[idx[s], idx[t]]*100:>5.0f}%" for t in states))
    diag = np.mean(np.diag(Tn))
    print(f"    mean self-persistence (diagonal) = {diag*100:.0f}% "
          f"→ {'sticky, regime-like' if diag > 0.7 else 'flickery'}")

    # ---- 4. IDENTIFIABILITY: is the vol-ordering of states stable across assets? ----
    print("\n[4] IDENTIFIABILITY (label-switching test) — vol-rank of each state, per asset:")
    order_rows = []
    for a, g in d.groupby("asset"):
        means = g.groupby("regime")["ann_vol"].mean()
        rank = means.rank().astype(int)                    # 1 = lowest vol
        order_rows.append({s: int(rank.get(s, -1)) for s in states})
    od = pd.DataFrame(order_rows, index=[a for a, _ in d.groupby("asset")])
    # consistency: fraction of assets whose ranking matches the pooled modal ranking
    modal = od.mode().iloc[0]
    consistent = (od == modal).all(axis=1).mean()
    print(f"    modal vol-ranking of states (1=calmest): "
          + ", ".join(f"state {s}=rank {int(modal[s])}" for s in states))
    print(f"    {consistent*100:.0f}% of assets share this exact ordering "
          f"→ {'IDENTIFIABLE (labels mean the same thing everywhere)' if consistent > 0.7 else 'labels drift across assets'}")

    print("\n  VERDICT: the regime module is interpretable to the extent that states carry a")
    print("  consistent economic meaning (1), faithfully separate volatility (2), persist (3),")
    print("  and keep a stable ordering across assets (4). Numbers above are the evidence.")


if __name__ == "__main__":
    main()
