"""VALIDATE the network shock-propagation module before trusting its magnitudes.

The research synthesis flagged the shock-transfer arithmetic as OUR extension — so we test
it three ways (its own acceptance criteria) on a fixed liquid basket:

  (a) ORDER-INVARIANCE — the generalized decomposition must be identical under column
      reordering (else it's the fragile Cholesky kind).
  (b) NET-SPILLOVER SANITY — broad-market / equity nodes should be NET TRANSMITTERS,
      bonds/gold NET RECEIVERS, in a stress window.
  (c) OUT-OF-SAMPLE PROPAGATION — on held-out days, fit the VAR on PRIOR data only, inject
      the source's REALIZED shock, and check the predicted cross-asset responses correlate
      with what actually happened that day. This is the real test of the magnitudes.

Ship the network scenario only if (a) holds exactly and (c) is materially positive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_broad import load_broad, ASSET_CLASS
from meridian.network import fit_var, girf_cumulative, connectedness, propagate

BASKET = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
          "GLD", "USO", "DBC", "EURUSD", "USDJPY", "AUDUSD"]
LAG, H = 2, 10


def returns_panel(d, names):
    return pd.DataFrame({n: np.log(d[n]["adjclose"]).diff() for n in names}).dropna()


def main():
    d = load_broad()
    names = [n for n in BASKET if n in d]
    R = returns_panel(d, names)
    R = R[R.index >= "2010-01-01"]
    print(f"network shock-propagation validation — {len(names)} assets, "
          f"{R.index.min().date()}→{R.index.max().date()}\n")

    # (a) ORDER-INVARIANCE
    res = fit_var(R, LAG); G = girf_cumulative(res, H)
    perm = np.random.RandomState(0).permutation(len(names))
    Rp = R.iloc[:, perm]; Gp = girf_cumulative(fit_var(Rp, LAG), H)
    # undo the permutation on Gp and compare
    inv = np.argsort(perm); Gp_un = Gp[np.ix_(inv, inv)]
    max_diff = np.abs(G - Gp_un).max()
    print(f"(a) order-invariance: max |G - permuted G| = {max_diff:.2e}  "
          f"→ {'PASS (order-invariant)' if max_diff < 1e-8 else 'FAIL'}")

    # (b) NET-SPILLOVER SANITY on a stress window (2020 COVID)
    stress = R[(R.index >= "2020-02-01") & (R.index <= "2020-05-31")]
    net = connectedness(stress, LAG, H).sort_values("net", ascending=False)
    top = ", ".join(f"{i}({ASSET_CLASS.get(i,'?')})" for i in net.index[:3])
    bot = ", ".join(f"{i}({ASSET_CLASS.get(i,'?')})" for i in net.index[-3:])
    print(f"(b) COVID-window NET transmitters: {top}")
    print(f"                 NET receivers:    {bot}")
    eq_net = np.mean([net.loc[a, "net"] for a in ["SPY", "QQQ", "IWM"] if a in net.index])
    safe_net = np.mean([net.loc[a, "net"] for a in ["TLT", "IEF", "GLD"] if a in net.index])
    print(f"    equity mean NET {eq_net:+.1f} vs bonds/gold mean NET {safe_net:+.1f} → "
          f"{'PASS (equities transmit, havens absorb)' if eq_net > safe_net else 'FAIL'}")

    # (c) OUT-OF-SAMPLE PROPAGATION: predict cross-asset response to SPY's realized shock
    src = "SPY"; win = 500
    idx = R.columns.get_loc(src)
    big_days = R.index[(R[src].abs() > R[src].std() * 2)]     # large-move days
    big_days = big_days[big_days > R.index[win]]
    corrs, hit = [], []
    for day in big_days:
        t = R.index.get_loc(day)
        past = R.iloc[t - win:t]
        try:
            pred = propagate(past, src, float(R[src].iloc[t]), LAG, H)
        except Exception:
            continue
        actual = R.iloc[t]
        others = [n for n in names if n != src]
        pv = pred[others].to_numpy(); av = actual[others].to_numpy()
        if np.std(pv) < 1e-9:
            continue
        corrs.append(np.corrcoef(pv, av)[0, 1])
        hit.append(np.mean(np.sign(pv) == np.sign(av)))         # direction hit-rate
    corrs = np.array(corrs); hit = np.array(hit)
    print(f"\n(c) out-of-sample propagation on {len(corrs)} large SPY-move days "
          f"(|move|>2σ, VAR fit on prior {win}d only):")
    print(f"    mean cross-asset corr(predicted, realized) = {np.nanmean(corrs):+.3f}")
    print(f"    mean direction hit-rate                    = {np.nanmean(hit)*100:.1f}%")
    ok = np.nanmean(corrs) > 0.15
    print(f"    → {'PASS — propagated magnitudes track realized moves' if ok else 'WEAK — present as directional only'}")

    print("\n  Verdict: network propagation is "
          + ("VALIDATED for directional multi-entity what-ifs "
             if ok and max_diff < 1e-8 else "usable but DIRECTIONAL-ONLY ")
          + "(linear, first-order; crises are nonlinear).")


if __name__ == "__main__":
    main()
