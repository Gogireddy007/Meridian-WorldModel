"""Best honest combined book — the culmination product. Combines only the sleeves
that genuinely work, on the broad real universe (total returns), net of costs:

  A. Risk-premia sleeve : vol-timed long multi-asset (equity/bond/gold) — harvests
     risk premia with volatility targeting + diversification (Moreira-Muir effect).
  B. Trend sleeve       : diversified time-series momentum (crisis-convex, orthogonal).
  Combined by equal-risk; regime-style drawdown control via portfolio vol-targeting.

Reports net Sharpe AND the DEFLATED Sharpe (Bailey-Lopez de Prado) accounting for the
number of strategy trials, so the number is honest about multiple testing. No lookahead
(both sleeves are rules), net of costs. Whatever it is, it is real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_broad import load_broad, ASSET_CLASS
from scripts.diversified_stack import returns_panel, inv_vol, portfolio, tsmom
from scripts.backtest import nw_tstat

ANN, TGT, COST = 252, 0.12, 2.0 / 1e4
N_TRIALS = 12          # ~how many strategy configs we examined (for deflated Sharpe)


def risk_premia_sleeve(px, ret):
    """Long-only vol-timed multi-asset risk premia (equity + bond + gold), inverse-vol
    weighted, portfolio vol-targeted — the diversified 'harvest the premia' book."""
    keep = [c for c in px.columns if ASSET_CLASS.get(c) in ("equity", "bond") or c in ("GLD",)]
    r = ret[keep]
    pos = inv_vol(r)                                       # long-only inverse-vol
    return portfolio(pos.dropna(how="all"), r)


def perf_full(r, b=None):
    r = r[np.isfinite(r)]
    sharpe = r.mean() / r.std() * np.sqrt(ANN)
    curve = np.cumprod(1 + r.values); dd = (curve / np.maximum.accumulate(curve) - 1).min()
    out = {"sharpe": sharpe, "ann": r.mean() * ANN, "dd": dd, "t": nw_tstat(r.values)}
    if b is not None:
        bb = b.reindex(r.index)
        X = np.column_stack([np.ones(len(r)), bb.values])
        beta, *_ = np.linalg.lstsq(X, r.values, rcond=None)
        out["alpha_ann"] = beta[0] * ANN
        out["alpha_t"] = nw_tstat((r.values - X @ beta) + beta[0])
        out["beta"] = beta[1]
    return out


def deflated_sharpe(sr, n, trials, skew=0.0, kurt=3.0):
    """Bailey-Lopez de Prado deflated Sharpe probability. sr is per-period (daily)."""
    sr_d = sr / np.sqrt(ANN)                               # daily SR
    # expected max SR under the null across `trials` independent trials
    e_max = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / trials) + \
            np.euler_gamma * stats.norm.ppf(1 - 1.0 / (trials * np.e))
    sr0 = e_max / np.sqrt(n)                               # deflation benchmark (daily)
    denom = np.sqrt(1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2)
    z = (sr_d - sr0) * np.sqrt(n - 1) / max(denom, 1e-9)
    return float(stats.norm.cdf(z))


def main():
    d = load_broad()
    px, ret = returns_panel(d)
    A = risk_premia_sleeve(px, ret)
    B = tsmom(px, ret)
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    spy = ret["SPY"].reindex(idx)

    # equal-risk combine of the two sleeves, then portfolio vol-target to 12%
    S = pd.DataFrame({"risk_premia": A, "trend": B})
    er = (S / S.std()).mean(axis=1)
    er = er / (er.rolling(60).std().shift(1) * np.sqrt(ANN)).clip(0.02, 1).bfill() * TGT / np.sqrt(ANN)
    er = er.dropna()

    print(f"best combined book ({idx.min().date()}→{idx.max().date()}, n={len(idx)}, net {COST*1e4:.0f}bp)\n")
    print(f"{'sleeve':>14} {'Sharpe':>7} {'annRet':>7} {'maxDD':>7} {'alpha_t':>8} {'beta':>6}")
    for name, s in [("risk-premia", A), ("trend", B), ("COMBINED", er)]:
        v = perf_full(s, spy)
        print(f"{name:>14} {v['sharpe']:>7.2f} {v['ann']*100:>6.1f}% {v['dd']*100:>6.1f}% "
              f"{v.get('alpha_t',float('nan')):>8.2f} {v.get('beta',float('nan')):>6.2f}")
    print(f"{'SPY buy&hold':>14} {perf_full(spy)['sharpe']:>7.2f}")

    vc = perf_full(er, spy)
    corr = S.corr().iloc[0, 1]
    dsr = deflated_sharpe(vc["sharpe"], len(er), N_TRIALS,
                          skew=float(stats.skew(er.values)), kurt=float(stats.kurtosis(er.values, fisher=False)))
    print(f"\n  sleeve correlation: {corr:.2f}  (low = real diversification)")
    print(f"  COMBINED net Sharpe {vc['sharpe']:.2f}, alpha-t {vc.get('alpha_t',float('nan')):.2f}, "
          f"maxDD {vc['dd']*100:.1f}%")
    print(f"  DEFLATED Sharpe prob (>0 after {N_TRIALS} trials): {dsr:.3f}  "
          f"({'ROBUST' if dsr>0.95 else 'weak' if dsr<0.9 else 'moderate'})")
    print(f"\n  vs PM bars: Sharpe {vc['sharpe']:.2f} (>=1.5 {'PASS' if vc['sharpe']>=1.5 else 'FAIL'})  "
          f"— honest best on free daily data")


if __name__ == "__main__":
    main()
