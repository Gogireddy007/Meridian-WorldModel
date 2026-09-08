"""Conditional EVT tail module (McNeil-Frey 2000, GARCH-EVT) — upgrades the risk pillar
to EXPECTED SHORTFALL (Basel III coherent risk measure), not just VaR.

Two-step: (1) filter returns by OUR vol forecast → standardized residuals z=r/sigma
(separates vol dynamics from tail shape); (2) Peaks-Over-Threshold: fit a Generalized
Pareto Distribution (GPD) to the LEFT-tail exceedances of z, then read VaR_q and
ES_q deep in the tail from the GPD, scaled back by sigma_t. Causal (expanding window,
periodic GPD refit). Backtested (Kupiec + Christoffersen) vs the parametric Student-t;
reports Expected Shortfall — the average loss BEYOND VaR — which the quantile VaR cannot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_wm import kupiec_pof, christoffersen_ind

RESULTS = Path(__file__).resolve().parent.parent / "results"
U_PCT = 0.90          # POT threshold = 90th pct of losses
REFIT = 63            # refit GPD quarterly (causal)


def gpd_var_es(losses, q, u_pct=U_PCT):
    """Fit GPD to left-tail losses; return standardized VaR_q and ES_q (McNeil-Frey)."""
    L = losses[np.isfinite(losses)]
    n = len(L); u = np.quantile(L, u_pct)
    exc = L[L > u] - u
    Nu = len(exc)
    if Nu < 30:
        return np.quantile(L, q), np.nan
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    xi = np.clip(xi, -0.4, 0.9)
    var = u + (beta / xi) * (((n / Nu) * (1 - q)) ** (-xi) - 1) if abs(xi) > 1e-4 else u + beta * np.log((n / Nu) * (1 / (1 - q)))
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi) if xi < 1 else var * 1.5
    return float(var), float(es)


def main():
    wm = pd.read_parquet(RESULTS / "wm3_predictions.parquet"); wm["date"] = pd.to_datetime(wm["date"])
    Q = 0.99                                                # extreme VaR/ES level
    viol_evt, viol_par, es_short = [], [], []
    for a, s in wm.groupby("asset"):
        s = s.sort_values("date").reset_index(drop=True)
        sigma = np.sqrt(np.exp(s["y_pred_log"].to_numpy()))
        r = s["r_next"].to_numpy()
        z = r / np.clip(sigma, 1e-8, None)                  # standardized returns
        n = len(r); ve = np.full(n, np.nan); ee = np.full(n, np.nan)
        stdvar, stdes = None, None
        for t in range(500, n):
            if stdvar is None or (t - 500) % REFIT == 0:
                stdvar, stdes = gpd_var_es(-z[:t], Q)       # losses = -z (left tail)
            ve[t] = -sigma[t] * stdvar                       # conditional VaR (negative return)
            ee[t] = -sigma[t] * stdes                        # conditional ES
        m = ~np.isnan(ve)
        viol_evt.append(r[m] < ve[m])
        # parametric Student-t VaR at Q for comparison
        par = s["scale"].to_numpy() * stats.t.ppf(1 - Q, s["df"].to_numpy())
        viol_par.append((r < par))
        # ES realized check: average loss on violation days vs predicted ES
        vd = m & (r < ve)
        if vd.sum() > 5:
            es_short.append((np.mean(r[vd]), np.mean(ee[vd])))   # (realized avg tail loss, predicted ES)

    print(f"Conditional EVT (GARCH-EVT) tail — {int(Q*100)}% VaR + Expected Shortfall\n")
    for name, v in [("EVT (GPD)", viol_evt), ("parametric Student-t", viol_par)]:
        vv = np.concatenate(v); vv = vv[np.isfinite(vv.astype(float))]
        _, pk, pi = kupiec_pof(vv, 1 - Q); _, pc = christoffersen_ind(vv)
        print(f"  {name:>22}  {int(Q*100)}%-VaR exceed {pi*100:.2f}% (target {(1-Q)*100:.1f}%)  Kupiec p={pk:.3f}  Christoff p={pc:.3f}")
    # Expected Shortfall calibration
    rl = np.array([x[0] for x in es_short]); pe = np.array([x[1] for x in es_short])
    print(f"\n  EXPECTED SHORTFALL (avg loss beyond VaR):")
    print(f"    realized avg tail loss {np.mean(rl)*100:+.2f}%  vs  EVT-predicted ES {np.mean(pe)*100:+.2f}%  "
          f"(ratio {np.mean(rl)/np.mean(pe):.2f}, ~1.0 = well-calibrated)")
    print("\n  ES = the coherent risk measure (Basel III): what you lose ON AVERAGE when the")
    print("  VaR is breached — the number a quantile VaR structurally cannot give you.")


if __name__ == "__main__":
    main()
