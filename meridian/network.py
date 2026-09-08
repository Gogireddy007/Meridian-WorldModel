"""Network shock-propagation — Build #1 from the EWM research synthesis.

Upgrades the single-beta "what-if" to a MULTI-ENTITY network simulation using a VAR +
GENERALIZED impulse response (Pesaran-Shin 1998, order-invariant). Given a shock to one
entity, it propagates through the estimated dynamic network to every other entity —
answering "who gets hit and how hard", including indirect paths, not just one beta.

Two objects, each used for what it is FOR:
  - generalized FEVD  → the CONNECTEDNESS table (variance shares; net transmitter/receiver).
  - cumulative GIRF   → the SHOCK-RESPONSE (return magnitudes; how a δ move in j moves i).

Honest scope: a linear VAR first-order propagation. Real crises are nonlinear (betas and
correlations jump); magnitudes are validated empirically (scripts/network_scenario.py)
before being presented as more than directional.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.api import VAR

warnings.simplefilter("ignore", ValueWarning)      # tz-naive daily index has no freq; fine
EPS = 1e-12


def fit_var(returns: pd.DataFrame, lag: int = 2):
    return VAR(returns).fit(lag)


def girf_cumulative(res, H: int = 10) -> np.ndarray:
    """Cumulative generalized impulse response over horizon H.

    G[i, j] = cumulative response of variable i to a ONE-UNIT (size = 1.0 in the shocked
    variable's own units) generalized shock in variable j. At impact this equals
    Sigma_ij / Sigma_jj (the covariance-implied co-move); later horizons add the VAR's
    dynamic propagation through the whole network.
    """
    Sigma = res.sigma_u.to_numpy() if hasattr(res.sigma_u, "to_numpy") else np.asarray(res.sigma_u)
    N = Sigma.shape[0]
    Theta = res.ma_rep(maxn=H)                       # (H+1, N, N)
    G = np.zeros((N, N))
    for h in range(H + 1):
        # response of all i to unit generalized shock in each j: Theta_h @ Sigma / Sigma_jj
        G += (Theta[h] @ Sigma) / np.diag(Sigma)[None, :]
    return G                                          # column j = responses to a shock in j


def propagate(returns: pd.DataFrame, source: str, shock: float,
              lag: int = 2, H: int = 10) -> pd.Series:
    """Estimate each entity's cumulative response to a `shock` (e.g. -0.05) in `source`."""
    res = fit_var(returns, lag)
    names = list(returns.columns)
    j = names.index(source)
    G = girf_cumulative(res, H)
    # scale unit response to the requested shock size (shock is in source's return units)
    resp = G[:, j] * (shock / (G[j, j] if abs(G[j, j]) > EPS else 1.0))
    return pd.Series(resp, index=names)


def connectedness(returns: pd.DataFrame, lag: int = 2, H: int = 10):
    """Generalized-FEVD net transmitter/receiver table (%) — the connectedness object."""
    res = fit_var(returns, lag)
    Sigma = res.sigma_u.to_numpy() if hasattr(res.sigma_u, "to_numpy") else np.asarray(res.sigma_u)
    N = Sigma.shape[0]
    Theta = res.ma_rep(maxn=H); sig = np.sqrt(np.diag(Sigma))
    num = np.zeros((N, N)); den = np.zeros(N)
    for h in range(H):
        Th = Theta[h]; TS = Th @ Sigma
        for i in range(N):
            den[i] += Th[i] @ Sigma @ Th[i]
            for k in range(N):
                num[i, k] += (TS[i, k] ** 2) / (sig[k] ** 2)
    C = num / den[:, None]
    C = C / C.sum(1, keepdims=True) * 100
    frm = C.sum(1) - np.diag(C); to = C.sum(0) - np.diag(C)
    return pd.DataFrame({"to": to, "from": frm, "net": to - frm}, index=returns.columns)
