"""Regime analysis: HMM baseline vs Meridian belief-state regimes.

Pre-registered regime criteria (PREREGISTRATION.md):
  * persistence = mean dwell time (avg consecutive run length of a state)
  * economic check = does conditioning vol on the regime lower OOS QLIKE?
  * HMM's native 1-step predictive log-likelihood is reported for the baseline.

Fair-comparison note: predictive log-likelihood is not comparable across
different observation spaces (returns vs belief vectors), so the head-to-head
uses persistence (unit = days) and the economic check (unit = QLIKE), which
are representation-invariant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evalproto import qlike


def mean_dwell(states: np.ndarray) -> float:
    """Average length of consecutive same-state runs."""
    if len(states) == 0:
        return np.nan
    runs, cur, ln = [], states[0], 1
    for s in states[1:]:
        if s == cur:
            ln += 1
        else:
            runs.append(ln); cur, ln = s, 1
    runs.append(ln)
    return float(np.mean(runs))


def fit_hmm_states(obs: np.ndarray, k: int, seed: int = 0, n_iter: int = 100):
    """Fit a Gaussian HMM and return (model, decoded_states, avg_loglik_per_obs)."""
    from hmmlearn.hmm import GaussianHMM

    obs = np.asarray(obs, float)
    if obs.ndim == 1:
        obs = obs[:, None]
    m = GaussianHMM(n_components=k, covariance_type="diag",
                    n_iter=n_iter, random_state=seed, tol=1e-3)
    m.fit(obs)
    states = m.predict(obs)
    ll = m.score(obs) / len(obs)
    return m, states, ll


def regime_conditioned_qlike(rv_true_log, y_pred_log, bias, states):
    """Economic check: refit a per-regime additive log-var offset (on the same
    OOS rows, leave-out via global mean) and measure QLIKE improvement.

    We compare base QLIKE (single bias) vs regime-specific bias (per-state mean
    residual), estimated on the pooled OOS — a descriptive economic-usefulness
    signal, not a new forecast model.
    """
    rv = np.exp(rv_true_log)
    base_var = np.exp(y_pred_log + bias)
    base_q = np.nanmean(qlike(rv, base_var))

    # per-regime mean of (log RV_true - y_pred) => regime-specific bias
    resid = rv_true_log - y_pred_log
    reg_bias = np.full_like(resid, 0.0)
    for s in np.unique(states):
        m = states == s
        reg_bias[m] = np.mean(resid[m])
    reg_var = np.exp(y_pred_log + reg_bias)
    reg_q = np.nanmean(qlike(rv, reg_var))
    return float(base_q), float(reg_q), float((base_q - reg_q) / base_q * 100)
