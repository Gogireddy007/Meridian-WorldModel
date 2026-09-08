"""Evaluation protocol — the scientific spine (see PREREGISTRATION.md).

Purged + embargoed walk-forward CV, pre-registered metrics (QLIKE, MSE on
log-RV, MZ R2), and the Diebold-Mariano test for equal predictive accuracy.

Everything operates on *variance* forecasts. Models predict `log RV_{t+1}`;
we exponentiate to variance for QLIKE.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Walk-forward splits with purge + embargo
# --------------------------------------------------------------------------- #
@dataclass
class WalkForward:
    """Expanding-window walk-forward splitter.

    min_train : first training block size (days)
    test_size : forward test block size (days)
    embargo   : days skipped after each test block before training resumes,
                and days purged from the *end* of train to avoid target overlap.
    """
    min_train: int = 1000
    test_size: int = 126
    embargo: int = 22

    def split(self, n: int):
        start_test = self.min_train
        while start_test < n:
            end_test = min(start_test + self.test_size, n)
            # purge: drop the last `embargo` rows of train (their t+1 target
            # window can overlap the test block edge)
            train_end = start_test - self.embargo
            if train_end <= 50:
                start_test = end_test
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(start_test, end_test)
            yield train_idx, test_idx
            start_test = end_test


# --------------------------------------------------------------------------- #
# Metrics (on variance unless noted)
# --------------------------------------------------------------------------- #
def qlike(rv_true: np.ndarray, var_pred: np.ndarray) -> np.ndarray:
    """Element-wise QLIKE loss on variance. Lower is better.

    L = RV/sigma^2 - log(RV/sigma^2) - 1  >= 0.
    """
    rv = np.clip(rv_true, EPS, None)
    s2 = np.clip(var_pred, EPS, None)
    r = rv / s2
    return r - np.log(r) - 1.0


def mse_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    return (y_true_log - y_pred_log) ** 2


def mz_r2(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Mincer-Zarnowitz R^2 from regressing realized on predicted (log space)."""
    x = np.column_stack([np.ones_like(y_pred_log), y_pred_log])
    beta, *_ = np.linalg.lstsq(x, y_true_log, rcond=None)
    resid = y_true_log - x @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y_true_log - y_true_log.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, EPS)


# --------------------------------------------------------------------------- #
# Diebold-Mariano test of equal predictive accuracy
# --------------------------------------------------------------------------- #
def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1):
    """One-sided DM test that model A has LOWER expected loss than B.

    Returns (dm_stat, p_value_one_sided). d = loss_a - loss_b; negative mean
    favors A. HAC (Newey-West) variance with small-sample (Harvey) correction.
    Uses a Student-t reference with (n-1) dof.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    # Newey-West long-run variance, lag = h-1
    lag = max(h - 1, 0)
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, lag + 1):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * (1.0 - k / (lag + 1)) * cov
    var = var / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    # Harvey, Leybourne, Newbold small-sample correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    # one-sided p for H1: mean(d) < 0  (A better)
    p = stats.t.cdf(dm, df=n - 1)
    return float(dm), float(p)


# --------------------------------------------------------------------------- #
# Container for OOS predictions from one model on one asset
# --------------------------------------------------------------------------- #
def walk_forward_calibrate(dates, y_true_log, y_pred_log, init=252, step=63,
                           embargo=22):
    """Leakage-safe affine + Jensen recalibration of a log-RV forecast.

    Fit  y_true_log ~ a + b*y_pred_log  on PAST rows only (strictly before the
    current block, minus embargo), then variance = exp(a + b*pred + 0.5*sigma^2).
    Applied identically to every model. For OLS-in-log models this is ~a no-op
    (a~0, b~1); for a neural head it fixes the level QLIKE cares about.

    Returns (var_pred, mask) where mask marks rows that received a calibration
    (the first `init` rows are used only to seed and are excluded).
    """
    order = np.argsort(dates)
    d = np.asarray(dates)[order]
    yt = np.asarray(y_true_log, float)[order]
    yp = np.asarray(y_pred_log, float)[order]
    n = len(d)
    var = np.full(n, np.nan)
    b_start = init
    while b_start < n:
        b_end = min(b_start + step, n)
        cut = d[b_start] - np.timedelta64(embargo, "D")
        tr = d < cut
        if tr.sum() >= 100:
            X = np.column_stack([np.ones(tr.sum()), yp[tr]])
            beta, *_ = np.linalg.lstsq(X, yt[tr], rcond=None)
            resid = yt[tr] - X @ beta
            s2 = float(np.var(resid))
            idx = slice(b_start, b_end)
            var[idx] = np.exp(beta[0] + beta[1] * yp[idx] + 0.5 * s2)
        b_start = b_end
    # unsort
    out = np.full(n, np.nan)
    out[order] = var
    mask = np.isfinite(out)
    return out, mask


def walk_forward_combine(dates, y_true_log, preds_log: dict, init=252, step=63, embargo=22):
    """Leakage-safe multivariate generalization of walk_forward_calibrate: fits

        y_true_log ~ a + sum_i b_i * preds_log[i]

    on PAST rows only (strictly before the current block, minus embargo), i.e. a
    walk-forward Granger-Ramanathan optimal forecast COMBINATION rather than a
    single-model recalibration (which is the m=1 special case). Returns the same
    (var_pred, mask) shape as walk_forward_calibrate so callers are interchangeable.
    """
    order = np.argsort(dates)
    d = np.asarray(dates)[order]
    yt = np.asarray(y_true_log, float)[order]
    names = list(preds_log.keys())
    Y = np.column_stack([np.asarray(preds_log[k], float)[order] for k in names])
    n = len(d)
    var = np.full(n, np.nan)
    b_start = init
    while b_start < n:
        b_end = min(b_start + step, n)
        cut = d[b_start] - np.timedelta64(embargo, "D")
        tr = (d < cut) & np.isfinite(Y).all(1) & np.isfinite(yt)
        if tr.sum() >= 100:
            X = np.column_stack([np.ones(tr.sum()), Y[tr]])
            beta, *_ = np.linalg.lstsq(X, yt[tr], rcond=None)
            resid = yt[tr] - X @ beta
            s2 = float(np.var(resid))
            idx = np.arange(b_start, b_end)
            idx = idx[np.isfinite(Y[idx]).all(1)]
            Xi = np.column_stack([np.ones(len(idx)), Y[idx]])
            var[idx] = np.exp(Xi @ beta + 0.5 * s2)
        b_start = b_end
    out = np.full(n, np.nan)
    out[order] = var
    mask = np.isfinite(out)
    return out, mask


@dataclass
class OOSResult:
    name: str
    asset: str
    dates: pd.DatetimeIndex
    y_true_log: np.ndarray          # realized log RV_{t+1}
    y_pred_log: np.ndarray          # predicted conditional mean of log RV_{t+1}
    logvar_bias: np.ndarray | None = None   # Jensen correction: variance = exp(pred + bias)
    extra: dict = field(default_factory=dict)

    @property
    def rv_true(self):
        return np.exp(self.y_true_log)

    @property
    def var_pred(self):
        b = 0.0 if self.logvar_bias is None else self.logvar_bias
        return np.exp(self.y_pred_log + b)

    def losses(self):
        return {
            "qlike": qlike(self.rv_true, self.var_pred),
            "mse_log": mse_log(self.y_true_log, self.y_pred_log),
        }

    def summary(self) -> dict:
        L = self.losses()
        return {
            "model": self.name,
            "asset": self.asset,
            "n": int(len(self.dates)),
            "qlike": float(np.nanmean(L["qlike"])),
            "mse_log": float(np.nanmean(L["mse_log"])),
            "mz_r2": mz_r2(self.y_true_log, self.y_pred_log),
        }
