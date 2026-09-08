"""Volatility baselines, fit under the pre-registered walk-forward protocol.

Each model implements fit(train_df) / predict(test_df) -> predicted log RV_{t+1}.
`run_model` drives it across purged walk-forward splits and returns an OOSResult.

Models: HAR-RV (champion), AR(1), AR(3), EWMA (RiskMetrics), GARCH(1,1).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .evalproto import EPS, OOSResult, WalkForward

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
class _OLS:
    """Small OLS with intercept; predicts log RV_{t+1}."""

    feat_cols: list[str] = []

    def __init__(self):
        self.beta = None

    def fit(self, tr: pd.DataFrame):
        X = self._design(tr)
        y = tr["y"].to_numpy()
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        return self._design(te) @ self.beta

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feat_cols].to_numpy()
        return np.column_stack([np.ones(len(X)), X])


class HARRV(_OLS):
    name = "HAR-RV"
    feat_cols = ["har_d", "har_w", "har_m"]
    logspace_mean = True


class AR1(_OLS):
    name = "AR(1)"
    feat_cols = ["log_rv"]
    logspace_mean = True


class AR3:
    """AR(3) on log RV (needs constructed lags)."""

    name = "AR(3)"
    logspace_mean = True

    def __init__(self):
        self.beta = None

    @staticmethod
    def _design(df):
        lr = df["log_rv"]
        X = np.column_stack([
            np.ones(len(df)),
            lr.to_numpy(),
            lr.shift(1).to_numpy(),
            lr.shift(2).to_numpy(),
        ])
        return X

    def fit(self, tr):
        X = self._design(tr)
        y = tr["y"].to_numpy()
        m = np.isfinite(X).all(1) & np.isfinite(y)
        self.beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        return self

    def predict(self, te):
        X = self._design(te)
        X = np.nan_to_num(X, nan=0.0)
        return X @ self.beta


class EWMA:
    """RiskMetrics EWMA variance. No training; lam fixed at 0.94.

    Predicts log of the EWMA variance carried to t+1.
    """

    name = "EWMA"

    def __init__(self, lam: float = 0.94):
        self.lam = lam

    def fit(self, tr):
        # seed variance from train tail so test starts warm
        r = tr["ret"].dropna().to_numpy()
        v = np.var(r[-100:]) if len(r) >= 20 else np.var(r) if len(r) else 1e-4
        self._seed = float(v)
        return self

    def predict(self, te):
        lam = self.lam
        v = getattr(self, "_seed", 1e-4)
        preds = []
        for r in te["ret"].to_numpy():
            r = 0.0 if not np.isfinite(r) else r
            v = lam * v + (1 - lam) * r * r  # variance forecast for t+1
            preds.append(np.log(max(v, EPS)))
        return np.array(preds)


class GARCH11:
    """GARCH(1,1) via arch; frozen params + manual 1-step recursion on test."""

    name = "GARCH(1,1)"

    def __init__(self):
        self.params = None
        self.scale = 100.0  # arch likes returns in %; we scale for stability

    def fit(self, tr):
        from arch import arch_model

        r = tr["ret"].dropna().to_numpy() * self.scale
        try:
            res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1,
                             dist="normal").fit(disp="off", show_warning=False)
            self.params = dict(res.params)
            self._last_var = float(res.conditional_volatility[-1] ** 2)
            self._last_resid = float(r[-1] - res.params.get("mu", 0.0))
        except Exception:  # noqa: BLE001
            self.params = None
        return self

    def predict(self, te):
        s = self.scale
        if not self.params:
            v = np.var((te["ret"].dropna().to_numpy() * s)) or 1.0
            return np.full(len(te), np.log(max(v / s / s, EPS)))
        w = self.params.get("omega", 0.0)
        a = self.params.get("alpha[1]", 0.0)
        b = self.params.get("beta[1]", 0.0)
        mu = self.params.get("mu", 0.0)
        var = self._last_var
        resid = self._last_resid
        preds = []
        for r in te["ret"].to_numpy():
            r = mu if not np.isfinite(r) else r * s
            var = w + a * resid * resid + b * var  # variance forecast for t+1
            resid = r - mu
            preds.append(np.log(max(var / (s * s), EPS)))
        return np.array(preds)


BASELINES = [HARRV, AR1, AR3, EWMA, GARCH11]


# --------------------------------------------------------------------------- #
def run_model(model_cls, df: pd.DataFrame, asset: str, wf: WalkForward) -> OOSResult:
    """Fit `model_cls` across purged walk-forward splits; collect OOS preds."""
    df = df.copy()
    # rows usable for modeling (features + target present)
    core = ["y", "log_rv", "har_d", "har_w", "har_m", "ret"]
    valid = df[core].notna().all(1).to_numpy()
    idx_all = np.arange(len(df))

    dates, ytrue, ypred, biases = [], [], [], []
    for tr_idx, te_idx in wf.split(len(df)):
        tr = df.iloc[tr_idx]
        tr = tr[valid[tr_idx]]
        te = df.iloc[te_idx]
        te_mask = valid[te_idx]
        te_use = te[te_mask]
        if len(tr) < 200 or len(te_use) == 0:
            continue
        model = model_cls().fit(tr)
        p = np.asarray(model.predict(te_use), float)
        # Jensen bias correction for log-mean models: variance forecast needs
        # +0.5*Var(residual) since E[exp(X)] = exp(E[X] + 0.5 Var[X]).
        bias = 0.0
        if getattr(model_cls, "logspace_mean", False):
            resid = tr["y"].to_numpy() - np.asarray(model.predict(tr), float)
            resid = resid[np.isfinite(resid)]
            bias = 0.5 * float(np.var(resid)) if resid.size > 10 else 0.0
        dates.append(te_use.index)
        ytrue.append(te_use["y"].to_numpy())
        ypred.append(p)
        biases.append(np.full(len(te_use), bias))

    if not dates:
        raise RuntimeError(f"no OOS predictions for {model_cls.name} / {asset}")
    return OOSResult(
        name=model_cls.name,
        asset=asset,
        dates=pd.DatetimeIndex(np.concatenate([d.values for d in dates])),
        y_true_log=np.concatenate(ytrue),
        y_pred_log=np.concatenate(ypred),
        logvar_bias=np.concatenate(biases),
    )
