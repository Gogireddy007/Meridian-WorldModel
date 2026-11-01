"""Scoring for the volatility benchmark (PREREGISTRATION.md Amendment 6).

Everything here operates on a panel of OOS forecasts (one row per asset-date),
already restricted to ONE common row support shared by every model being
compared. Significance is computed on the DATE-level loss series (mean across
assets per date), because the assets on a given date are strongly correlated
and treating ~40k stacked panel rows as independent overstates significance by
orders of magnitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from meridian.evalproto import qlike

EPS = 1e-12


def nw_lag(T: int) -> int:
    """Newey-West automatic lag, floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def hac_mean_test(d: np.ndarray, lag: int | None = None):
    """One-sided test that mean(d) < 0, Newey-West HAC variance, Student-t (T-1 dof).

    Returns (t_stat, p_one_sided). For a DM test, d = loss_model - loss_ref per date,
    so a small p means the model has significantly LOWER loss than the reference.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = d.size
    if T < 20:
        return np.nan, np.nan
    lag = nw_lag(T) if lag is None else lag
    dbar = d.mean()
    e = d - dbar
    var = np.mean(e * e)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1)) * np.mean(e[k:] * e[:-k])
    var /= T
    if var <= 0:
        return np.nan, np.nan
    t = dbar / np.sqrt(var)
    return float(t), float(stats.t.cdf(t, df=T - 1))


def date_level(df: pd.DataFrame, col: str) -> pd.Series:
    """Cross-asset mean of `col` per date (the time series all tests run on)."""
    return df.groupby("date")[col].mean().sort_index()


def block_bootstrap_ratio_ci(sum_m: np.ndarray, sum_ref: np.ndarray, block=22, reps=2000,
                             seed=0, alpha=0.05):
    """Moving-block bootstrap (over dates) CI for the panel QLIKE ratio - 1.

    Inputs are per-date SUMS of row losses (both models on the same rows), so the
    full-sample statistic sum(sum_m)/sum(sum_ref) - 1 equals the reported ratio of
    pooled row means exactly, and each resample is a date-block resample of it.
    """
    rng = np.random.default_rng(seed)
    T = len(sum_m)
    nblk = int(np.ceil(T / block))
    starts_max = T - block
    out = np.empty(reps)
    for r in range(reps):
        starts = rng.integers(0, starts_max + 1, size=nblk)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        out[r] = sum_m[idx].sum() / sum_ref[idx].sum() - 1.0
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def per_asset_ic(df: pd.DataFrame, col: str) -> float:
    """Mean over assets of the time-series Spearman corr(forecast, realized log RV)."""
    ics = []
    for _, g in df.groupby("asset"):
        if len(g) >= 30 and g[col].std() > 0:
            ics.append(stats.spearmanr(g[col], g["y"])[0])
    return float(np.mean(ics)) if ics else np.nan


def per_asset_mz_r2(df: pd.DataFrame, col: str) -> float:
    """Mean over assets of the Mincer-Zarnowitz R^2 (realized log RV on forecast)."""
    r2 = []
    for _, g in df.groupby("asset"):
        x, y = g[col].to_numpy(), g["y"].to_numpy()
        if len(g) >= 30 and np.std(x) > 0:
            r2.append(np.corrcoef(x, y)[0, 1] ** 2)
    return float(np.mean(r2)) if r2 else np.nan


def hit_rate(df: pd.DataFrame, col: str) -> float:
    """Share of rows where sign(forecast - log RV_t) == sign(log RV_{t+1} - log RV_t)."""
    pred_dir = np.sign(df[col].to_numpy() - df["har_d"].to_numpy())
    real_dir = np.sign(df["y"].to_numpy() - df["har_d"].to_numpy())
    nz = real_dir != 0
    return float(np.mean(pred_dir[nz] == real_dir[nz]))


def score_models(df: pd.DataFrame, models: list[str], ref: str, rv_col="rv_next",
                 y_col="y", run_mcs=True) -> dict:
    """Score `models` (columns of log-variance forecasts) on `df`, which must already be
    restricted to a common support where every listed column is finite.

    Returns {model: metrics} plus a '_meta' entry.
    """
    work = df[["asset", "date", y_col, rv_col, "har_d"] + models].copy()
    if y_col != "y":
        work = work.rename(columns={y_col: "y"})
    for m in models:
        work[f"_ql_{m}"] = qlike(work[rv_col].to_numpy(), np.exp(work[m].to_numpy()))
        work[f"_se_{m}"] = (work["y"] - work[m]) ** 2

    ql_date = {m: date_level(work, f"_ql_{m}") for m in models}
    ql_sum = {m: work.groupby("date")[f"_ql_{m}"].sum().sort_index() for m in models}
    T = len(ql_date[ref])
    lag = nw_lag(T)

    ref_asset = work.groupby("asset")[f"_ql_{ref}"].mean()
    out = {}
    for m in models:
        q = float(work[f"_ql_{m}"].mean())
        rec = {
            "QLIKE": q,
            "RMSE_log": float(np.sqrt(work[f"_se_{m}"].mean())),
            "MZ_R2": per_asset_mz_r2(work, m),
            "IC": per_asset_ic(work, m),
            "hit_rate": hit_rate(work, m),
            "assets_beating_ref": int((work.groupby("asset")[f"_ql_{m}"].mean() < ref_asset).sum()),
        }
        if m == ref:
            rec.update({"pct_vs_ref": 0.0, "ci95_pct_vs_ref": None, "DM_t": None, "DM_p": None})
        else:
            dl = (ql_date[m] - ql_date[ref]).to_numpy()
            t, p = hac_mean_test(dl, lag)
            lo, hi = block_bootstrap_ratio_ci(ql_sum[m].to_numpy(), ql_sum[ref].to_numpy())
            rec.update({"pct_vs_ref": q / float(work[f"_ql_{ref}"].mean()) - 1.0,
                        "ci95_pct_vs_ref": [lo, hi], "DM_t": t, "DM_p": p})
        out[m] = rec

    meta = {"n_rows": int(len(work)), "n_dates": int(T), "n_assets": int(work["asset"].nunique()),
            "date_range": [str(work["date"].min().date()), str(work["date"].max().date())],
            "ref": ref, "nw_lag": lag}
    if run_mcs and len(models) > 1:
        from arch.bootstrap import MCS
        LM = pd.DataFrame(ql_date)
        mcs = MCS(LM, size=0.10, reps=1000, block_size=22, method="R", seed=0)
        mcs.compute()
        pv = mcs.pvalues["Pvalue"].to_dict()
        inc = set(mcs.included)
        for m in models:
            out[m]["MCS_p"] = float(pv.get(m, np.nan))
            out[m]["in_MCS90"] = m in inc
    out["_meta"] = meta
    return out
