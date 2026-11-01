"""Scoring for the tail-risk benchmark (PREREGISTRATION.md Amendment 7).

VaR/ES are in return space (negative numbers for a left tail); a violation is r < VaR.
Comparative tests run on the DATE-level loss series (mean across assets per date), as in
meridian/scoring.py; absolute backtests run per asset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from meridian.scoring import hac_mean_test, nw_lag, block_bootstrap_ratio_ci


def tick_loss(r, v, a):
    e = r - v
    return (a - (e < 0)) * e


def fz0_loss(r, v, e, a):
    """FZ0 loss (Patton, Ziegel & Chen 2019) for (VaR, ES) at level a; needs e <= v < 0.
    Differences between models are invariant to rescaling (r, v, e) by a common c > 0."""
    v = np.minimum(v, -1e-10)
    e = np.minimum(e, v)
    hit = (r <= v).astype(float)
    return -hit * (v - r) / (a * e) + v / e + np.log(-e) - 1.0


def kupiec(hits, a):
    n, x = len(hits), int(hits.sum())
    pi = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - a)
    elif x == n:
        lr = -2 * n * np.log(a)
    else:
        lr = -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(stats.chi2.sf(lr, 1)), pi


def christoffersen_cc(hits, a):
    """Conditional coverage LR_uc + LR_ind ~ chi2(2)."""
    h = hits.astype(int)
    n00 = int(((h[:-1] == 0) & (h[1:] == 0)).sum()); n01 = int(((h[:-1] == 0) & (h[1:] == 1)).sum())
    n10 = int(((h[:-1] == 1) & (h[1:] == 0)).sum()); n11 = int(((h[:-1] == 1) & (h[1:] == 1)).sum())

    def ll(p, k0, k1):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return k0 * np.log(1 - p) + k1 * np.log(p)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    lr_ind = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    pk, _ = kupiec(hits, a)
    lr_uc = stats.chi2.isf(pk, 1) if pk > 0 else 1e3
    return float(stats.chi2.sf(max(lr_uc + lr_ind, 0.0), 2))


def dq_test(hits, var, a, lags=4):
    """Engle-Manganelli Dynamic Quantile test: Hit_t - a on const, `lags` lagged hits, VaR_t."""
    hs = hits.astype(float) - a
    T = len(hs)
    if T <= lags + 10:
        return np.nan
    X = np.column_stack([np.ones(T - lags)] + [hs[lags - j - 1:T - j - 1] for j in range(lags)]
                        + [var[lags:]])
    y = hs[lags:]
    XtX = X.T @ X
    try:
        b = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.nan
    dq = float(b @ XtX @ b) / (a * (1 - a))
    return float(stats.chi2.sf(dq, X.shape[1]))


def es_exceedance_test(r, v, e):
    """McNeil-Frey style: on violation days, mean of (r - ES)/|VaR| should be 0; one-sided
    p for 'realized tail losses are worse than the forecast ES'. Returns (p, ratio, n_viol)."""
    m = r < v
    k = int(m.sum())
    if k < 5:
        return np.nan, np.nan, k
    x = (r[m] - e[m]) / np.abs(v[m])
    t = x.mean() / (x.std(ddof=1) / np.sqrt(k))
    return float(stats.t.cdf(t, k - 1)), float(r[m].sum() / e[m].sum()), k


def block_bootstrap_mean_ci(d, block=22, reps=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    T = len(d); nblk = int(np.ceil(T / block))
    out = np.empty(reps)
    for i in range(reps):
        st = rng.integers(0, T - block + 1, size=nblk)
        out[i] = d[(st[:, None] + np.arange(block)[None, :]).ravel()[:T]].mean()
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def score_tail(df, models, ref, k, a, es_models, run_mcs=True):
    """Score VaR (all `models`) and VaR+ES (`es_models`) at level a on a common support.

    df columns: asset, date, r_next, scale, VaR_<m>_<k>, ES_<m>_<k>."""
    w = df[["asset", "date", "r_next", "scale"]].copy()
    r = w["r_next"].to_numpy(); sc = w["scale"].to_numpy()
    for m in models:
        v = df[f"VaR_{m}_{k}"].to_numpy()
        w[f"_tick_{m}"] = tick_loss(r / sc, v / sc, a)
        w[f"_hit_{m}"] = (r < v)
        if m in es_models:
            w[f"_fz_{m}"] = fz0_loss(r, v, df[f"ES_{m}_{k}"].to_numpy(), a)
    by = w.groupby("date")
    T = w["date"].nunique(); lag = nw_lag(T)
    tick_mean = {m: by[f"_tick_{m}"].mean().to_numpy() for m in models}
    tick_sum = {m: by[f"_tick_{m}"].sum().to_numpy() for m in models}
    fz_mean = {m: by[f"_fz_{m}"].mean().to_numpy() for m in es_models}
    out = {}
    for m in models:
        v = df[f"VaR_{m}_{k}"].to_numpy()
        hits = w[f"_hit_{m}"].to_numpy()
        pa = {"kupiec": 0, "cc": 0, "dq": 0}
        for asset, idx in w.groupby("asset").indices.items():
            o = np.argsort(w["date"].to_numpy()[idx]); ii = idx[o]
            h = hits[ii]
            pa["kupiec"] += kupiec(h, a)[0] > 0.05
            pa["cc"] += christoffersen_cc(h, a) > 0.05
            pa["dq"] += (dq_test(h, v[ii] / sc[ii], a) or 0) > 0.05
        rec = {"violation_rate": float(hits.mean()), "tick": float(w[f"_tick_{m}"].mean()),
               "assets_pass_kupiec": int(pa["kupiec"]), "assets_pass_cc": int(pa["cc"]),
               "assets_pass_dq": int(pa["dq"]), "has_es": m in es_models}
        if m != ref:
            dtk = tick_mean[m] - tick_mean[ref]
            t, p = hac_mean_test(dtk, lag)
            rec.update({"tick_pct_vs_ref": rec["tick"] / float(w[f"_tick_{ref}"].mean()) - 1,
                        "tick_ci95_pct_vs_ref": list(block_bootstrap_ratio_ci(tick_sum[m], tick_sum[ref])),
                        "tick_DM_p": p})
        else:
            rec.update({"tick_pct_vs_ref": 0.0, "tick_ci95_pct_vs_ref": None, "tick_DM_p": None})
        if m in es_models:
            ev = df[f"ES_{m}_{k}"].to_numpy()
            pe, ratio, nv = es_exceedance_test(r, v, ev)
            rec.update({"FZ0": float(w[f"_fz_{m}"].mean()), "es_test_p": pe, "es_ratio": ratio,
                        "n_violations": nv})
            if m != ref:
                d = fz_mean[m] - fz_mean[ref]
                t, p = hac_mean_test(d, lag)
                rec.update({"FZ0_diff_vs_ref": float(d.mean()),
                            "FZ0_diff_ci95": list(block_bootstrap_mean_ci(d)), "FZ0_DM_p": p})
            else:
                rec.update({"FZ0_diff_vs_ref": 0.0, "FZ0_diff_ci95": None, "FZ0_DM_p": None})
        out[m] = rec
    if run_mcs:
        from arch.bootstrap import MCS
        for key, L in (("tick", tick_mean), ("FZ0", fz_mean)):
            if len(L) < 2:
                continue
            mcs = MCS(pd.DataFrame(L), size=0.10, reps=1000, block_size=22, method="R", seed=0)
            mcs.compute()
            pv = mcs.pvalues["Pvalue"].to_dict(); inc = set(mcs.included)
            for m in L:
                out[m][f"{key}_MCS_p"] = float(pv.get(m, np.nan))
                out[m][f"{key}_in_MCS90"] = m in inc
    out["_meta"] = {"n_rows": int(len(w)), "n_dates": int(T), "n_assets": int(w["asset"].nunique()),
                    "date_range": [str(w["date"].min().date()), str(w["date"].max().date())],
                    "ref": ref, "alpha": a, "nw_lag": lag}
    return out
