"""Tail-risk benchmark -- PREDICTION GENERATOR (PREREGISTRATION.md Amendment 7).

For every core/breadth asset-date in results/benchmark_vol_predictions.parquet, writes each
model's next-day Value-at-Risk and Expected Shortfall of the close-to-close log return
r_{t+1}, forecast at the close of day t, to results/benchmark_tail_predictions.parquet.
It computes NO metrics: scoring is scripts/score_tail_benchmark.py.

Walk-forward: the same folds as the volatility benchmark. Parametric models are fit on
returns dated before each fold's cutoff (test start minus 22 union-calendar days) and then
run with fixed parameters through the fold. Volatility-filtered models use the saved
out-of-sample log-variance forecasts of the audited volatility benchmark.

Models (zero conditional mean throughout)
  HS               historical simulation, trailing 500 returns
  RiskMetrics      EWMA(0.94) variance, normal
  GARCH_N          GARCH(1,1), normal
  GARCH_t          GARCH(1,1), Student-t
  GJR_skewt        GJR-GARCH(1,1), Hansen skewed-t
  GJR_FHS          GJR-GARCH(1,1) (QMLE) + filtered historical simulation  [reference]
  GJR_EVT          GJR-GARCH(1,1) (QMLE) + GPD on standardized residuals (McNeil-Frey)
  CAViaR_AS        asymmetric-slope CAViaR (Engle-Manganelli), VaR only
  HAR_FHS, HAR_EVT, LightGBM_FHS, Meridian_FHS, Meridian_EVT, Meridian_FHS_ACI
                   filtered by the saved OOS volatility forecast sigma^2_{t+1|t}
Filtered models use the trailing <= 1000 past standardized returns (minimum 250).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from scripts.benchmark_vol import load_clean, EMBARGO_D
from meridian.features import realized_variance

RESULTS = Path(__file__).resolve().parent.parent / "results"
ALPHAS = {"010": 0.010, "025": 0.025, "050": 0.050}
HS_WIN = 500
Z_WIN, Z_MIN = 1000, 250
EVT_U, EVT_REFIT = 0.90, 21
ACI_GAMMA = 0.005
VOL_SOURCES = {"HAR": "HAR", "LightGBM": "LightGBM", "Meridian": "Meridian_M0"}


# --------------------------------------------------------------------------- #
# Standardized-distribution quantiles / ES
# --------------------------------------------------------------------------- #
_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)


def es_from_ppf(ppf, a):
    """ES_a = (1/a) * int_0^a ppf(u) du (Gauss-Legendre on a log-spaced variable)."""
    # substitute u = a * exp(-s), s in (0, 30): du = -u ds; the integrand stays smooth
    s = (_GL_X + 1) * 15.0
    u = a * np.exp(-s)
    return float(np.sum(_GL_W * 15.0 * ppf(u) * u) / a)


def std_quantiles(dist, params, a):
    if dist == "normal":
        q = stats.norm.ppf(a)
        return q, -stats.norm.pdf(q) / a
    from arch.univariate import StudentsT, SkewStudent
    d = StudentsT() if dist == "t" else SkewStudent()
    ppf = lambda u: np.asarray(d.ppf(np.asarray(u), np.asarray(params)), float)
    return float(ppf(np.array([a]))[0]), es_from_ppf(ppf, a)


# --------------------------------------------------------------------------- #
# Empirical / EVT tail of standardized returns
# --------------------------------------------------------------------------- #
def emp_var_es(z, a):
    q = np.quantile(z, a)
    tail = z[z <= q]
    return q, (tail.mean() if tail.size else q)


def gpd_fit(z):
    """GPD on the left-tail losses of z. Returns (u, xi, beta, n, Nu) or None."""
    Lz = -z
    u = np.quantile(Lz, EVT_U)
    exc = Lz[Lz > u] - u
    if exc.size < 25:
        return None
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    return u, float(np.clip(xi, -0.45, 0.9)), float(beta), Lz.size, exc.size


def gpd_var_es(fit, a):
    u, xi, beta, n, Nu = fit
    if abs(xi) < 1e-6:
        var = u + beta * np.log(Nu / (n * a))
        es = var + beta
    else:
        var = u + beta / xi * ((n * a / Nu) ** (-xi) - 1)
        es = (var + beta - xi * u) / (1 - xi)
    return -var, -es                       # back to return space (negative numbers)


def filtered_tail(sig, rnext, kind, valid=None):
    """VaR/ES per origin from the trailing standardized returns z = r_{s+1}/sigma_{s+1|s}.

    sig[t] = forecast sigma of r_next[t]; z history for origin t uses origins < t only."""
    n = len(sig)
    z = rnext / sig
    out = {k: (np.full(n, np.nan), np.full(n, np.nan)) for k in ALPHAS}
    lvl = {k: a for k, a in ALPHAS.items()}                  # ACI adaptive levels
    fit = None
    for t in range(n):
        if (valid is not None and not valid[t]) or not np.isfinite(sig[t]):
            continue
        zh = z[max(0, t - Z_WIN):t]
        zh = zh[np.isfinite(zh)]
        if zh.size < Z_MIN:
            continue
        if kind == "EVT" and (fit is None or t % EVT_REFIT == 0):
            fit = gpd_fit(zh)
        for k, a in ALPHAS.items():
            if kind == "FHS":
                q, e = emp_var_es(zh, a)
            elif kind == "ACI":
                q, e = emp_var_es(zh, float(np.clip(lvl[k], 1e-4, 0.5)))
            else:
                q, e = gpd_var_es(fit, a) if fit is not None else emp_var_es(zh, a)
            out[k][0][t], out[k][1][t] = sig[t] * q, sig[t] * e
            if kind == "ACI" and np.isfinite(rnext[t]):
                # the outcome r_next[t] is only used after the forecast for t is made
                lvl[k] += ACI_GAMMA * (a - float(rnext[t] < out[k][0][t]))
    return out


# --------------------------------------------------------------------------- #
# GARCH family
# --------------------------------------------------------------------------- #
GARCH = {"GARCH_N": (dict(vol="GARCH", p=1, o=0, q=1), "normal"),
         "GARCH_t": (dict(vol="GARCH", p=1, o=0, q=1), "t"),
         "GJR_skewt": (dict(vol="GARCH", p=1, o=1, q=1), "skewt"),
         "GJR_QMLE": (dict(vol="GARCH", p=1, o=1, q=1), "normal")}


def garch_fold(ret, train_mask, kind):
    """Returns (sigma_next, dist_params) with sigma_next[t] = sd of ret[t+1] given ret[..t]."""
    from arch import arch_model
    spec, dist = GARCH[kind]
    r = np.nan_to_num(ret * 100.0, nan=0.0)
    try:
        res = arch_model(r[train_mask], mean="Zero", dist=dist, rescale=False, **spec).fit(disp="off")
        fx = arch_model(r, mean="Zero", dist=dist, rescale=False, **spec).fix(res.params)
        s = np.asarray(fx.conditional_volatility) / 100.0
    except Exception:
        return None, None
    nxt = np.full(len(r), np.nan)
    nxt[:-1] = s[1:]
    nd = {"normal": 0, "t": 1, "skewt": 2}[dist]
    return nxt, (res.params.to_numpy()[-nd:] if nd else None)


# --------------------------------------------------------------------------- #
# CAViaR, asymmetric slope:  v_{t+1} = b0 + b1 v_t + b2 r_t^+ + b3 r_t^-
# --------------------------------------------------------------------------- #
def caviar_path(b, r, v0):
    u = b[0] + b[2] * np.clip(r, 0, None) + b[3] * np.clip(-r, 0, None)
    y, _ = signal.lfilter([1.0], [1.0, -b[1]], u, zi=[b[1] * v0])
    return y                                   # y[t] = VaR for r[t+1]


def caviar_fit(r, a):
    v0 = np.quantile(r[:300], a)
    tgt = r[1:]

    def obj(b):
        if not (-0.999 < b[1] < 0.999):
            return 1e9
        v = caviar_path(b, r, v0)[:-1]
        e = tgt - v
        return float(np.mean((a - (e < 0)) * e))
    best = None
    for b1 in (0.7, 0.85, 0.95):
        for b23 in (-0.1, -0.3):
            x0 = np.array([v0 * (1 - b1) * 0.5, b1, b23, 2 * b23])
            res = optimize.minimize(obj, x0, method="Nelder-Mead",
                                    options=dict(maxiter=3000, xatol=1e-7, fatol=1e-10))
            if best is None or res.fun < best.fun:
                best = res
    return best.x, v0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    V = pd.read_parquet(RESULTS / "benchmark_vol_predictions.parquet")
    V["date"] = pd.to_datetime(V["date"])
    d = load_clean()
    union = np.array(sorted(set().union(*[set(o.index) for o in d["prices"].values()])))
    folds = V.groupby("fold")["date"].min().sort_index()
    fold_cut = {f: union[max(0, np.searchsorted(union, s) - EMBARGO_D)] for f, s in folds.items()}

    out = []
    for a, g in V.groupby("asset"):
        g = g.sort_values("date").reset_index(drop=True)
        ret = realized_variance(d["prices"][a])["ret"]
        rnext_full = ret.shift(-1)
        dates_full = ret.index
        pos = dates_full.get_indexer(g["date"])
        assert (pos >= 0).all()
        rec = pd.DataFrame({"asset": a, "date": g["date"], "fold": g["fold"], "breadth": g["breadth"],
                            "r_next": rnext_full.to_numpy()[pos]})
        r_full = ret.to_numpy()
        rn = rnext_full.to_numpy()
        first_oos = g["date"].min()
        rec["scale"] = float(np.nanstd(r_full[dates_full < first_oos]))
        cols = {}

        def put(name, k, v, e=None):
            cols.setdefault(f"VaR_{name}_{k}", np.full(len(g), np.nan))
            cols.setdefault(f"ES_{name}_{k}", np.full(len(g), np.nan))
            cols[f"VaR_{name}_{k}"][:] = v
            if e is not None:
                cols[f"ES_{name}_{k}"][:] = e

        # HS and RiskMetrics (no estimated parameters)
        for k, al in ALPHAS.items():
            hv, he = np.full(len(g), np.nan), np.full(len(g), np.nan)
            for i, p in enumerate(pos):
                h = r_full[max(1, p - HS_WIN + 1):p + 1]
                h = h[np.isfinite(h)]
                if h.size >= HS_WIN // 2:
                    hv[i], he[i] = emp_var_es(h, al)
            put("HS", k, hv, he)
        r0 = np.nan_to_num(r_full)
        s2 = np.empty(len(r0)); s2[0] = np.var(r0[1:31])
        for t in range(1, len(r0)):
            s2[t] = 0.94 * s2[t - 1] + 0.06 * r0[t - 1] ** 2
        s_next = np.sqrt(0.94 * s2 + 0.06 * r0 ** 2)[pos]           # sd of r[t+1] given r[..t]
        for k, al in ALPHAS.items():
            q, e = std_quantiles("normal", None, al)
            put("RiskMetrics", k, s_next * q, s_next * e)

        # GARCH family + CAViaR, refit per fold
        par = {m: {k: (np.full(len(g), np.nan), np.full(len(g), np.nan)) for k in ALPHAS}
               for m in ["GARCH_N", "GARCH_t", "GJR_skewt", "GJR_FHS", "GJR_EVT", "CAViaR_AS"]}
        for f, cut in fold_cut.items():
            fm = (g["fold"] == f).to_numpy()
            if not fm.any():
                continue
            tr = np.asarray(dates_full < cut)
            fpos = pos[fm]
            for m in ["GARCH_N", "GARCH_t", "GJR_skewt"]:
                nxt, dp = garch_fold(r_full, tr, m)
                if nxt is None:
                    continue
                for k, al in ALPHAS.items():
                    q, e = std_quantiles(GARCH[m][1], dp, al)
                    par[m][k][0][fm], par[m][k][1][fm] = nxt[fpos] * q, nxt[fpos] * e
            nxt, _ = garch_fold(r_full, tr, "GJR_QMLE")
            if nxt is not None:
                # filter the whole history with this fold's parameters; z at origin s uses r[s+1]
                for kind, name in (("FHS", "GJR_FHS"), ("EVT", "GJR_EVT")):
                    keep = np.zeros(len(nxt), bool); keep[fpos] = True
                    ft = filtered_tail(nxt, rn, kind, valid=keep)
                    for k in ALPHAS:
                        par[name][k][0][fm], par[name][k][1][fm] = ft[k][0][fpos], ft[k][1][fpos]
            rtr = r_full[tr][1:]
            rtr = rtr[np.isfinite(rtr)]
            for k, al in ALPHAS.items():
                b, v0 = caviar_fit(rtr, al)
                start = int(np.argmax(np.isfinite(r_full)))
                path = np.full(len(r_full), np.nan)
                path[start:] = caviar_path(b, np.nan_to_num(r_full[start:]), v0)
                par["CAViaR_AS"][k][0][fm] = path[fpos]
        for m, byk in par.items():
            for k, (v, e) in byk.items():
                put(m, k, v, None if m == "CAViaR_AS" else e)

        # volatility-filtered models from the saved OOS forecasts
        for src, col in VOL_SOURCES.items():
            sig = np.sqrt(np.exp(g[col].to_numpy(float)))
            rr = rec["r_next"].to_numpy()
            kinds = ["FHS"] + (["EVT", "ACI"] if src == "Meridian" else ["EVT"] if src == "HAR" else [])
            for kind in kinds:
                ft = filtered_tail(sig, rr, kind)
                name = f"{src}_{'FHS_ACI' if kind == 'ACI' else kind}"
                for k in ALPHAS:
                    put(name, k, ft[k][0], ft[k][1])
        for c, v in cols.items():
            rec[c] = v
        out.append(rec)
        print(f"  {a:7s} rows={len(g)}  [{time.time() - t0:.0f}s]", flush=True)

    P = pd.concat(out, ignore_index=True)
    P = P.drop(columns=[c for c in P.columns if c.startswith("ES_CAViaR")])
    P.to_parquet(RESULTS / "benchmark_tail_predictions.parquet")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=RESULTS.parent).stdout.strip()
    meta = {"rows": int(len(P)), "alphas": ALPHAS, "hs_window": HS_WIN, "z_window": Z_WIN, "z_min": Z_MIN,
            "evt_threshold": EVT_U, "evt_refit": EVT_REFIT, "aci_gamma": ACI_GAMMA,
            "vol_sources": VOL_SOURCES, "git_commit_at_run": commit, "runtime_s": round(time.time() - t0)}
    (RESULTS / "benchmark_tail_predictions_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(P)} rows -> results/benchmark_tail_predictions.parquet ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
