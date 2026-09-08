"""Meridian Enterprise World Model — the interactive query engine.

A user asks about ANY entity ("Apple", "bitcoin", "the yen", "TSLA", "Nifty 50").
The engine:
  1. RESOLVES the name/ticker to a market symbol (Yahoo search — global equities,
     ETFs, FX, crypto, futures, indices).
  2. FETCHES its daily history live (always current → continual, no stale training).
  3. RUNS the validated module bank on it — volatility forecast, regime state,
     tail risk (EVT VaR + Expected Shortfall), market connectedness — each of which
     was benchmarked and earned its place (see WORLD_MODEL.md).
  4. EXPLAINS the result in plain language: what it is, what state it's in, what the
     model expects, and the risks — an in-depth, auditable answer, not a black box.
  5. If NO data is found, it says so honestly and asks the user to supply data or a
     source, rather than inventing an answer.

Every number below is computed causally from real fetched data. Nothing is faked.
"""
from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from meridian.data import _get, fetch_yahoo
from meridian.features import realized_variance, har_features

TRADING_DAYS = 252
EPS = 1e-12

# Common natural-language aliases → Yahoo symbols (fast path before web search).
ALIASES = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "eth": "ETH-USD",
    # commodities → liquid ETFs (clean daily data; continuous-futures =F have roll gaps)
    "gold": "GLD", "silver": "SLV", "oil": "USO", "crude": "USO", "brent": "BNO",
    "natural gas": "UNG", "copper": "CPER",
    "euro": "EURUSD=X", "yen": "JPY=X", "pound": "GBPUSD=X", "sterling": "GBPUSD=X",
    "dollar": "DX=F", "dxy": "DX=F", "swiss franc": "CHF=X", "aussie": "AUDUSD=X",
    "sp500": "^GSPC", "s&p": "^GSPC", "s&p 500": "^GSPC", "spx": "^GSPC",
    "nasdaq": "^IXIC", "dow": "^DJI", "russell": "^RUT", "vix": "^VIX",
    "nifty": "^NSEI", "nifty 50": "^NSEI", "sensex": "^BSESN", "ftse": "^FTSE",
    "nikkei": "^N225", "dax": "^GDAXI", "hang seng": "^HSI",
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA", "meta": "META",
    "treasuries": "TLT", "10 year": "^TNX", "bonds": "TLT",
}


def resolve(query: str) -> dict | None:
    """Map a natural-language entity or ticker to a market symbol.

    Order: exact alias → try as a literal ticker → Yahoo web search (names/companies).
    Returns {symbol, name, type, exchange} or None if nothing plausible is found.
    """
    q = query.strip()
    ql = q.lower()
    if ql in ALIASES:
        sym = ALIASES[ql]
        return {"symbol": sym, "name": q.title(), "type": "alias", "exchange": ""}

    # Yahoo search endpoint resolves company/asset names AND validates tickers.
    try:
        url = ("https://query1.finance.yahoo.com/v1/finance/search?q="
               + urllib.parse.quote(q) + "&quotesCount=6&newsCount=0")
        j = json.loads(_get(url, tries=2, timeout=15))
        quotes = j.get("quotes", [])
    except Exception:
        quotes = []

    if quotes:
        # Prefer an exact symbol match, else the highest-ranked tradable quote.
        exact = [x for x in quotes if x.get("symbol", "").upper() == q.upper()]
        pick = (exact or quotes)[0]
        return {
            "symbol": pick.get("symbol", ""),
            "name": pick.get("shortname") or pick.get("longname") or pick.get("symbol", ""),
            "type": pick.get("quoteType", "").lower(),
            "exchange": pick.get("exchDisp", ""),
        }
    return None


@dataclass
class Analysis:
    query: str
    resolved: dict
    n_days: int
    last_date: str
    last_price: float
    # volatility
    vol_now_ann: float
    vol_fc_1d_ann: float
    vol_fc_5d_ann: float
    vol_pct: float          # where current vol sits in its own history (0-1)
    # regime
    regime: str
    regime_note: str
    # returns / trend
    ret_1m: float
    ret_12m: float
    drawdown: float
    # tail risk (1-day, 99%)
    var99: float
    es99: float
    # connectedness
    beta_mkt: float
    corr_mkt: float
    data_quality_ok: bool = True
    notes: list = field(default_factory=list)


def _har_forecast(rv: pd.Series, ret: pd.Series | None = None):
    """Causal HAR-RV (+ leverage / bad-vol channel) fit on the asset's own history →
    next-day log-RV and iterated 5-day. Returns (vol_1d_ann, vol_5d_ann).

    The leverage term is the variance from DOWN moves (bad vol). Validated on the
    benchmark (scripts/shar_validate.py): ADDING it to the robust Garman-Klass HAR
    improves OOS QLIKE +0.76% on average — most on equities (SPY +3.6%), ~0 on FX,
    exactly as the leverage effect predicts. Kept because it earned it, not by default."""
    # winsorize RV so continuous-futures roll gaps / bad ticks can't dominate the fit
    hi = float(np.nanquantile(rv, 0.995))
    rv = rv.clip(upper=hi)
    lrv = np.log(rv + EPS)
    har = har_features(rv)
    cols = [np.log(har["rv_d"] + EPS), np.log(har["rv_w"] + EPS), np.log(har["rv_m"] + EPS)]
    use_lev = ret is not None
    if use_lev:
        neg = (ret.reindex(rv.index).clip(upper=0) ** 2)   # bad (down-move) variance
        cols.append(np.log(neg + EPS))
    X = np.column_stack(cols)
    y = lrv.shift(-1).to_numpy()
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xo, yo = X[ok], y[ok]
    A = np.column_stack([np.ones(len(Xo)), Xo])
    beta, *_ = np.linalg.lstsq(A, yo, rcond=None)
    jb = 0.5 * np.var(yo - A @ beta)                       # Jensen correction log→level
    lev = float(np.log((neg.iloc[-1] if use_lev else 0.0) + EPS)) if use_lev else None
    # robustness cap: never forecast more than 3× the entity's RECENT (trailing-year) realized
    # RV — continuous-futures roll gaps / bad ticks otherwise make the HAR iteration explode
    cap = float(np.nanmax(rv.iloc[-252:])) * 3.0
    d, w, m = rv.iloc[-1], rv.iloc[-5:].mean(), rv.iloc[-22:].mean()
    base = [1.0, np.log(d + EPS), np.log(w + EPS), np.log(m + EPS)]
    rv1 = min(np.exp(beta @ np.array(base + ([lev] if use_lev else [])) + jb), cap)
    # iterate 5 days; hold the leverage term at its latest value (its info decays as one of 4 terms)
    hist = list(rv.iloc[-22:].to_numpy()); fcs = []; capped = rv1 >= cap * 0.999
    for _ in range(5):
        dd, ww, mm = hist[-1], np.mean(hist[-5:]), np.mean(hist[-22:])
        row = [1.0, np.log(dd + EPS), np.log(ww + EPS), np.log(mm + EPS)]
        raw = np.exp(beta @ np.array(row + ([lev] if use_lev else [])) + jb)
        rvf = min(raw, cap); capped = capped or raw >= cap
        fcs.append(rvf); hist.append(rvf)
    return (float(np.sqrt(rv1 * TRADING_DAYS)),
            float(np.sqrt(np.mean(fcs) * TRADING_DAYS)), bool(capped))


def _regime(rv: pd.Series):
    """Transparent 3-state regime from the vol distribution + trend (named, auditable)."""
    v = np.sqrt(rv * TRADING_DAYS)
    pct = float((v <= v.iloc[-1]).mean())
    recent = v.iloc[-10:].mean(); prior = v.iloc[-40:-10].mean()
    rising = recent > prior * 1.15
    if pct >= 0.85:
        return "Stress", pct, "volatility in the top 15% of its own history"
    if pct >= 0.6 or rising:
        return "Transition", pct, ("volatility elevated and rising" if rising
                                   else "volatility above its median")
    return "Calm", pct, "volatility in the lower/normal part of its range"


def _tail(ret: pd.Series, sigma_now: float, q=0.99):
    """McNeil-Frey conditional EVT: GPD on standardized-loss tail → 1-day VaR & ES."""
    r = ret.dropna().to_numpy()
    sd = np.std(r[-252:]) if len(r) >= 252 else np.std(r)
    z = r / (sd + EPS)
    L = -z; u = np.quantile(L, 0.90); exc = L[L > u] - u; Nu = len(exc); n = len(L)
    daily_sig = sigma_now / np.sqrt(TRADING_DAYS)           # today's 1-day sigma
    if Nu >= 30:
        xi, _, beta = stats.genpareto.fit(exc, floc=0); xi = float(np.clip(xi, -0.4, 0.9))
        if abs(xi) > 1e-4:
            vq = u + (beta / xi) * (((n / Nu) * (1 - q)) ** (-xi) - 1)
        else:
            vq = u + beta * np.log((n / Nu) / (1 - q))
        es = vq / (1 - xi) + (beta - xi * u) / (1 - xi) if xi < 1 else vq * 1.5
    else:
        vq = -np.quantile(z, 1 - q); es = vq * 1.3
    return float(-daily_sig * vq), float(-daily_sig * es)   # signed (negative = loss)


def _connectedness(ret: pd.Series, mkt: pd.Series | None):
    if mkt is None:
        return np.nan, np.nan
    j = pd.concat([ret.rename("a"), mkt.rename("m")], axis=1).dropna().iloc[-252:]
    if len(j) < 60:
        return np.nan, np.nan
    beta = np.cov(j["a"], j["m"])[0, 1] / (np.var(j["m"]) + EPS)
    corr = float(j["a"].corr(j["m"]))
    return float(beta), corr


def analyze(query: str, market: pd.Series | None = None) -> Analysis | dict:
    """Full world-model analysis of one entity. Returns Analysis, or a dict with a
    'need_data' message if the entity can't be resolved / has no usable history."""
    r = resolve(query)
    if r is None or not r.get("symbol"):
        return {"need_data": True, "query": query,
                "message": (f"I couldn't find market data for “{query}”. It may be a private "
                            "company, a non-traded entity, or a name I can't map to a ticker. "
                            "Share a data file (dates + prices) or point me to the source "
                            "(exchange ticker / URL) and I'll analyze it.")}
    try:
        ohlc = fetch_yahoo(r["symbol"])
    except Exception as e:
        return {"need_data": True, "query": query, "resolved": r,
                "message": (f"Resolved “{query}” to {r['symbol']} but couldn't fetch its "
                            f"history ({e}). Point me to another source or share the data.")}
    if len(ohlc) < 120:
        return {"need_data": True, "query": query, "resolved": r,
                "message": (f"{r['symbol']} has only {len(ohlc)} days of history — too little "
                            "to forecast reliably. Share more history if you have it.")}

    rvf = realized_variance(ohlc); rv = rvf["rv"].dropna(); ret = rvf["ret"].dropna()
    px = ohlc["adjclose"]
    vol_now = float(np.sqrt(rv.iloc[-22:].mean() * TRADING_DAYS))
    vol1, vol5, capped = _har_forecast(rv, ret)
    # data-quality flag: capped forecast or many extreme daily jumps ⇒ likely rolled futures / bad ticks
    gap_frac = float((ret.abs() > 0.25).mean())
    dq = capped or gap_frac > 0.005
    regime, pct, rnote = _regime(rv)
    var99, es99 = _tail(ret, vol1)
    beta, corr = _connectedness(ret, market)

    dd = float(px.iloc[-1] / px.cummax().iloc[-1] - 1.0)
    ret_1m = float(px.iloc[-1] / px.iloc[-22] - 1.0) if len(px) > 22 else np.nan
    ret_12m = float(px.iloc[-1] / px.iloc[-252] - 1.0) if len(px) > 252 else np.nan

    return Analysis(
        query=query, resolved=r, n_days=len(ohlc),
        last_date=str(ohlc.index[-1].date()), last_price=float(px.iloc[-1]),
        vol_now_ann=vol_now, vol_fc_1d_ann=vol1, vol_fc_5d_ann=vol5, vol_pct=pct,
        regime=regime, regime_note=rnote, ret_1m=ret_1m, ret_12m=ret_12m, drawdown=dd,
        var99=var99, es99=es99, beta_mkt=beta, corr_mkt=corr, data_quality_ok=not dq,
    )


def explain(a: Analysis) -> str:
    """Plain-language, in-depth narrative of the analysis — the user-facing answer."""
    r = a.resolved
    kind = r.get("type") or "asset"
    ex = f" ({r['exchange']})" if r.get("exchange") else ""
    L = []
    L.append(f"# {r['name']} — {r['symbol']}{ex}")
    L.append(f"*{kind.capitalize()} · {a.n_days} trading days of history · latest close "
             f"{a.last_price:,.2f} on {a.last_date}*\n")
    if not a.data_quality_ok:
        L.append("> ⚠️ **Data-quality warning:** this series shows contract-roll gaps or "
                 "extreme jumps (common for continuous-futures `=F` symbols). Forecasts are "
                 "capped and less reliable — prefer the liquid ETF equivalent (e.g. GLD for "
                 "gold, USO for oil) for a clean read.\n")

    L.append("## What the model expects (volatility)")
    L.append(f"- Current volatility: **{a.vol_now_ann*100:.1f}%** annualized "
             f"(this is at the **{a.vol_pct*100:.0f}th percentile** of its own history).")
    L.append(f"- Forecast next-day vol: **{a.vol_fc_1d_ann*100:.1f}%**; "
             f"average over the next week: **{a.vol_fc_5d_ann*100:.1f}%** "
             f"({'rising' if a.vol_fc_5d_ann > a.vol_now_ann else 'easing'}).")

    L.append("\n## Regime")
    L.append(f"- State: **{a.regime}** — {a.regime_note}.")
    if a.regime == "Stress":
        L.append("- In stress regimes moves cluster and tails fatten; size positions down and "
                 "widen risk limits.")
    elif a.regime == "Transition":
        L.append("- Transition regimes often precede larger moves in either direction — the "
                 "model is flagging elevated uncertainty, not a direction.")
    else:
        L.append("- Calm regime: recent moves are within normal range; base-case risk applies.")

    L.append("\n## Trend & drawdown")
    if np.isfinite(a.ret_1m):
        L.append(f"- Past month: **{a.ret_1m*100:+.1f}%**"
                 + (f"; past year: **{a.ret_12m*100:+.1f}%**." if np.isfinite(a.ret_12m) else "."))
    L.append(f"- Currently **{a.drawdown*100:.1f}%** below its all-time high (in this history).")

    L.append("\n## Tail risk (1-day, 99% confidence — conditional EVT)")
    L.append(f"- Value-at-Risk: a day this bad or worse happens ~1% of the time → about "
             f"**{a.var99*100:.1f}%**.")
    L.append(f"- Expected Shortfall: *when* that 1% tail hits, the average loss is about "
             f"**{a.es99*100:.1f}%** (the Basel III coherent risk number).")

    if np.isfinite(a.beta_mkt):
        L.append("\n## Connectedness to the broad market (SPY)")
        L.append(f"- Beta **{a.beta_mkt:.2f}**, correlation **{a.corr_mkt:.2f}** over the past "
                 f"year — {'moves with' if a.corr_mkt > 0.3 else ('moves against' if a.corr_mkt < -0.3 else 'largely independent of')} "
                 "the equity market.")

    L.append("\n---")
    L.append("*Every figure is computed live from real fetched data with causal (no-lookahead) "
             "methods. The volatility, regime, and tail modules are the same ones benchmarked in "
             "Meridian (vol beats HAR-RV by ~9% QLIKE; EVT Expected Shortfall calibrated to ~1.0). "
             "This is a measurement-and-forecast read, not investment advice.*")
    return "\n".join(L)


def compare(q1: str, q2: str, market: pd.Series | None = None) -> str:
    """Side-by-side world-model read of two entities — risk, regime, expected vol."""
    a1, a2 = analyze(q1, market=market), analyze(q2, market=market)
    for a, q in ((a1, q1), (a2, q2)):
        if isinstance(a, dict):
            return "## Need data\n" + a["message"]
    def row(lbl, f, pct=True, sign=False):
        v1, v2 = f(a1), f(a2)
        fmt = (lambda x: f"{x*100:+.1f}%" if sign else f"{x*100:.1f}%") if pct else (lambda x: f"{x:.2f}")
        return f"| {lbl} | {fmt(v1)} | {fmt(v2)} |"
    L = [f"# {a1.resolved['name']} ({a1.resolved['symbol']}) vs "
         f"{a2.resolved['name']} ({a2.resolved['symbol']})\n",
         f"| Metric | {a1.resolved['symbol']} | {a2.resolved['symbol']} |",
         "|---|---|---|",
         f"| Regime | {a1.regime} | {a2.regime} |",
         row("Current vol (ann)", lambda a: a.vol_now_ann),
         row("Forecast 1-week vol", lambda a: a.vol_fc_5d_ann),
         row("Past-month return", lambda a: a.ret_1m, sign=True),
         row("Past-year return", lambda a: a.ret_12m, sign=True),
         row("Drawdown from high", lambda a: a.drawdown, sign=True),
         row("1-day 99% VaR", lambda a: a.var99, sign=True),
         row("1-day 99% Exp. Shortfall", lambda a: a.es99, sign=True),
         row("Market beta", lambda a: a.beta_mkt, pct=False)]
    riskier = a1 if a1.vol_fc_5d_ann > a2.vol_fc_5d_ann else a2
    calmer = a2 if riskier is a1 else a1
    L.append(f"\n**Read:** {riskier.resolved['symbol']} is the higher-risk of the two "
             f"(forecast week vol {riskier.vol_fc_5d_ann*100:.1f}% vs "
             f"{calmer.vol_fc_5d_ann*100:.1f}%), and its worst-1%-day loss "
             f"({riskier.es99*100:.1f}%) is deeper. "
             f"Regimes: {a1.resolved['symbol']}={a1.regime}, {a2.resolved['symbol']}={a2.regime}.")
    L.append("\n*Live data, causal methods — a comparative risk read, not advice.*")
    return "\n".join(L)


def scenario(query: str, market_shock_pct: float, market: pd.Series | None = None) -> str:
    """First-order 'what-if': propagate a broad-market shock to the entity via its beta,
    contextualized by its current regime and tail. Honest scope: a linear beta
    propagation (the connectedness link), not a full nonlinear crisis simulation."""
    a = analyze(query, market=market)
    if isinstance(a, dict):
        return "## Need data\n" + a["message"]
    if not np.isfinite(a.beta_mkt):
        return (f"I can analyze {a.resolved['symbol']} but have no market benchmark to "
                "propagate a market shock through. Provide a benchmark and I'll simulate it.")
    expected = a.beta_mkt * market_shock_pct
    # crude tail-aware band: +/- one 1-day sigma around the beta-implied move
    sig = a.vol_fc_1d_ann / np.sqrt(TRADING_DAYS)
    lo, hi = expected - sig, expected + sig
    L = [f"# Scenario: broad market {market_shock_pct*100:+.0f}% → {a.resolved['name']} "
         f"({a.resolved['symbol']})\n",
         f"- Estimated **beta {a.beta_mkt:.2f}** to the market (past year).",
         f"- First-order expected move: **{expected*100:+.1f}%** "
         f"(likely range {lo*100:+.1f}% to {hi*100:+.1f}% given today's 1-day vol).",
         f"- Current regime: **{a.regime}** — "
         + ("in stress, betas tend to rise toward 1 and this estimate understates the downside."
            if a.regime == "Stress" else
            "betas can rise in a real sell-off, so treat this as a floor on the downside move."),
         f"- For reference, its ordinary 1-day 99% tail loss is **{a.es99*100:.1f}%** (Exp. Shortfall).",
         "\n*This is a linear beta propagation through the connectedness link — a first-order "
         "simulation. Real crises are nonlinear (betas and correlations jump); treat the downside "
         "as conservative. Live data, not advice.*"]
    return "\n".join(L)


def portfolio(queries: list[str], market: pd.Series | None = None) -> str:
    """Portfolio-level world-model read of a basket of entities: live covariance →
    portfolio volatility, diversification benefit, global-min-variance weights, and
    portfolio 1-day VaR. Uses Ledoit-Wolf shrinkage (robust in higher dimension)."""
    from sklearn.covariance import LedoitWolf
    rets, names, syms = {}, [], []
    for q in queries:
        r = resolve(q)
        if r is None or not r.get("symbol"):
            continue
        try:
            ohlc = fetch_yahoo(r["symbol"])
        except Exception:
            continue
        s = realized_variance(ohlc)["ret"].dropna()
        if len(s) < 252:
            continue
        rets[r["symbol"]] = s; names.append(r["name"]); syms.append(r["symbol"])
    if len(rets) < 2:
        return ("## Need data\nI need at least two entities with usable history to build a "
                "portfolio. Some names didn't resolve or lacked data — try tickers.")
    R = pd.DataFrame(rets).dropna().iloc[-756:]              # last ~3y common history
    mu = R.mean().to_numpy(); S = LedoitWolf().fit(R.to_numpy()).covariance_
    n = len(syms); ew = np.ones(n) / n
    def pvol(w): return float(np.sqrt(w @ S @ w * TRADING_DAYS))
    inv1 = np.linalg.inv(S + 1e-8 * np.eye(n)) @ np.ones(n)
    w_gmv = inv1 / inv1.sum()
    ew_vol = pvol(ew); gmv_vol = pvol(w_gmv)
    avg_standalone = float(np.mean([np.sqrt(S[i, i] * TRADING_DAYS) for i in range(n)]))
    # portfolio 1-day 99% VaR (equal-weight), Cornish-Fisher-free normal approx on daily
    daily_sig = ew_vol / np.sqrt(TRADING_DAYS)
    var99 = float(stats.norm.ppf(0.01) * daily_sig)
    L = [f"# Portfolio — {', '.join(syms)}\n",
         f"*{n} entities · equal-weighted unless noted · covariance from last "
         f"{len(R)} common trading days (Ledoit-Wolf shrinkage)*\n",
         "## Risk",
         f"- Equal-weight portfolio volatility: **{ew_vol*100:.1f}%** annualized.",
         f"- Average standalone volatility of the members: **{avg_standalone*100:.1f}%** — "
         f"diversification removes **{(1-ew_vol/avg_standalone)*100:.0f}%** of that risk.",
         f"- Portfolio 1-day 99% VaR: about **{var99*100:.1f}%**.",
         "\n## Minimum-variance allocation (what lowers risk most)",
         f"- The global-min-variance mix has volatility **{gmv_vol*100:.1f}%** "
         f"(vs {ew_vol*100:.1f}% equal-weight).",
         "| Entity | Min-var weight |", "|---|---|"]
    order = np.argsort(-w_gmv)
    for i in order:
        L.append(f"| {syms[i]} | {w_gmv[i]*100:+.0f}% |")
    L.append("\n*Negative weights = the risk-minimizing mix would short that name; live data, "
             "causal covariance. Risk read, not an allocation recommendation.*")
    return "\n".join(L)


# macro anchors always included in a world-scenario so propagation has real structure
WORLD_ANCHORS = ["SPY", "QQQ", "TLT", "IEF", "HYG", "GLD", "USO", "EURUSD=X", "USDJPY=X"]


def world_scenario(source: str, shock: float, extra: list[str] | None = None,
                   lag: int = 2, H: int = 10) -> str:
    """Multi-entity 'simulate the world': shock one entity, propagate through the VALIDATED
    generalized-IRF network to a basket of macro anchors + any user entities. Reports the
    estimated response of each, and who transmits/absorbs. (network_scenario.py validates
    this OOS: +0.72 corr, 79% direction on held-out large-move days.)"""
    from meridian.network import propagate, connectedness
    src = resolve(source)
    if src is None or not src.get("symbol"):
        return f"## Need data\nCouldn't resolve the shock source “{source}”. Try a ticker."
    want = [src["symbol"]] + WORLD_ANCHORS + [
        (resolve(e) or {}).get("symbol", "") for e in (extra or [])]
    seen, syms = set(), []
    for s in want:
        if s and s not in seen:
            seen.add(s); syms.append(s)
    rets = {}
    for s in syms:
        try:
            rets[s] = realized_variance(fetch_yahoo(s))["ret"].dropna()
        except Exception:
            continue
    R = pd.DataFrame(rets).dropna().iloc[-1000:]
    if src["symbol"] not in R.columns or R.shape[1] < 3 or len(R) < 300:
        return ("## Need data\nNot enough overlapping history to build the network for this "
                "scenario. Try more liquid entities or fewer names.")
    resp = propagate(R, src["symbol"], shock, lag, H).sort_values()
    net = connectedness(R, lag, H)
    L = [f"# World scenario: {src['symbol']} shocked {shock*100:+.0f}%\n",
         f"*First-order propagation through a {R.shape[1]}-entity generalized-IRF network "
         f"(VAR fit on last {len(R)} days). Validated OOS: +0.72 corr, 79% direction.*\n",
         "## Estimated response of each entity", "| Entity | Est. move |", "|---|---|"]
    for s, v in resp.items():
        tag = "  ← shocked" if s == src["symbol"] else ""
        L.append(f"| {s}{tag} | {v*100:+.1f}% |")
    trans = net.sort_values("net", ascending=False).index[:3].tolist()
    absorb = net.sort_values("net").index[:3].tolist()
    L.append(f"\n**Transmission read:** in this basket the biggest shock *sources* are "
             f"{', '.join(trans)}; the biggest *absorbers* are {', '.join(absorb)}. "
             f"A {shock*100:+.0f}% move in {src['symbol']} propagates most strongly to "
             f"{resp.index[0] if shock < 0 else resp.index[-1]}.")
    L.append("\n*Linear first-order network propagation — directionally validated OOS (+0.72 corr) "
             "as a CO-MOVEMENT forecast. Forbes-Rigobon caveat (forbes_rigobon.py): stress-period "
             "co-movement is overwhelmingly the mechanical volatility effect (stable interdependence "
             "amplified by vol), NOT new crisis-specific contagion channels — so treat these as "
             "vol-scaled interdependence, and tail magnitudes as conservative (real crises add "
             "nonlinearity this linear map omits).*")
    return "\n".join(L)


def ask(query: str, market: pd.Series | None = None) -> str:
    """Top-level router: detects comparison / scenario / single-entity intent."""
    q = query.strip()
    # comparison intent: normalize separators to " vs " and split
    norm = q
    for sep in [" versus ", " compared to ", " compare with ", " compare "]:
        norm = norm.replace(sep, " vs ").replace(sep.upper(), " vs ")
    if " vs " in f" {norm.lower()} ":
        bits = [b.strip(" ,") for b in norm.split(" vs ") if b.strip(" ,")]
        # a leading bare "compare" verb may survive as the first token
        bits = [b[len("compare"):].strip() if b.lower().startswith("compare ") else b for b in bits]
        if len(bits) >= 2:
            return compare(bits[0], bits[1], market=market)
    a = analyze(q, market=market)
    if isinstance(a, dict):
        return "## Need data\n" + a["message"]
    return explain(a)
