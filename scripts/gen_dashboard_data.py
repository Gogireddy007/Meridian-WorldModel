"""Generate a real-engine snapshot for the Enterprise World Model dashboard.

Runs the ACTUAL modules on a diverse universe and bakes the outputs into JSON the
(offline, CSP-sandboxed) artifact reads. The network response is stored per -1% market
move; since first-order propagation is linear, the dashboard slider faithfully scales it.
Nothing is fabricated — every number is a module output, snapshotted as of the run date.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.engine import analyze, resolve
from meridian.data import fetch_yahoo
from meridian.features import realized_variance
from meridian.network import propagate, connectedness
from sklearn.covariance import LedoitWolf

RESULTS = Path(__file__).resolve().parent.parent / "results"

UNIVERSE = [
    ("Apple", "equity"), ("Microsoft", "equity"), ("Nvidia", "equity"),
    ("Tesla", "equity"), ("JPMorgan", "equity"),
    ("SPY", "index"), ("QQQ", "index"), ("Nikkei", "index"),
    ("bitcoin", "crypto"), ("ethereum", "crypto"),
    ("gold", "commodity"), ("oil", "commodity"),
    ("TLT", "bond"), ("HYG", "bond"),
    ("euro", "fx"), ("yen", "fx"),
]
NET_BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "JPM",
              "TLT", "HYG", "GLD", "USO", "EURUSD=X", "USDJPY=X"]
PORT_BASKET = ["SPY", "QQQ", "TLT", "HYG", "GLD", "AAPL", "NVDA", "BTC-USD"]
ANN = 252


def market_series():
    return realized_variance(fetch_yahoo("SPY"))["ret"].dropna()


def entity_cards(mkt):
    cards = []
    for q, cls in UNIVERSE:
        a = analyze(q, market=mkt)
        if isinstance(a, dict):
            continue
        cards.append({
            "query": q, "asset_class": cls, "name": a.resolved["name"], "symbol": a.resolved["symbol"],
            "last_price": round(a.last_price, 2), "last_date": a.last_date, "n_days": a.n_days,
            "vol_now": round(a.vol_now_ann * 100, 1), "vol_1d": round(a.vol_fc_1d_ann * 100, 1),
            "vol_1w": round(a.vol_fc_5d_ann * 100, 1), "vol_pct": round(a.vol_pct * 100),
            "regime": a.regime, "var99": round(a.var99 * 100, 1), "es99": round(a.es99 * 100, 1),
            "beta": round(a.beta_mkt, 2) if np.isfinite(a.beta_mkt) else None,
            "corr": round(a.corr_mkt, 2) if np.isfinite(a.corr_mkt) else None,
            "ret_1m": round(a.ret_1m * 100, 1) if np.isfinite(a.ret_1m) else None,
            "ret_12m": round(a.ret_12m * 100, 1) if np.isfinite(a.ret_12m) else None,
            "drawdown": round(a.drawdown * 100, 1), "dq_ok": a.data_quality_ok,
        })
    return cards


def net_returns(names):
    return pd.DataFrame({n: np.log(fetch_yahoo(n)["adjclose"]).diff() for n in names}).dropna()


def network_block():
    R = net_returns(NET_BASKET).iloc[-1000:]
    resp = propagate(R, "SPY", -0.01)                 # response to a -1% SPY move (linear)
    net = connectedness(R)
    return {
        "symbols": list(R.columns),
        "unit_response_pct": {k: round(v * 100, 3) for k, v in resp.items()},  # per -1% SPY
        "net": {k: round(float(net.loc[k, "net"]), 1) for k in R.columns},
        "transmitters": net.sort_values("net", ascending=False).index[:3].tolist(),
        "receivers": net.sort_values("net").index[:3].tolist(),
    }


def portfolio_block():
    R = net_returns(PORT_BASKET).iloc[-756:]
    S = LedoitWolf().fit(R.to_numpy()).covariance_
    n = len(PORT_BASKET); ew = np.ones(n) / n
    inv1 = np.linalg.inv(S + 1e-8 * np.eye(n)) @ np.ones(n); w = inv1 / inv1.sum()
    pv = lambda ww: float(np.sqrt(ww @ S @ ww * ANN))
    avg_sa = float(np.mean([np.sqrt(S[i, i] * ANN) for i in range(n)]))
    return {
        "symbols": PORT_BASKET,
        "ew_vol": round(pv(ew) * 100, 1), "gmv_vol": round(pv(w) * 100, 1),
        "avg_standalone": round(avg_sa * 100, 1),
        "div_benefit": round((1 - pv(ew) / avg_sa) * 100),
        "gmv_weights": {PORT_BASKET[i]: round(w[i] * 100) for i in range(n)},
    }


def main():
    mkt = market_series()
    snap = {
        "as_of": str(pd.Timestamp(mkt.index[-1]).date()),
        "entities": entity_cards(mkt),
        "network": network_block(),
        "portfolio": portfolio_block(),
    }
    out = RESULTS / "dashboard_data.json"
    out.write_text(json.dumps(snap, indent=2))
    print(f"wrote {out}  ({len(snap['entities'])} entities, "
          f"{len(snap['network']['symbols'])} network nodes, as_of {snap['as_of']})")


if __name__ == "__main__":
    main()
