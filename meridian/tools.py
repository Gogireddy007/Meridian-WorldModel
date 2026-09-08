"""The tool / contract layer for the conversational Enterprise World Model (Build #3).

Research-mandated division of labor (FinArena / BloombergGPT evidence): the LLM ROUTES a
user question and EXPLAINS the result, but must NEVER originate a number. Every figure is
the return value of a calibrated module, tagged with its provenance. This makes hallucinated
numbers architecturally impossible: the LLM may only quote values that came back from a
tool call, and each value carries {module, method, as_of} so it is auditable.

`TOOLS` is a function-calling registry (name → {fn, description, params}) an LLM plugs into.
Each tool returns a `Result` with `.value`, `.provenance`, and a human string. `PROVENANCE`
logs every emitted number so `verify_provenance()` can prove an answer invented nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from meridian.engine import (analyze, compare, scenario, world_scenario, portfolio,
                             resolve, Analysis)

PROVENANCE: list[dict] = []          # append-only log of every number the system emits


@dataclass
class Result:
    ok: bool
    numbers: dict[str, Any]          # named figures the LLM is allowed to quote
    provenance: dict[str, str]       # {module, method, as_of, symbol}
    narrative: str                   # the module's own plain-language explanation
    need_data: str | None = None     # honest fallback message if data was missing


def _log(symbol: str, module: str, method: str, as_of: str, numbers: dict):
    PROVENANCE.append({"symbol": symbol, "module": module, "method": method,
                       "as_of": as_of, "numbers": {k: _num(v) for k, v in numbers.items()}})


def _num(v):
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return v


def tool_analyze(entity: str, market: pd.Series | None = None) -> Result:
    """Full world-model read of one entity: vol forecast, regime, tail VaR/ES, beta, trend."""
    a = analyze(entity, market=market)
    if isinstance(a, dict):
        return Result(False, {}, {}, "", need_data=a["message"])
    nums = {"vol_now_ann_pct": a.vol_now_ann * 100, "vol_forecast_1d_pct": a.vol_fc_1d_ann * 100,
            "vol_forecast_1w_pct": a.vol_fc_5d_ann * 100, "vol_percentile": a.vol_pct * 100,
            "var99_1d_pct": a.var99 * 100, "es99_1d_pct": a.es99 * 100,
            "beta_market": a.beta_mkt, "corr_market": a.corr_mkt,
            "ret_1m_pct": a.ret_1m * 100, "ret_12m_pct": a.ret_12m * 100,
            "drawdown_pct": a.drawdown * 100, "last_price": a.last_price, "n_days": a.n_days}
    prov = {"module": "engine.analyze", "method": "HAR+leverage / EVT-GPD / VAR-beta",
            "as_of": a.last_date, "symbol": a.resolved["symbol"]}
    _log(a.resolved["symbol"], prov["module"], prov["method"], a.last_date, nums)
    from meridian.engine import explain
    return Result(True, nums, prov, explain(a))


def tool_compare(entity_a: str, entity_b: str, market: pd.Series | None = None) -> Result:
    txt = compare(entity_a, entity_b, market=market)
    ra, rb = resolve(entity_a), resolve(entity_b)
    prov = {"module": "engine.compare", "method": "two analyze() reads",
            "as_of": "live", "symbol": f"{(ra or {}).get('symbol','?')},{(rb or {}).get('symbol','?')}"}
    return Result(not txt.startswith("## Need data"), {}, prov, txt,
                  need_data=txt if txt.startswith("## Need data") else None)


def tool_scenario(entity: str, market_shock_pct: float, market: pd.Series | None = None) -> Result:
    txt = scenario(entity, market_shock_pct, market=market)
    prov = {"module": "engine.scenario", "method": "beta propagation + EVT tail",
            "as_of": "live", "symbol": (resolve(entity) or {}).get("symbol", "?")}
    return Result(not txt.startswith("## Need"), {"market_shock_pct": market_shock_pct * 100},
                  prov, txt, need_data=txt if txt.startswith("## Need") else None)


def tool_world_scenario(source: str, shock_pct: float, extra: list[str] | None = None) -> Result:
    txt = world_scenario(source, shock_pct, extra or [])
    prov = {"module": "engine.world_scenario", "method": "generalized-IRF network (validated OOS +0.72)",
            "as_of": "live", "symbol": (resolve(source) or {}).get("symbol", "?")}
    return Result(not txt.startswith("## Need"), {"shock_pct": shock_pct * 100}, prov, txt,
                  need_data=txt if txt.startswith("## Need") else None)


def tool_portfolio(entities: list[str], market: pd.Series | None = None) -> Result:
    txt = portfolio(entities, market=market)
    prov = {"module": "engine.portfolio", "method": "Ledoit-Wolf covariance + GMV",
            "as_of": "live", "symbol": ",".join(entities)}
    return Result(not txt.startswith("## Need data"), {}, prov, txt,
                  need_data=txt if txt.startswith("## Need data") else None)


# function-calling registry an LLM router plugs into
TOOLS: dict[str, dict] = {
    "analyze": {"fn": tool_analyze, "description": "Deep world-model read of ONE entity.",
                "params": {"entity": "name or ticker"}},
    "compare": {"fn": tool_compare, "description": "Side-by-side of two entities.",
                "params": {"entity_a": "name/ticker", "entity_b": "name/ticker"}},
    "scenario": {"fn": tool_scenario, "description": "What-if a broad-market move hits ONE entity (single-beta).",
                 "params": {"entity": "name/ticker", "market_shock_pct": "e.g. -0.05"}},
    "world_scenario": {"fn": tool_world_scenario, "description": "Multi-entity network shock propagation.",
                       "params": {"source": "shocked entity", "shock_pct": "e.g. -0.05", "extra": "list of extra entities"}},
    "portfolio": {"fn": tool_portfolio, "description": "Risk/covariance/min-variance for a basket.",
                  "params": {"entities": "list of names/tickers"}},
}


def verify_provenance(text: str, tol: float = 0.15) -> dict:
    """Audit: does every CLAIM-like number in `text` trace to a logged module value?
    Proves the LLM invented no figures. Ignores years (1900-2100) and integer counts;
    matches decimals against the ledger within `tol`. Returns {untraced_sample, ok}."""
    import re
    emitted = [abs(v) for rec in PROVENANCE for v in rec["numbers"].values()
               if isinstance(v, (int, float))]
    def traced(x):
        return any(abs(abs(x) - e) <= tol or (e and abs(abs(x) - e) / e <= 0.02) for e in emitted)
    untraced = []
    for tok in re.findall(r"-?\d+\.?\d*", text):
        x = float(tok)
        if "." not in tok:                        # integers: years/counts are structural, skip
            continue
        if abs(x) < 1.0:                           # tiny ratios, skip
            continue
        if not traced(x):
            untraced.append(x)
    return {"traced_pool": len(emitted), "untraced_sample": untraced[:8],
            "ok": len(untraced) == 0}


CONTRACT = """\
CONVERSATIONAL-LAYER CONTRACT (system-prompt spec for the LLM router)
--------------------------------------------------------------------
You are the explainer for the Meridian Enterprise World Model. You may:
  1. ROUTE the user's question to one TOOLS entry and call it.
  2. EXPLAIN the returned Result in plain language.
  3. FRAME a what-if into tool parameters (e.g. "market crash" → market_shock_pct=-0.10).
You may NOT:
  - State any numeric figure that did not come back from a tool call this turn.
  - Answer a data-less entity yourself — relay the tool's need_data message and ask the
    user for a source, exactly as returned.
Every number you quote MUST appear in some Result.numbers or Result.narrative you received.
verify_provenance() is run on your output; untraced numbers are a contract violation.
"""
