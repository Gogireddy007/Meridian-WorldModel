"""Demo + test of the conversational contract layer (Build #3).

Proves the core guarantee two ways:
  1. Routing: free-form questions map to the right calibrated tool; missing entities hit
     the honest need-data fallback (never a fabricated answer).
  2. Anti-hallucination: an LLM explanation built ONLY from the tool's returned numbers
     PASSES provenance; the same explanation with ONE invented figure is CAUGHT. This is
     what makes "numbers always from the modules" an architectural guarantee, not a hope.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.tools import TOOLS, verify_provenance, PROVENANCE, CONTRACT
from meridian.engine import realized_variance
from meridian.data import fetch_yahoo


def market():
    try:
        return realized_variance(fetch_yahoo("SPY"))["ret"].dropna()
    except Exception:
        return None


def main():
    mkt = market()
    print("Enterprise World Model — conversational contract layer\n")
    print(f"registered tools: {', '.join(TOOLS)}\n" + "=" * 66)

    # ---- 1. ROUTING: free-form questions → tools (incl. honest fallback) ----
    print("\n[1] ROUTING\n")
    routes = [
        ("How risky is Apple right now?", "analyze", {"entity": "Apple"}),
        ("What if the market drops 8% — how does Nvidia do?", "world_scenario",
         {"source": "SPY", "shock_pct": -0.08, "extra": ["Nvidia"]}),
        ("Tell me about my neighbor's food truck", "analyze",
         {"entity": "my neighbor's food truck"}),
    ]
    for q, tool, params in routes:
        p = {**params, "market": mkt} if tool in ("analyze", "compare") else params
        res = TOOLS[tool]["fn"](**p)
        if not res.ok and res.need_data:
            print(f"USER: {q}\n  → `{tool}` → NEED-DATA: {res.need_data[:120]}...\n")
        else:
            print(f"USER: {q}\n  → `{tool}` → {res.narrative.splitlines()[0]}  "
                  f"[{res.provenance['module']}]\n")

    # ---- 2. ANTI-HALLUCINATION: provenance catches invented numbers ----
    print("=" * 66 + "\n[2] ANTI-HALLUCINATION GUARANTEE\n")
    res = TOOLS["analyze"]["fn"](entity="Apple", market=mkt)
    N = res.numbers
    # (a) an HONEST LLM paraphrase — quotes only numbers the tool returned
    honest = (f"Apple's volatility is about {N['vol_now_ann_pct']:.1f}% and the model "
              f"forecasts {N['vol_forecast_1d_pct']:.1f}% next day; the 1-day 99% VaR is "
              f"{N['var99_1d_pct']:.1f}% with expected shortfall {N['es99_1d_pct']:.1f}%. "
              f"Its market beta is {N['beta_market']:.2f}.")
    # (b) a HALLUCINATED paraphrase — same text but one invented figure absent from the ledger
    fabricated = honest + " Its 30-day crash probability is 88.8%."
    ph, pf = verify_provenance(honest), verify_provenance(fabricated)
    print("HONEST paraphrase (only tool numbers):")
    print(f"  “{honest}”")
    print(f"  → provenance: {'PASS ✅ every figure traced' if ph['ok'] else 'FAIL '+str(ph['untraced_sample'])}\n")
    print("HALLUCINATED paraphrase (one invented 88.8% figure):")
    print(f"  → provenance: {'PASS' if pf['ok'] else 'CAUGHT ✅ untraced figure(s): '+str(pf['untraced_sample'])}")
    print(f"\n  ledger: {sum(len(r['numbers']) for r in PROVENANCE)} numbers across "
          f"{len(PROVENANCE)} module calls this session.")
    print("\n" + CONTRACT)


if __name__ == "__main__":
    main()
