"""Meridian Enterprise World Model — interactive CLI.

    python3 scripts/ask.py "Apple"
    python3 scripts/ask.py "bitcoin"
    python3 scripts/ask.py "Apple vs Tesla"                # comparison
    python3 scripts/ask.py --scenario "Apple" -0.05        # what-if: market -5%
    python3 scripts/ask.py "Nifty 50"

Resolves ANY entity, fetches its data live, runs the module bank, and explains.
With no argument, runs a short demo across several asset classes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.engine import ask, analyze, scenario, portfolio, world_scenario
from meridian.data import fetch_yahoo
from meridian.features import realized_variance


def market_series():
    """SPY daily returns as the market benchmark for connectedness."""
    try:
        return realized_variance(fetch_yahoo("SPY"))["ret"].dropna()
    except Exception:
        return None


def main():
    mkt = market_series()
    args = sys.argv[1:]
    if args and args[0] == "--scenario":
        entity = args[1]; shock = float(args[2])
        print(scenario(entity, shock, market=mkt))
        return
    if args and args[0] == "--portfolio":
        print(portfolio(args[1:], market=mkt))
        return
    if args and args[0] == "--world":
        source = args[1]; shock = float(args[2]); extra = args[3:]
        print(world_scenario(source, shock, extra))
        return
    if args:
        q = " ".join(args)
        print(ask(q, market=mkt))
        return
    # demo across asset classes
    print("Meridian Enterprise World Model — demo across asset classes\n")
    for q in ["Apple", "bitcoin", "gold", "euro", "Tesla"]:
        print("=" * 70)
        print(ask(q, market=mkt))
        print()


if __name__ == "__main__":
    main()
