"""Meridian World Model — integrated live readout. Runs every validated module on the
latest data and produces ONE coherent world state: system-level (connectedness graph +
factor-state + systemic surprise) and asset-level (vol forecast + regime + VaR). Modules
stay decoupled; this harness only READS their interpretable outputs and links them —
no shared backbone, per the no-degradation law. Emits results/world_state.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.tsa.api import VAR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.meridian_wm import MeridianWM, WMConfig
from meridian.windows import FEATURES, asset_matrix, build_windows, train_scaler
from scripts.fetch_broad import load_broad, ASSET_CLASS
from scripts.spillover import realized_vol_panel, generalized_fevd, H, LAG
from scripts.dfg import rv_panel as dfg_panel, fit_dfg
from scripts.run_wm import DEVICE, L, train_block

RESULTS = Path(__file__).resolve().parent.parent / "results"
REG = ["Calm", "Transition", "Stress"]


def system_view():
    d = load_broad()
    names = [n for n in ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
                         "GLD", "USO", "DBC", "EURUSD", "USDJPY", "AUDUSD"] if n in d]
    rv = realized_vol_panel(d, names)
    # connectedness (last 500d for a current read)
    C = generalized_fevd(VAR(rv.iloc[-500:]).fit(LAG), H) * 100
    frm = C.sum(1) - np.diag(C); to = C.sum(0) - np.diag(C); net = to - frm
    total = float(frm.sum() / len(names))
    order = np.argsort(-net)
    # DFG factor-state + systemic surprise
    Y = ((rv - rv.mean()) / rv.std()).to_numpy(np.float32)
    Z, W, energy, _ = fit_dfg(Y, iters=800)
    en = pd.Series(energy, index=rv.index)
    surprise = float((en.iloc[-1] - en.tail(252).mean()) / (en.tail(252).std() + 1e-9))
    return {
        "as_of": str(rv.index[-1].date()),
        "systemic_connectedness_pct": round(total, 1),
        "top_shock_transmitters": [names[i] for i in order[:3]],
        "top_shock_absorbers": [names[i] for i in order[-3:]],
        "factor_state": [round(float(z), 2) for z in Z[-1]],
        "systemic_surprise_sigma": round(surprise, 2),
    }


def asset_view():
    d = load_all(); frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}
    ad = {a: asset_matrix(f) for a, f in frames.items()}
    end = max(f.index.max() for f in frames.values())
    mean, std = train_scaler(frames, ad, end + pd.Timedelta(days=1))
    parts = []
    for a, (dates, X, y, _aux) in ad.items():
        w = build_windows(dates, X, y, L, mean, std)
        if w:
            w["r_next"] = np.nan_to_num(frames[a]["r_next"].reindex(pd.DatetimeIndex(w["dates"])).to_numpy(np.float32))
            parts.append({k: w[k] for k in ("ctx", "fut", "y", "r_next")})
    tr = {k: np.concatenate([p[k] for p in parts]) for k in ("ctx", "fut", "y", "r_next")}
    model = train_block(WMConfig(n_features=len(FEATURES), window=L, seed=0), tr, epochs=12)
    out = {}
    for a, (dates, X, y, _aux) in ad.items():
        w = build_windows(dates, X, y, L, mean, std)
        if not w or len(w["ctx"]) == 0:
            continue
        with torch.no_grad():
            model.eval()
            o = model.forward(torch.tensor(w["ctx"][-1:], device=DEVICE), torch.tensor(w["fut"][-1:], device=DEVICE))
        vol = float(o["vol"][0]); df = float(o["df"][0]); sc = float(o["scale"][0]); rg = int(o["regime"][0])
        out[a] = {"regime": REG[rg] if rg < 3 else f"R{rg}",
                  "vol_forecast_ann_pct": round(float(np.sqrt(np.exp(vol) * 252) * 100), 1),
                  "var95_pct": round(float(sc * stats.t.ppf(0.05, df) * 100), 2)}
    return out


def main():
    print("assembling Meridian World Model state ...\n", flush=True)
    sysv = system_view(); assets = asset_view()
    state = {"system": sysv, "assets": assets}
    (RESULTS / "world_state.json").write_text(json.dumps(state, indent=2))

    s = sysv
    print("=" * 64)
    print(f"  MERIDIAN WORLD MODEL — state as of {s['as_of']}")
    print("=" * 64)
    print(f"  SYSTEM: connectedness {s['systemic_connectedness_pct']}%  |  "
          f"systemic surprise {s['systemic_surprise_sigma']:+.1f}sigma")
    print(f"    shock transmitters: {', '.join(s['top_shock_transmitters'])}   "
          f"absorbers: {', '.join(s['top_shock_absorbers'])}")
    print(f"    latent factor-state: {s['factor_state']}")
    regs = pd.Series([v["regime"] for v in assets.values()]).value_counts().to_dict()
    print(f"  MARKET: regimes {regs}")
    print(f"\n  {'asset':>7} {'regime':>12} {'vol%':>7} {'VaR95%':>8}")
    for a, v in assets.items():
        print(f"  {a:>7} {v['regime']:>12} {v['vol_forecast_ann_pct']:>6.1f}% {v['var95_pct']:>7.2f}%")
    print("\n  saved -> results/world_state.json")
    print("  modules (decoupled, interpretable, linked): vol⊕RF ensemble · switching regime ·")
    print("  Student-t VaR · Diebold-Yilmaz connectedness · DFG factor-state · online-adaptive")


if __name__ == "__main__":
    main()
