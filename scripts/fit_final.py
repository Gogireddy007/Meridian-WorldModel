"""Fit a final Meridian on all data through today; save model+scaler and export
a demo JSON: recent market-state timeline (belief regime, vol forecast, surprise).

This powers the live demo. Uses only past-through-t info for each row's forecast
(the model is trained once on the full history, then run forward — for a *demo*
readout this is fine; the honest OOS numbers come from run_meridian/compare).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.model import Meridian, MeridianConfig
from meridian.windows import FEATURES, asset_matrix, build_windows, train_scaler
from scripts.run_meridian import DEVICE, L, train_block

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(demo_asset="SPY", n_days=90):
    t0 = time.time()
    d = load_all()
    macro = d["macro"]
    frames = {a: build_asset_frame(o, macro) for a, o in d["prices"].items()}
    asset_data = {a: asset_matrix(f) for a, f in frames.items()}

    # scaler on all-but-last-year to keep the demo forward-ish; train on all
    end = frames[demo_asset].index.max()
    mean, std = train_scaler(frames, asset_data, end + pd.Timedelta(days=1))

    parts = []
    for a, (dates, X, y, _aux) in asset_data.items():
        w = build_windows(dates, X, y, L, mean, std)
        if w:
            parts.append({k: w[k] for k in ("ctx", "fut", "y")})
    tr = {k: np.concatenate([p[k] for p in parts]) for k in ("ctx", "fut", "y")}

    cfg = MeridianConfig(n_features=len(FEATURES), window=L, seed=0)
    model = train_block(cfg, tr, epochs=12)
    torch.save({"state": model.state_dict(), "cfg": cfg.__dict__,
                "mean": mean, "std": std}, RESULTS / "meridian_final.pt")

    # demo timeline for one asset
    dates, X, y = asset_data[demo_asset]
    w = build_windows(dates, X, y, L, mean, std)
    with torch.no_grad():
        model.eval()
        out = model.forward(torch.tensor(w["ctx"], device=DEVICE),
                            torch.tensor(w["fut"], device=DEVICE))
    vol = out["vol"].cpu().numpy()
    energy = out["energy"].cpu().numpy()
    belief = out["h"].cpu().numpy()

    # regimes from belief via 3-state HMM
    from meridian.regimes import fit_hmm_states
    _, states, _ = fit_hmm_states(belief, k=3)
    # order regimes by realized vol level (0=calm .. 2=stress)
    rv_by_state = {s: np.exp(w["y"][states == s]).mean() for s in np.unique(states)}
    order = {s: r for r, s in enumerate(sorted(rv_by_state, key=rv_by_state.get))}
    states = np.array([order[s] for s in states])

    dts = pd.DatetimeIndex(w["dates"])
    tail = slice(-n_days, None)
    # surprise z-score vs trailing 252d
    en = pd.Series(energy, index=dts)
    en_z = (en - en.rolling(252, min_periods=30).mean()) / (en.rolling(252, min_periods=30).std() + 1e-9)

    rv_ann = np.sqrt(np.exp(w["y"]) * 252) * 100          # realized vol %, annualized
    fc_ann = np.sqrt(np.exp(vol) * 252) * 100             # Meridian forecast vol %
    names = ["Calm", "Transition", "Stress"]

    timeline = []
    for i in range(len(dts))[tail]:
        timeline.append({
            "date": dts[i].strftime("%Y-%m-%d"),
            "regime": names[states[i]],
            "vol_forecast": round(float(fc_ann[i]), 2),
            "vol_realized": round(float(rv_ann[i]), 2),
            "surprise": round(float(en_z.iloc[i]) if np.isfinite(en_z.iloc[i]) else 0.0, 2),
        })
    latest = timeline[-1]
    prev = timeline[-2] if len(timeline) > 1 else latest
    payload = {
        "asset": demo_asset,
        "as_of": latest["date"],
        "latest": latest,
        "changed": {
            "regime_from": prev["regime"], "regime_to": latest["regime"],
            "vol_delta": round(latest["vol_forecast"] - prev["vol_forecast"], 2),
        },
        "timeline": timeline,
    }
    (RESULTS / "demo_state.json").write_text(json.dumps(payload, indent=2))
    print(f"saved meridian_final.pt + demo_state.json  ({time.time()-t0:.0f}s)")
    print(json.dumps({"as_of": payload["as_of"], "latest": latest,
                      "changed": payload["changed"]}, indent=2))


if __name__ == "__main__":
    main()
