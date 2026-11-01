"""Frozen forward test of the volatility forecast (PREREGISTRATION.md Amendment 7).

    python scripts/forward_test.py freeze    # once: fetch data, train + save frozen models
    python scripts/forward_test.py predict   # each trading day: fetch data, log new forecasts
    python scripts/forward_test.py score     # any time: score logged forecasts whose outcome exists

Models: Meridian M0 (5 seeds, pooled HAR-anchored init — the audited configuration) and
per-asset HAR-RV, both trained ONCE on all data up to the freeze date and never changed.
They are compared in their raw form (no recalibration), as in the conservative holdout
check `holdout_vs_raw_HAR` of the audited benchmark.

Data lives in results/forward/data/ (its own snapshot), so the audited benchmark inputs in
data/ are never overwritten. Forecasts are appended to results/forward/forecast_log.csv
and never edited. A row is `precommitted` when its origin date was the asset's latest
available bar at logging time, i.e. the outcome could not have been in the data; only those
rows count as forward results.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import meridian.data as mdata
from meridian.data import EQUITIES, FX, BREADTH, FRED_SERIES, fetch_yahoo
from meridian.evalproto import qlike
from meridian.model import Meridian
from meridian.scoring import hac_mean_test
from meridian.windows import FEATURES, AUX_HORIZONS, train_scaler
import scripts.benchmark_vol as bv

ROOT = Path(__file__).resolve().parent.parent
FWD = ROOT / "results" / "forward"
FDATA, FROZEN, LOG = FWD / "data", FWD / "frozen", FWD / "forecast_log.csv"
ENSEMBLE = 5
SEED_BASE = 9000


def fetch():
    """Refresh the forward-test data snapshot (prices from Yahoo, macro from FRED)."""
    import io
    FDATA.mkdir(parents=True, exist_ok=True)
    for name, sym in ({s: s for s in EQUITIES} | FX | BREADTH).items():
        fetch_yahoo(sym).to_parquet(FDATA / f"px_{name}.parquet")
    cols = {}
    for s in FRED_SERIES:
        # FRED stalls requests with a browser user agent (meridian.data._get); plain curl works
        raw = subprocess.run(["curl", "-sS", "--retry", "3", "-m", "90", "-A", "curl/8.7.1",
                              f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"],
                             capture_output=True, check=True, timeout=300).stdout.decode()
        df = pd.read_csv(io.StringIO(raw))
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        ser = pd.to_numeric(df.set_index(df.columns[0])[s], errors="coerce").dropna()
        cols[s] = ser[ser.index >= pd.Timestamp(mdata.DEFAULT_START)]
    pd.DataFrame(cols).sort_index().to_parquet(FDATA / "macro.parquet")


def use_snapshot():
    mdata.DATA_DIR = FDATA                     # load_all() reads its module-global DATA_DIR


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def latest_inputs(frames):
    """Per asset: last-row HAR features and the last Meridian context window (features only,
    no target needed), for every origin date with finite features."""
    out = {}
    for a, f in frames.items():
        F = f[FEATURES]
        ok = np.isfinite(F.to_numpy(float)).all(1)
        out[a] = (f.index[ok], F[ok].to_numpy(np.float32), f.loc[ok, bv.HAR3].to_numpy(float))
    return out


def freeze():
    if (FROZEN / "manifest.json").exists():
        sys.exit("already frozen — the forward test must not be re-frozen (delete results/forward to restart)")
    fetch()
    use_snapshot()
    R, _, _, d, frames, asset_win = bv.build()
    last = R["date"].max()
    end = last + pd.Timedelta(days=1)
    core = R[R["asset"].isin(bv.CORE_ASSETS)]
    pooled = bv.ols(core, bv.HAR3)[0]
    pooled_aux = {h: bv.ols(core, bv.HAR3, h)[0] for h in AUX_HORIZONS}
    per_asset = {}
    for a in R["asset"].unique():
        ra = R[R["asset"] == a]
        per_asset[(a, "y")] = bv.ols(ra, bv.HAR3)
        for h in AUX_HORIZONS:
            per_asset[(a, h)] = bv.ols(ra, bv.HAR3, h)
    core_win = {a: v for a, v in asset_win.items() if a in bv.CORE_ASSETS}
    mean, std = train_scaler(frames, core_win, end)
    W = bv.meridian_windows(asset_win, frames, mean, std, per_asset)
    parts = []
    for a, w in W.items():
        ok = np.isfinite(w["off_y"]) & np.isfinite(w["off_y_5"]) & np.isfinite(w["off_y_22"])
        if a in bv.CORE_ASSETS and ok.any():
            parts.append({k: w[k][ok] for k in bv.KEYS})
    TR = bv._tensors({k: np.concatenate([p[k] for p in parts]) for k in bv.KEYS})
    FROZEN.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in range(ENSEMBLE):
        cfg = bv.variant_cfg("M0", SEED_BASE + s)
        m = bv.train_meridian("M0", cfg, TR, None, mean, std, pooled, pooled_aux)
        p = FROZEN / f"meridian_M0_seed{s}.pt"
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, p)
        files[p.name] = sha(p)
        print(f"  trained seed {s}", flush=True)
    np.savez(FROZEN / "scaler.npz", mean=mean, std=std)
    har = {a: {"beta": per_asset[(a, "y")][0].tolist(), "jensen": per_asset[(a, "y")][1]}
           for a in R["asset"].unique() if per_asset[(a, "y")] is not None}
    (FROZEN / "har.json").write_text(json.dumps(har, indent=2))
    for n in ("scaler.npz", "har.json"):
        files[n] = sha(FROZEN / n)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_data_through": str(last.date()), "git_commit": commit,
                "model": "Meridian M0, 5 seeds, pooled HAR-anchored init", "reference": "per-asset HAR-RV",
                "train_rows": int(TR["y"].shape[0]), "sha256": files}
    (FROZEN / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    predict(refresh=False)


def load_frozen():
    man = json.loads((FROZEN / "manifest.json").read_text())
    for n, h in man["sha256"].items():
        if sha(FROZEN / n) != h:
            sys.exit(f"frozen file {n} changed since freeze — refusing to forecast")
    sc = np.load(FROZEN / "scaler.npz")
    models = []
    for s in range(ENSEMBLE):
        m = Meridian(bv.variant_cfg("M0", SEED_BASE + s)).to(bv.DEV)
        m.load_state_dict(torch.load(FROZEN / f"meridian_M0_seed{s}.pt"))
        m.eval()
        models.append(m)
    return man, sc["mean"], sc["std"], json.loads((FROZEN / "har.json").read_text()), models


def predict(refresh=True):
    man, mean, std, har, models = load_frozen()
    if refresh:
        fetch()
    use_snapshot()
    d = bv.load_clean()
    frames = {a: bv.build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}
    log = pd.read_csv(LOG, parse_dates=["origin_date"]) if LOG.exists() else None
    done = set() if log is None else set(zip(log["asset"], log["origin_date"]))
    start = pd.Timestamp(man["train_data_through"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for a, (dates, X, H) in latest_inputs(frames).items():
        if a not in har:
            continue
        Xs = (X - mean) / std
        for i in np.where(dates > start)[0]:           # origins after the last training origin
            t = dates[i]
            if (a, t) in done or i < bv.L_NEURAL - 1:
                continue
            ctx = torch.tensor(Xs[None, i - bv.L_NEURAL + 1:i + 1], device=bv.DEV)
            with torch.no_grad():
                mer = float(np.mean([m.forward(ctx)["vol"].cpu().numpy()[0] for m in models]))
            b = har[a]["beta"]
            h = float(b[0] + H[i] @ np.array(b[1:]) + har[a]["jensen"])
            rows.append({"asset": a, "origin_date": t.date().isoformat(), "Meridian_M0": mer, "HAR": h,
                         "logged_at_utc": now, "precommitted": bool(t == dates[-1]),
                         "breadth": a in BREADTH})
    if rows:
        new = pd.DataFrame(rows)
        new.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"logged {len(rows)} new forecasts ({sum(r['precommitted'] for r in rows)} pre-committed) -> {LOG}")


def score():
    use_snapshot()
    d = bv.load_clean()
    log = pd.read_csv(LOG, parse_dates=["origin_date"])
    real = []
    for a, o in d["prices"].items():
        f = bv.build_asset_frame(o, d["macro"])
        real.append(pd.DataFrame({"asset": a, "origin_date": f.index, "rv_next": f["rv_next"].to_numpy()}))
    S = log.merge(pd.concat(real), on=["asset", "origin_date"], how="left")
    S = S[S["precommitted"] & ~S["breadth"] & np.isfinite(S["rv_next"])]
    if S.empty:
        print("no pre-committed core forecasts with a realized outcome yet"); return
    for m in ("Meridian_M0", "HAR"):
        S[f"q_{m}"] = qlike(S["rv_next"].to_numpy(), np.exp(S[m].to_numpy()))
    dl = S.groupby("origin_date")[["q_Meridian_M0", "q_HAR"]].mean()
    T = len(dl)
    pct = S["q_Meridian_M0"].mean() / S["q_HAR"].mean() - 1
    t, p = hac_mean_test((dl["q_Meridian_M0"] - dl["q_HAR"]).to_numpy())
    res = {"scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_dates": T, "n_forecasts": int(len(S)),
           "date_range": [str(S["origin_date"].min().date()), str(S["origin_date"].max().date())],
           "QLIKE_Meridian_M0": float(S["q_Meridian_M0"].mean()), "QLIKE_HAR": float(S["q_HAR"].mean()),
           "pct_vs_HAR": float(pct), "DM_p": None if not np.isfinite(p) else float(p),
           "note": "DM test needs >= 20 dates; under ~250 dates the result is noisy."}
    (FWD / "forward_score.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"freeze": freeze, "predict": predict, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "predict"]()
