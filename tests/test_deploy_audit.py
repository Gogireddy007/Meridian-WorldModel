"""Fast, offline checks of the frozen-model deployment (scripts/deploy_frozen.py, score_deploy.py).

The full audit with real data is scripts/audit_deploy.py; these tests pin down the
mechanisms it relies on so a regression is caught by `pytest tests -q`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.deploy_frozen as dfz  # noqa: E402
import scripts.score_deploy as sd  # noqa: E402
from meridian.features import build_asset_frame  # noqa: E402
from meridian.windows import FEATURES  # noqa: E402

DEP = ROOT / "results" / "deploy"


def synthetic_ohlc(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    px = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0, "adjclose": c}, index=idx)
    macro = pd.DataFrame({"VIXCLS": 15 + rng.normal(0, 1, n), "DGS10": 4.0, "DGS2": 3.5}, index=idx)
    return px, macro


def test_window_at_origin_ignores_the_future():
    px, macro = synthetic_ohlc()
    t = px.index[150]
    f_full = build_asset_frame(px, macro)
    fut = px.copy()
    fut.loc[fut.index > t, ["open", "high", "low", "close"]] *= 3.0          # wreck the future
    f_bad = build_asset_frame(fut, macro)
    mean, std = np.zeros(len(FEATURES), np.float32), np.ones(len(FEATURES), np.float32)
    d1, X1 = dfz.windows_for(f_full, FEATURES, mean, std)
    d2, X2 = dfz.windows_for(f_bad, FEATURES, mean, std)
    i1, i2 = list(d1).index(t), list(d2).index(t)
    np.testing.assert_array_equal(X1[i1], X2[i2])
    assert X1[i1].shape == (dfz.bv.L_NEURAL, len(FEATURES))


def test_scorer_oracle_and_no_tests_on_tiny_windows():
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.bdate_range("2026-11-02", periods=3):
        for a in "ABC":
            rv = float(np.exp(rng.normal(-10, 0.5)))
            rows.append({"asset": a, "target_date": d, "rv_next": rv, "y": np.log(rv),
                         "HAR": np.log(rv) + 0.3, "Oracle": np.log(rv)})
    sc = sd.score_window(pd.DataFrame(rows), ["Oracle", "HAR"])
    assert sc["models"]["Oracle"]["QLIKE"] == pytest.approx(0.0, abs=1e-12)
    assert sc["models"]["Oracle"]["DM_p"] is None and sc["models"]["Oracle"]["ci95_pct_vs_ref"] is None
    assert sc["meta"]["tests_run"] is False


def test_eval_rows_exclude_training_targets():
    R = pd.DataFrame({"end_1": pd.to_datetime(["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03"])})
    m = dfz.build_eval_frame(R, "A")
    assert m.tolist() == [False, False, True, True]
    m = dfz.build_eval_frame(pd.DataFrame({"end_1": pd.to_datetime(["2026-09-08", "2026-09-30", "2026-10-01"])}), "B")
    assert m.tolist() == [False, False, True]


def test_frozen_weights_match_manifest():
    man = json.loads((dfz.FROZEN_A / "manifest.json").read_text())
    for n, h in man["sha256"].items():
        assert dfz.sha(dfz.FROZEN_A / n) == h, n


def test_legacy_checkpoint_loads_strictly_with_11_inputs():
    m, mean, std = dfz.load_B()
    assert len(mean) == len(dfz.LEGACY_FEATURES) == 11 and not m.training


@pytest.mark.skipif(not (DEP / "predictions_A.parquet").exists(), reason="deployment not run")
def test_saved_predictions_are_out_of_sample():
    meta = json.loads((DEP / "deploy_meta.json").read_text())
    for s in ("A", "B"):
        P = pd.read_parquet(DEP / f"predictions_{s}.parquet")
        cut = pd.Timestamp(meta["snapshots"][s]["train_label_end_max"])
        assert (P["target_date"] > cut).all()
        assert (P["target_date"] > P["date"]).all()
        bf = meta["snapshots"][s]["baseline_fit"]
        assert pd.Timestamp(bf["train_label_end_max"]) <= cut


def test_forward_test_drops_in_progress_bars():
    from scripts.forward_test import complete_bars
    idx = pd.to_datetime(["2026-11-03", "2026-11-04", "2026-11-05"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    at = pd.Timestamp("2026-11-04 23:30", tz="UTC")              # 18:30 New York
    assert complete_bars(df, fx=False, now=at).index.max() == pd.Timestamp("2026-11-04")
    assert complete_bars(df, fx=True, now=at).index.max() == pd.Timestamp("2026-11-03")
    at = pd.Timestamp("2026-11-05 01:47", tz="UTC")              # 20:47 New York (scheduled run)
    assert complete_bars(df, fx=True, now=at).index.max() == pd.Timestamp("2026-11-04")
    assert complete_bars(df, fx=False, now=at).index.max() == pd.Timestamp("2026-11-04")


def test_fx_bars_dated_one_day_early_in_uk_summer():
    from meridian.data import fx_true_day, fx_bar_end_utc
    from scripts.forward_test import complete_bars
    lab = pd.to_datetime(["2026-08-09", "2026-08-13", "2026-11-02", "2026-03-29", "2026-03-27"])
    assert list(fx_true_day(lab)) == list(pd.to_datetime(["2026-08-10", "2026-08-14", "2026-11-02", "2026-03-30", "2026-03-27"]))
    assert fx_bar_end_utc(lab[:1])[0] == pd.Timestamp("2026-08-10 23:00")
    # summer evening run at 00:47 UTC on Aug 11: the bar labelled Aug 10 is Aug 11's session, still open
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-08-09", "2026-08-10"]))
    assert complete_bars(df, fx=True, now=pd.Timestamp("2026-08-11 00:47", tz="UTC")).index.max() == pd.Timestamp("2026-08-09")
    assert complete_bars(df, fx=True, now=pd.Timestamp("2026-08-11 23:30", tz="UTC")).index.max() == pd.Timestamp("2026-08-10")


def test_strict_precommitment_uses_real_world_target_close():
    from scripts.forward_test import target_close_utc
    idx = pd.to_datetime(["2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"])
    d = {"prices": {"SPY": pd.DataFrame({"close": 1.0}, index=idx), "EURUSD": pd.DataFrame({"close": 1.0}, index=idx)}}
    S = pd.DataFrame({"asset": ["SPY", "EURUSD"], "origin_date": pd.to_datetime(["2026-11-06", "2026-11-06"])})
    t = target_close_utc(S, d)
    assert t.iloc[0] == pd.Timestamp("2026-11-09 21:00")      # Monday 16:00 New York (EST)
    assert t.iloc[1] == pd.Timestamp("2026-11-10 00:00")      # London midnight ending Monday's FX bar
