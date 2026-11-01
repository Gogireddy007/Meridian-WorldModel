"""Harness correctness tests (PREREGISTRATION.md Amendment 6).

Run: pytest tests/test_harness.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from meridian.evalproto import qlike, walk_forward_calibrate
from meridian.scoring import hac_mean_test, score_models


def test_qlike_nonnegative_and_minimized_at_truth():
    rv = np.exp(np.random.default_rng(0).normal(-9, 1, 1000))
    assert (qlike(rv, rv) >= -1e-12).all()
    assert np.allclose(qlike(rv, rv), 0.0, atol=1e-9)
    for scale in (0.5, 0.9, 1.1, 2.0):
        assert qlike(rv, rv * scale).mean() > 0


def test_calibration_has_no_lookahead():
    rng = np.random.default_rng(1)
    n = 1500
    dates = pd.bdate_range("2010-01-01", periods=n).to_numpy()
    yp = rng.normal(-9, 1, n)
    yt = yp + rng.normal(0, 0.5, n)
    base, _ = walk_forward_calibrate(dates, yt, yp)
    k = 900
    yt2 = yt.copy()
    yt2[k:] = rng.normal(5, 3, n - k)             # scramble everything from row k on
    alt, _ = walk_forward_calibrate(dates, yt2, yp)
    # calibrated values strictly before the block containing row k cannot change
    block_start = 252 + ((k - 252) // 63) * 63
    assert np.allclose(base[:block_start], alt[:block_start], equal_nan=True)


def test_hac_dm_is_calibrated_under_null():
    """Equal-accuracy, autocorrelated loss differentials: p-values should be ~uniform,
    i.e. a 5% test rejects ~5% of the time, not ~always (the old pooled-row bug)."""
    rng = np.random.default_rng(2)
    rejections = 0
    trials = 400
    for _ in range(trials):
        e = rng.normal(size=1500)
        d = np.convolve(e, np.ones(5) / 5, mode="same")   # autocorrelated, mean zero
        _, p = hac_mean_test(d)
        rejections += p < 0.05
    rate = rejections / trials
    assert 0.01 <= rate <= 0.10, rate


def test_score_models_equal_models_not_significant():
    """Two forecasts with identical quality on a correlated panel must not look significant."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2015-01-01", periods=800)
    rows = []
    common = rng.normal(0, 0.6, len(dates))           # cross-asset common shock per date
    for a in range(11):
        y = -9 + common + rng.normal(0, 0.4, len(dates))
        rows.append(pd.DataFrame({"asset": f"A{a}", "date": dates, "y": y, "rv_next": np.exp(y),
                                  "har_d": y + rng.normal(0, 0.3, len(dates)),
                                  "m1": y + rng.normal(0, 0.5, len(dates)) + 0.1,
                                  "m2": y + rng.normal(0, 0.5, len(dates)) + 0.1}))
    df = pd.concat(rows, ignore_index=True)
    res = score_models(df, ["m1", "m2"], "m1", run_mcs=False)
    assert res["m2"]["DM_p"] > 0.01


def test_no_train_label_crosses_test_start():
    """For every fold and horizon, rows used for fitting a horizon-h target end before test start."""
    import benchmark_vol as bv
    R, *_ = bv.build()
    dates = np.array(sorted(R["date"].unique()))
    i, checked = bv.MIN_TRAIN_D, 0
    while i < len(dates):
        test_start, cutoff = dates[i], dates[i - bv.EMBARGO_D]
        for tgt, end in bv.TARGET_END.items():
            m = bv.train_rows(R, cutoff, test_start, tgt)
            assert (R.loc[m, end] < test_start).all()
            assert (R.loc[m, "date"] < cutoff).all()
            checked += 1
        i += bv.TEST_D
    assert checked > 0


def test_no_degenerate_rv_rows_after_cleaning():
    """Invalid OHLC bars (stale or close outside range) must not reach features or targets."""
    import benchmark_vol as bv
    R, *_ = bv.build()
    assert (R["y"] > -20).all() and (R["har_d"] > -20).all()
    for o in bv.load_clean()["prices"].values():
        assert (o["high"] > o["low"]).all()
        assert ((o["close"] >= o["low"]) & (o["close"] <= o["high"])).all()
