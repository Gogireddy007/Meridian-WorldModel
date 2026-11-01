"""Sanity tests for the tail-risk harness (PREREGISTRATION.md Amendment 7)."""
import numpy as np
from scipy import stats

from meridian.riskscore import fz0_loss, tick_loss, kupiec, christoffersen_cc, dq_test
from scripts.benchmark_tail import filtered_tail, ALPHAS


def test_fz0_minimized_at_true_var_es():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(400_000)
    a = 0.025
    q = stats.norm.ppf(a); e = -stats.norm.pdf(q) / a
    best = fz0_loss(r, q, e, a).mean()
    for dv, de in ((0.2, 0), (-0.2, 0), (0, 0.3), (0, -0.3), (0.2, 0.2)):
        assert fz0_loss(r, q + dv, e + de, a).mean() > best


def test_fz0_differences_scale_invariant():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(1000)
    d1 = fz0_loss(r, -2.0, -2.4, 0.025) - fz0_loss(r, -1.8, -2.2, 0.025)
    d2 = fz0_loss(7 * r, -14.0, -16.8, 0.025) - fz0_loss(7 * r, -12.6, -15.4, 0.025)
    assert np.allclose(d1, d2)


def test_tick_minimized_at_quantile():
    rng = np.random.default_rng(2)
    r = rng.standard_normal(200_000)
    q = stats.norm.ppf(0.01)
    assert tick_loss(r, q, 0.01).mean() < min(tick_loss(r, q + 0.2, 0.01).mean(),
                                              tick_loss(r, q - 0.2, 0.01).mean())


def test_backtests_calibrated_under_null():
    rng = np.random.default_rng(3)
    a = 0.025
    rej = np.mean([[kupiec(h, a)[0] < 0.05, christoffersen_cc(h, a) < 0.05,
                    dq_test(h, rng.standard_normal(len(h)), a) < 0.05]
                   for h in (rng.random((400, 2000)) < a)], axis=0)
    assert (rej < 0.12).all(), rej


def test_filtered_tail_has_no_lookahead():
    rng = np.random.default_rng(4)
    n = 1500
    sig = np.exp(rng.normal(0, 0.3, n)) * 0.01
    r = sig * rng.standard_t(5, n)
    base = filtered_tail(sig, r, "ACI")
    t = 900
    r2 = r.copy(); r2[t:] = rng.standard_normal(n - t)          # scramble outcomes from t on
    alt = filtered_tail(sig, r2, "ACI")
    for k in ALPHAS:
        assert np.allclose(base[k][0][:t + 1], alt[k][0][:t + 1], equal_nan=True)
