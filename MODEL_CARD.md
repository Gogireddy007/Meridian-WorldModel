# Meridian — Model Card

A **modular, interpretable** system for daily financial-market prediction: a bank of
separately-trained specialist modules, each doing one inspectable job, bridged by a
transparent combiner. Data: 11 assets (8 equities + 3 FX), daily OHLC from Yahoo and VIX /
Treasury yields from FRED, 2007–2026.

> **Numbers policy (PREREGISTRATION.md Amendment 6).** This card shows only numbers produced by
> the audited benchmark harness and stored in `results/benchmark_vol.json`. Earlier headline
> claims (σ-multiples, "+6.8% vs HAR", "+8.99% at 8.9σ", held-out-asset and per-year claims,
> regime / VaR / Sharpe figures) were computed before the audit — with significance tests on
> stacked panel rows, a pooled HAR baseline and invalid FX price bars — and are **withdrawn**.

## Volatility module — validated

SSM belief core + JEPA predictor, QLIKE-trained log-variance head, HAR-anchored initialization,
5-seed ensemble, retrained per walk-forward fold (configuration "M0"). Selected on the
2012–2018 development period; reported on the untouched 2019–2026 holdout, against per-asset
HAR-RV and 14 other traditional, ML and deep models.

<!-- BENCH:START -->
Holdout 2019-01-01 → 2026-06-29 (2168 trading dates × 11 assets, 20906 forecasts, never used for any modelling choice). All models get the same walk-forward recalibration; reference = per-asset HAR-RV.

| Model | QLIKE ↓ | vs HAR (95% CI) | DM p | IC ↑ | Hit rate | Assets beating HAR |
|---|---|---|---|---|---|---|
| GR combo: HAR + LightGBM + Meridian M0 (pooled HAR anchor) | 0.3022 | -8.2% [-12.4, -5.2] | <0.001 | 0.594 | 68.3% | 11/11 |
| Meridian M0 (pooled HAR anchor) | 0.3045 | -7.5% [-11.3, -4.7] | <0.001 | 0.591 | 68.1% | 11/11 |
| LightGBM | 0.3118 | -5.3% [-9.3, -2.5] | <0.001 | 0.583 | 67.9% | 10/11 |
| HAR-X (+VIX) | 0.3126 | -5.0% [-7.7, -2.9] | <0.001 | 0.578 | 66.9% | 10/11 |
| Ridge | 0.3132 | -4.9% [-8.7, -2.1] | <0.001 | 0.578 | 67.2% | 9/11 |
| Random forest | 0.3171 | -3.7% [-6.8, -1.4] | <0.001 | 0.573 | 67.6% | 11/11 |
| LHAR (leverage) | 0.3204 | -2.7% [-5.4, -0.5] | 0.011 | 0.569 | 68.0% | 10/11 |
| HAR-RV, per asset (reference) | 0.3292 | +0.0% | — | 0.562 | 66.9% | 0/11 |
| TimeMixer | 0.3392 | +3.0% [+0.3, +6.1] | 0.957 | 0.542 | 67.0% | 3/11 |
| EGARCH(1,1) | 0.3695 | +12.2% [+8.1, +16.8] | 1.000 | 0.515 | 68.2% | 0/11 |
| GJR-GARCH(1,1) | 0.3758 | +14.1% [+9.9, +18.9] | 1.000 | 0.504 | 68.1% | 0/11 |
| GARCH(1,1) | 0.3782 | +14.9% [+11.0, +18.6] | 1.000 | 0.490 | 67.7% | 0/11 |
| EWMA RiskMetrics λ=0.94 | 0.3758 | +14.1% [+7.9, +21.8] | 1.000 | 0.513 | 66.8% | 0/11 |
| Random walk (log RV_t) | 0.3884 | +18.0% [+12.9, +24.7] | 1.000 | 0.472 | 62.3% | 0/11 |

Conservative check against the **strongest** HAR form (raw per-asset HAR, which scores better than its calibrated version here): Meridian M0 alone -6.8% [-9.7, -4.5], DM p <0.001; combination -7.0% [-9.5, -4.9], DM p <0.001; Meridian beats HAR on 11/11 assets and in 14/15 calendar years (2012–2026). Full tables: `results/benchmark_vol.md`.
<!-- BENCH:END -->

Reading: Meridian is the best single volatility model in this benchmark, and its edge over
HAR holds across assets and years. The improvement is statistically clear; whether it exceeds
5% against the strongest HAR form is likely but not certain (see the interval above).

## Other modules — not re-validated under the audited harness

| Module | Job | Output | Status |
|---|---|---|---|
| **Regime** (sticky switching-SSM) | market regime | persistent posterior probabilities | not re-validated; earlier "beats HMM" claim withdrawn |
| **Tail** (Student-t return head) | next-day VaR | quantiles | not re-validated; earlier calibration claims withdrawn |
| **Surprise** (JEPA latent energy) | novelty signal | per-day energy z-score | descriptive only |
| **Combiner** (BOA online aggregation) | allocate across modules | inspectable weights | not re-validated |

These modules exist in the code (`meridian/meridian_wm.py`, `meridian/switching.py`,
`scripts/run_wm.py`, `scripts/modular_stack.py`); they need the same treatment the volatility
module got (saved predictions, date-level significance, clean data) before any number is quoted.

## What it does NOT achieve

- **A better volatility forecast is not, by itself, alpha.** Vol-timing on a single signal does
  not reliably generate net alpha (Cederburg et al. 2020 JFE; DeMiguel et al. 2024 JF). No
  trading or Sharpe number is claimed here.
- **Hit rate is not evidence of skill for this task**: nearly every model, including GARCH, calls
  next-day volatility up/down correctly about two-thirds of the time, because volatility persists.
- **Data limits**: daily range-based realized variance only (no intraday data, so HARQ, jump
  models and Realized GARCH are not tested); free delayed data; 11 assets.

## Roadmap

1. Re-validate the regime, tail and combiner modules under the audited harness.
2. Forward test: run the frozen configuration on data that arrives after 2026-11-01, unchanged.
3. Intraday realized variance (5-minute data) — enables HARQ / HAR-CJ comparisons.
4. Cross-asset breadth (bonds, commodities, futures) and options-implied volatility.

## How to reproduce

```bash
python -m pytest tests/test_harness.py -q
BENCH_VARIANTS=M0 BENCH_ENSEMBLE=5 python scripts/benchmark_vol.py
python scripts/score_vol_benchmark.py results/benchmark_vol_predictions.parquet
python scripts/render_benchmark_md.py
```

## Provenance & discipline

Locked pre-registration with six disclosed amendments (`PREREGISTRATION.md`); Amendment 6
declared the audit fixes and the dev/holdout selection rule before any audited result was
computed, and recorded the selection before the holdout was viewed. JEPA built from primitives
(EB-JEPA has no tabular path). See `README.md`, `results/benchmark_vol.md`, `RESULTS.md`
(dated lab log), `ARCHITECTURE.md`.
