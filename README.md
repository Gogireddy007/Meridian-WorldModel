# Meridian

A market **belief-state** core that reads daily equity + FX data, tracks the
current regime, forecasts next-day volatility, and emits a live **surprise
score** — benchmarked honestly against HAR-RV (volatility) and a Gaussian HMM
(regimes) under a pre-registered, leakage-controlled protocol.

> Adapts the JEPA idea (predict the *latent* of the future, not the pixels) to
> financial time series, with a diagonal **SSM belief core** and a **SIGReg**
> anti-collapse regularizer, plus supervised volatility and regime read-outs.

## Why this is set up to be honest, not flattering

- **`PREREGISTRATION.md` is locked before any model result was inspected.** It
  fixes the champions (HAR-RV, HMM), the primary metric (QLIKE), the splits
  (purged + embargoed walk-forward), and the win margins (≥5% QLIKE reduction
  **and** Diebold–Mariano p<0.05). The bar cannot move after seeing results.
- **HAR-RV is made *strong*, not a strawman:** log-space forecasts get the
  standard Jensen (½·Var) correction before QLIKE, so we race a fair champion.
- Every OOS number comes from expanding walk-forward with a 22-day purge/embargo
  at every train/test boundary. Feature scaling is fit on train only.

## Benchmark: Meridian vs traditional volatility models

One audited harness (`scripts/benchmark_vol.py` → `scripts/score_vol_benchmark.py` →
`scripts/render_benchmark_md.py`) produces every number below; nothing here is typed by hand.
Expanding walk-forward over 2011–2026, 12 folds, 22-day embargo, training rows purged by label
end date, all scalers fit on train only, every model's raw forecast saved to
`results/benchmark_vol_predictions.parquet`. Significance is tested on the **date-level** loss
series (Newey–West HAC), not on stacked asset-day rows. Meridian's configuration was chosen on
the 2012–2018 development period only (`PREREGISTRATION.md` Amendment 6); the table is the
untouched 2019–2026 holdout.

Compared models: random walk, 22-day mean, EWMA, GARCH(1,1), GJR-GARCH, EGARCH, HAR-RV (per
asset and pooled), LHAR, SHAR, HAR-X (+VIX), Ridge, random forest, LightGBM, TimeMixer, and
Meridian (SSM belief core, QLIKE-trained, 5-seed ensemble), alone and in a walk-forward
Granger–Ramanathan combination with HAR and LightGBM.

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

**Honest read:**
- Meridian is the best single model on the holdout, ahead of every traditional model including
  LightGBM, and the improvement over HAR holds on every asset and in nearly every year. Combining
  it with HAR and LightGBM adds little on top of Meridian alone.
- The point estimate clears the pre-registered 5% bar, but against the strongest HAR form the
  95% interval reaches just below 5%: the improvement is clearly real; its size is uncertain.
- The ML baselines (LightGBM, Ridge, HAR-X) are strong; the GARCH family, EWMA and TimeMixer are
  worse than HAR on QLIKE.
- **Hit rate does not separate the models**: almost all score ~66–68%, because volatility
  persistence makes the up/down call easy. It is reported, not used as evidence of skill.
- Three proposed Meridian improvements (per-asset HAR offset, asset embedding, early stopping)
  did *not* beat the existing configuration on the development period — recorded in
  `PREREGISTRATION.md` as a negative result.
- Limits: daily OHLC range-based realized variance (no intraday data, so HARQ / jump / Realized
  GARCH models are not tested); free delayed Yahoo/FRED data; a better volatility forecast is not
  by itself a profitable trading strategy.

**What the audit fixed** (all numbers from before it are withdrawn — see `RESULTS.md` labels):
significance computed on stacked panel rows (vastly overstated), raw vs calibrated tables on
different rows, a pooled HAR baseline instead of per-asset HAR, label leakage into test blocks
for 5/22-day targets, full-sample scaling in TimeMixer, 1-day-stale EWMA/GARCH forecasts, and
~500 invalid Yahoo FX price bars that read as zero volatility. `pytest tests/test_harness.py`
checks the leakage, calibration and significance fixes.

## Tail-risk benchmark (VaR / Expected Shortfall)

Next-day Value-at-Risk and Expected Shortfall of close-to-close returns, against the standard
risk models: historical simulation, RiskMetrics, GARCH-normal / -t, GJR-GARCH-skewed-t,
GJR-GARCH with filtered historical simulation (the reference) and with EVT (McNeil–Frey), and
CAViaR. Meridian's volatility forecast is turned into a tail forecast by filtering returns
through it and reading the tail from past standardized returns. Pre-registered in
`PREREGISTRATION.md` Amendment 7 (variant chosen on 2012–2018, reported on 2019–2026). Primary
score: the FZ0 loss, which rewards getting VaR and ES right jointly.

<!-- TAIL:START -->
Holdout 2019-01-01 → 2026-06-29 (2168 dates × 11 assets), next-day return VaR and ES at 2.5%, reference GJR-GARCH + FHS (reference). Lower FZ0 loss is better.

| Model | FZ0 loss ↓ | Δ vs reference (95% CI) | DM p | In MCS | Violations (target 2.5%) | Assets passing CC / DQ | ES ratio (realized/forecast) |
|---|---|---|---|---|---|---|---|
| Meridian + FHS | -3.6394 | -0.1226 [-0.1530, -0.0959] | <0.001 | yes | 2.54% | 11/11 · 10/11 | 0.97 |
| Meridian + EVT | -3.6387 | -0.1218 [-0.1533, -0.0949] | <0.001 | yes | 2.49% | 11/11 · 10/11 | 0.98 |
| HAR-RV + FHS | -3.6289 | -0.1182 [-0.1440, -0.0927] | <0.001 | yes | 2.58% | 11/11 · 8/11 | 0.99 |
| HAR-RV + EVT | -3.6284 | -0.1179 [-0.1449, -0.0912] | <0.001 | yes | 2.57% | 11/11 · 7/11 | 0.99 |
| Meridian + FHS + adaptive conformal | -3.6257 | -0.1080 [-0.1448, -0.0782] | <0.001 | no | 2.47% | 11/11 · 11/11 | 0.98 |
| LightGBM vol + FHS | -3.6208 | -0.1048 [-0.1306, -0.0796] | <0.001 | no | 2.58% | 11/11 · 10/11 | 0.98 |
| GJR-GARCH-skewed-t | -3.5294 | -0.0013 [-0.0124, +0.0093] | 0.404 | no | 2.60% | 10/11 · 10/11 | 0.99 |
| GJR-GARCH + EVT (McNeil-Frey) | -3.5290 | -0.0030 [-0.0054, -0.0007] | 0.007 | no | 2.67% | 11/11 · 11/11 | 0.98 |
| GJR-GARCH + FHS (reference) | -3.5255 | — | — | no | 2.71% | 11/11 · 11/11 | 0.98 |
| GARCH(1,1)-t | -3.5203 | +0.0067 [-0.0097, +0.0234] | 0.799 | no | 2.87% | 10/11 · 7/11 | 1.00 |
| GARCH(1,1)-normal | -3.5049 | +0.0179 [-0.0036, +0.0427] | 0.951 | no | 3.01% | 9/11 · 7/11 | 1.14 |
| RiskMetrics EWMA, normal | -3.4538 | +0.0626 [+0.0215, +0.1094] | 0.999 | no | 3.48% | 4/11 · 1/11 | 1.16 |
| Historical simulation (500d) | -3.3320 | +0.1836 [+0.0660, +0.3735] | 0.997 | no | 2.80% | 5/11 · 1/11 | 1.07 |

Head-to-head against HAR-RV + FHS (both volatility-filtered): FZ0 Δ -0.0037 [-0.0333, +0.0175], DM p 0.369 at 2.5%; 1% VaR quantile loss -3.1% [-7.4%, +0.2%], DM p 0.076. Full tables: `results/benchmark_tail.md`.
<!-- TAIL:END -->

**Honest read:**
- Every model that uses a realized-volatility forecast (Meridian, HAR, LightGBM) beats every
  returns-only model (the GARCH family, RiskMetrics, historical simulation, CAViaR) by a clear,
  significant margin. Meridian-based tails also pass the dynamic-quantile backtest on more
  assets (10/11) than HAR-based tails (7–8/11), i.e. their violations cluster less.
- Meridian-based tails are the best in the table, but their edge over **HAR-based tails** is
  small and **not statistically significant**. Most of the tail gain comes from using
  range-based realized volatility at all, which HAR also does.
- On 2012–2018 HAR-based tails were slightly better than Meridian's: the ranking at the top is
  not stable across periods.
- ES is well calibrated for the volatility-filtered models (realized/forecast ratio ≈ 1).

## Frozen deployment: October–November 2026 (no retraining)

The saved Meridian weights were run forward, untouched, on market data after their training period
and benchmarked against 15 other models fit at the same cut-off. Generated by `scripts/score_deploy.py`;
full tables and the WM explanation are in `DEPLOYMENT.md`, and the leak/bluff audit (`scripts/audit_deploy.py`)
is in `results/deploy/audit_report.md`.

<!-- DEPLOY:START -->
- **Snapshot B, Oct-2026** (25 target dates × 11 assets): Meridian legacy (frozen; Jensen term from pre-cutoff rows): rank 16/18, +32.8% vs HAR [-1.0, +47.0]; Meridian legacy, raw E[log RV] head (no Jensen term): rank 18/18, +97.5% vs HAR [+36.9, +123.2]; Meridian legacy + level offset fit on pre-cutoff rows (secondary): rank 11/18, +3.1% vs HAR [-2.7, +7.7]. Best model: 22-day mean (-6.0%).
- **Snapshot B, Nov-2026** (3 target dates × 11 assets): Meridian legacy (frozen; Jensen term from pre-cutoff rows): rank 13/18, +29.9% vs HAR; Meridian legacy, raw E[log RV] head (no Jensen term): rank 18/18, +105.0% vs HAR; Meridian legacy + level offset fit on pre-cutoff rows (secondary): rank 2/18, -8.7% vs HAR. Best model: Random forest (-9.1%).
- **Snapshot A, Nov-2026** (3 target dates × 11 assets): Meridian M0 (frozen, 5 seeds): rank 2/18, -8.6% vs HAR; Meridian M0, real-time VIX vintage (previous-day VIX/yields): rank 1/18, -11.5% vs HAR. Best model: Meridian M0, real-time VIX vintage (previous-day VIX/yields) (-11.5%).
- **Pre-committed live log** (2 target dates): Meridian M0 +2.1% vs frozen HAR.
<!-- DEPLOY:END -->

Honest read: neither frozen Meridian snapshot shows a statistically established gain over HAR in
this window. Snapshot A (the audited model) has only 3 target dates after its training ended, too few
for any test; its ranking there is anecdotal. Its pre-committed live log is roughly level with HAR.
Snapshot B (pre-audit, trained on uncleaned bars) is worse than HAR as deployed, and its raw head
significantly so. That is consistent with the audit that withdrew it, not a new failure. The Oct
(25-date) intervals are wide (several percent either side), so months more of forward data are
needed before any deployment claim.

## Validation report (PREREGISTRATION.md Amendment 8)

`results/Meridian_Validation_Report.pdf` gathers the full validation in one document: the frozen
deployment, the longest post-training window for the legacy model and a frozen Meridian-WM (V1/V3),
Meridian-WM in the audited walk-forward harness (V2, including its pre-registered NaN failure and
the Amendment 8a re-run), tail risk, the audit, and the defects found along the way. It is generated
by `scripts/build_validation_pdf.py`, and every number in it is read from the results JSON files.

## Forward test

`scripts/forward_test.py` freezes Meridian M0 and per-asset HAR (trained once, weights hashed)
and appends a forecast for every next trading day to `results/forward/forecast_log.csv`. Only
forecasts logged before their target day count. Run `python scripts/forward_test.py predict`
each trading day after the close, and `python scripts/forward_test.py score` to score it.
Only a few days are scored so far (see the frozen-deployment section above); a meaningful read needs months of new data.

## Layout

```
PREREGISTRATION.md        locked evaluation protocol + win criteria
meridian/
  data.py                 Yahoo (daily OHLC) + FRED (VIX, yields), cached
  features.py             realized-variance estimators, HAR features, targets
  evalproto.py            purged walk-forward, QLIKE, MZ-R2, walk-forward calibration/combination
  scoring.py              benchmark statistics: date-level HAC DM, MCS, bootstrap CIs, IC, hit rate
  riskscore.py            tail statistics: FZ0 / quantile loss, Kupiec, Christoffersen, DQ, ES test
  lgbm_worker.py          LightGBM baseline (separate process: cannot share one with PyTorch-MPS)
  baselines.py            HAR-RV, AR(1), AR(3), EWMA, GARCH(1,1)
  model.py                SSM belief core + JEPA predictor + SIGReg + heads
  windows.py              leakage-safe windowing / scaling
  regimes.py              HMM + Meridian-belief regime metrics
scripts/
  benchmark_vol.py        audited benchmark: fits all models per fold, saves raw forecasts
  score_vol_benchmark.py  scores saved forecasts -> results/benchmark_vol.json
  render_benchmark_md.py  JSON -> results/benchmark_vol.md + README/MODEL_CARD summary blocks
  benchmark_tail.py       VaR/ES of every risk model per fold -> benchmark_tail_predictions.parquet
  score_tail_benchmark.py scores them -> results/benchmark_tail.{json,md} + summary blocks
  forward_test.py         frozen forward test: freeze / predict / score
  run_baselines.py        fit baselines -> results/baseline_predictions.parquet
  run_meridian.py         walk-forward train/eval -> meridian_predictions.parquet
  compare.py              volatility verdict vs HAR-RV (pre-registered)
  compare_regimes.py      regime verdict vs HMM
  fit_final.py            final model + demo_state.json for the live demo
demo/                     self-contained market-state dashboard
tests/test_harness.py     leakage / calibration / significance / data-cleaning tests
tests/test_tail.py        tail-loss, backtest-calibration and no-look-ahead tests
```

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn torch pyarrow arch statsmodels hmmlearn lightgbm pytest
python -m pytest tests -q                                 # harness correctness checks
BENCH_VARIANTS=M0 BENCH_ENSEMBLE=5 python scripts/benchmark_vol.py   # ~20 min on Apple MPS
python scripts/score_vol_benchmark.py results/benchmark_vol_predictions.parquet
python scripts/render_benchmark_md.py                      # regenerates the tables in this README
python scripts/benchmark_tail.py                           # ~2 min; needs the vol predictions
python scripts/score_tail_benchmark.py                     # tail tables + README blocks
python scripts/forward_test.py predict                     # daily, after the close
```

## Provenance & relationship to prior work

- **EB-JEPA** — Meta FAIR, `facebookresearch/eb_jepa` (Apache-2.0). Predict latent
  representations of future context; the "energy" (latent prediction error) is our
  surprise score. EB-JEPA ships image/video/action-conditioned examples only — **no
  tabular/time-series path** — so Meridian implements the JEPA mechanism from
  primitives rather than forking that codebase. Concepts borrowed: energy objective,
  EMA target encoder, variance/covariance-style anti-collapse.
- **Fin-JEPA** — `cedricwyh/fin-jepa` (MIT), Wang et al. JEPA for financial time
  series: 64-d latents, SIGReg (λ=0.1), PriceEncoder (MLP) + **Transformer**
  predictor. **Two deliberate divergences:** (1) Meridian uses an **SSM belief
  core** (per project spec), *not* a Transformer predictor; (2) Meridian **forecasts
  volatility and benchmarks HAR-RV**, which Fin-JEPA does not — Fin-JEPA's downstream
  eval is a prediction-error ("VoE") study that itself reports the surprise signal is
  weak-to-negligible, corroborating our caution about the surprise score.
- **SIGReg** (LeJEPA line) — regularize embeddings toward an isotropic Gaussian to
  prevent collapse without stop-gradient tricks.
- **HAR-RV** — Corsi (2009), the volatility benchmark that is famously hard to beat.
- Data is free/delayed (Yahoo, FRED) per project scope.

Current head-to-head numbers: `results/benchmark_vol.md` (generated). `RESULTS.md` is a dated lab log; its sections are labeled as not reproduced or superseded.
# RENWM
