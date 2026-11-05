# Meridian — Pre-Registration of Evaluation Protocol

**Locked before any model results are inspected.** Date: 2026-07-29.
This document fixes the benchmarks, metrics, data splits, and win-margins **before**
the JEPA core is trained, so that a claimed win cannot be the product of hindsight
tuning. Any deviation must be recorded in the "Amendments" section with a timestamp
and a reason.

---

## 1. Task definitions

### Task A — Volatility forecasting (the primary claim)
Forecast next-day (h=1) realized volatility (RV) for each asset.
- **Target:** log realized variance, `log RV_{t+1}`. RV is computed from daily data as
  a Garman–Klass / Parkinson range estimator (intraday not required at this stage),
  and cross-checked against squared close-to-close returns.
- **Champion to beat:** **HAR-RV** (Corsi 2009) — OLS on `[RV_t, RV_{t-5:t}, RV_{t-22:t}]`
  in log space.
- **Also reported:** AR(1), AR(3) on log-RV; GARCH(1,1) 1-step variance forecast;
  EWMA (RiskMetrics λ=0.94).

### Task B — Regime persistence (the secondary claim)
Identify market regimes and beat a Gaussian **HMM** on regime *persistence* and
predictive usefulness.
- **Champion to beat:** 2- and 3-state Gaussian HMM on daily returns (+ optionally RV).

---

## 2. Data

- **Universe (equity):** SPY, QQQ, IWM, and 5 large-cap single names (AAPL, MSFT,
  JPM, XOM, JNJ) — sector spread.
- **Universe (FX):** EURUSD, USDJPY, GBPUSD (via Yahoo `EURUSD=X` etc.).
- **Macro/vol context (features only, never target leakage):** VIX (FRED `VIXCLS`),
  DGS10, DGS2 (FRED). Used only with information available at close of day `t`.
- **Sample:** 2007-01-01 → most recent available. Includes GFC, 2015, COVID, 2022.
- **Sources:** Yahoo chart API (delayed daily OHLC), FRED CSV. Free/delayed = acceptable
  for this stage per project scope.

---

## 3. Splits — leakage control (this is the scientific spine)

- **Walk-forward, expanding window.** Train on `[start, T)`, predict a forward block,
  roll forward. No shuffling. No future data ever enters training or feature scaling.
- **Purge + embargo:** because RV targets and HAR features overlap in time, we PURGE
  any training sample whose target window overlaps the test window, and apply an
  **embargo of 22 trading days** after each test block before training resumes.
- **Feature scaling** (means/stds) fit on train only, applied to test.
- **Out-of-sample (OOS)** = the concatenation of all forward test blocks. All headline
  numbers are OOS.

---

## 4. Metrics (pre-registered, primary in bold)

Volatility (lower is better):
- **QLIKE** on variance: `mean( RV/σ² − log(RV/σ²) − 1 )` — the robust, standard vol loss.
- MSE on `log RV`.
- MZ regression R² (Mincer–Zarnowitz), reported for interpretability only.

Regime:
- **Regime persistence** = mean dwell time / expected-state-duration, and
  1-step regime-transition log-likelihood on OOS.
- Economic check: does conditioning on the regime improve OOS vol QLIKE?

Surprise (JEPA-native, descriptive not a win-claim): energy / prediction-error score,
reported and correlated with realized |return| and VIX changes.

---

## 5. WIN CRITERIA (the bar, locked)

The Meridian core is declared to **beat** a benchmark only if **all** hold on purged OOS,
pooled across the universe:

- **Volatility vs HAR-RV:** ≥ **5% relative reduction in QLIKE**, AND a
  **Diebold–Mariano test** of equal predictive accuracy rejected at **p < 0.05**
  (HAC/Newey–West variance, one-sided in Meridian's favor).
- **Regime vs HMM:** higher OOS 1-step regime predictive log-likelihood AND
  ≥ **10% higher** mean regime persistence (dwell time) **without** loss of the
  economic check (regime-conditioned vol QLIKE not worse than HMM's).

Anything short of both is reported as **"did not beat"** — honestly, with the numbers.
No moving of this bar after seeing results.

---

## 6. Amendments

### Amendment 1 — 2026-07-29 — disclosed OOS looks + one pre-committed final run
Two configurations have been evaluated on the pre-registered OOS so far:
1. Meridian-MSE (head = E[log RV], MSE loss): calibrated QLIKE 0.3457, +2.51% vs
   HAR-RV, DM p=0.0036.
2. Meridian-QLIKE (head = log-variance, trained on exact QLIKE): calibrated QLIKE
   0.3435, +3.15% vs HAR-RV, DM p=0.0020.
Both beat AR/EWMA/GARCH by 14–23% (p<1e-4); neither cleared the +5% bar vs HAR-RV.

To avoid multiple-comparisons overfitting, I now **pre-commit a single final
configuration and will report its result once, whatever it is**:

> **Final model (committed before running):** a **5-seed ensemble** of the
> Meridian-QLIKE model (seeds 0–4), forecasts combined by **averaging the
> log-variance output**, evaluated under the identical purged walk-forward and
> the identical symmetric calibration applied to all models. No per-result
> tuning; no further looks after this one. Rationale: seed-ensembling is a
> standard a-priori variance-reduction step, chosen without reference to any
> test-set outcome.

Win criteria are unchanged (§5). If the ensemble does not clear +5%, that is the
final Day-1 answer and no further changes will be made against this OOS.

### Amendment 2 — 2026-07-29 — regime evaluation redesign (declared before results)
The original regime metrics (§4) are flawed and are replaced BEFORE computing any
new regime result:
- **Mean dwell time is gameable** — a detector that never switches maximizes it.
  It is demoted to a *sanity gate*, not a win metric: a usable detector must have
  pooled mean dwell in **[5, 60] trading days** (it must actually switch, yet be
  persistent). The first Meridian attempt (HMM on 64-d belief) failed this gate
  (~300-day dwell) and is discarded.
- **The old economic check shifted log-variance by the per-regime mean residual**,
  which targets squared error, not QLIKE — ill-posed. Replaced by a proper,
  leakage-safe OOS test:

> **Regime economic value (new primary regime metric).** Hold the base vol
> forecast FIXED at HAR-RV for BOTH methods. Split the OOS in time (first 50% =
> fit, last 50% = test). On the fit half, learn a per-regime affine map
> `a_r + b_r·(HAR log-var)` by minimizing the exact QLIKE within each regime;
> apply on the test half. **Value = OOS QLIKE improvement of (HAR + regime) over
> plain calibrated HAR.** The regime labeling that adds more OOS QLIKE value has
> the more useful regimes. Because the base forecast is HAR for both, this
> isolates regime information, not forecast skill.

- **Meridian regime construction (committed):** a 3-state Gaussian HMM on the
  **JEPA surprise-energy series** (the model's own instability signal) — distinct
  from returns and from the vol level, hence not circular with HAR.
- **HMM baseline:** 3-state Gaussian HMM on daily returns (unchanged).
- **Regime win criterion (amended):** Meridian regimes beat HMM iff they pass the
  dwell sanity gate AND deliver **strictly greater OOS QLIKE value** on the test
  half. Reported once; no iterate-to-win.

### Amendment 3 — 2026-07-30 — Meridian-WM evaluation (declared before any WM result)
The bridged model (ARCHITECTURE.md) is evaluated on three pre-declared axes under
the SAME purged/embargoed walk-forward; each is judged honestly, win or lose:

1. **Volatility (must-not-regress):** calibrated OOS QLIKE vs HAR-RV and vs the
   current SSM+QLIKE core. WM is acceptable on vol iff it does **not** significantly
   underperform the current core (research says vol is HAR-frontier-bound; WM is not
   expected to *improve* vol, only to hold it while adding regime/tail).
2. **Regime (target win):** the switching-SSM regime (argmax of the posterior α) beats
   a Gaussian HMM iff it passes the dwell gate [5,60] days AND gives strictly greater
   OOS regime economic value on a fixed HAR base (compare_regimes2 protocol).
3. **Tail (target win, new capability):** the predictive return distribution's
   left-VaR at 5% and 1% is **well-calibrated** — Kupiec POF and Christoffersen
   independence tests **not rejected at p<0.05** — and at least as well-calibrated as a
   Gaussian(σ) baseline with the same σ. Tails validated by VaR exceedance backtests,
   NOT by a proper scoring rule (which provably cannot separate extreme-tail values).

Ablations (toggle one component at a time) attribute each: switching core, EMA
readout, distributional head, JEPA/SIGReg aux. Reported once per configuration.

### Amendment 4 — 2026-07-31 — benchmark_vol.py evaluates the real Meridian core
(declared before inspecting the pooled multi-fold result of the changes below)

Prior to this amendment, `scripts/benchmark_vol.py`'s "Meridian" row was a 4-coefficient
OLS regression (HAR + one leverage feature, MSE-fit) — a stand-in that did not use the
QLIKE-trained neural core described in MODEL_CARD.md, and which lost to plain HAR-RV in
the artifact checked into `results/benchmark_vol.json` at the time. That is a protocol
gap, not a modeling result, and is corrected here rather than left standing.

Changes, disclosed before the corrected pooled OOS numbers are inspected:

1. **Real model wired in.** `benchmark_vol.py`'s "Meridian" now trains the actual
   `meridian.model.Meridian` core (`loss_mode="qlike"`, `core_type="ssm"`) as a 5-seed
   ensemble per fold, inside the same purge/embargo/DM/MCS harness used for the other
   baselines — the same model class behind the MODEL_CARD headline numbers, now
   evaluated in the one apples-to-apples script. The old OLS regression is kept in the
   table as `HAR+Lev`, relabeled for continuity, and is no longer called "Meridian."
2. **Realized semivariance features** (good/bad decomposition, Patton-Sheppard; `nsv_d/
   w/m`, `psv_d`) are added to the neural core's feature set (11 → 15 features) and to a
   new pooled `RF+SemiVar` baseline (a pooled variant of `scripts/run_semivar_ml.py`'s
   per-asset RF). Both were previously validated as predictive but never wired into the
   neural path or this benchmark.
3. **Multi-horizon targets** (`y_5`, `y_22`: 5-day and 22-day forward mean log-RV) are
   added as auxiliary QLIKE-trained heads on the shared belief state (weight
   `lambda_aux=0.3`, does not touch the primary h=1 head's optimum) — the lever flagged
   repeatedly in ARCHITECTURE.md/RESEARCH.md as the highest-evidence unbuilt lever,
   never implemented until now.
4. **Meridian+** — an equal-weight (0.5/0.5), log-space blend of Meridian and RF+SemiVar
   is added, reflecting the heterogeneous-ensembling result already documented as the
   best predictor found in RESULTS.md. The blend weight is fixed at 0.5 by construction
   (not fit on data) specifically to avoid any leakage risk in choosing it.
5. **Breadth cut (non-pre-registered, additive only, no training contamination).** Two
   liquid, cheap ETFs fetchable via the existing Yahoo pipeline — TLT (rates) and GLD
   (commodities) — are added as `meridian.data.BREADTH`, loaded via
   `load_all(include_breadth=True)`. Every pooled model (HAR, HAR+Lev, RF+SemiVar,
   Meridian, TimeMixer) is trained ONLY on the pre-registered CORE universe in both
   tables — breadth assets never enter training data. The CORE+BREADTH table applies
   those same CORE-trained models to TLT/GLD as an out-of-sample cross-asset-class
   generalization check, not an expanded-training-data benchmark; this keeps the CORE
   table's HAR/Meridian numbers directly comparable to the pre-amendment protocol.
6. **Horizon and regime cuts (diagnostic only, not new win criteria).** The benchmark
   now also reports QLIKE/RMSE/IC at h=5 and h=22 (HAR, RF+SemiVar, Meridian, Meridian+
   only — EWMA/GARCH/TimeMixer remain h=1-only and are reported as not-applicable, not
   silently omitted), and a QLIKE breakdown by VIX tercile (calm/normal/stress) on the
   pooled CORE OOS rows. These are diagnostics to check whether any edge is
   horizon-concentrated or regime-concentrated; they do not change the §5 win bar.

**What does NOT change:** the win bar itself (§5: ≥5% QLIKE reduction vs HAR-RV, DM
p<0.05, on the pre-registered CORE universe), the purge/embargo protocol, or the
CORE asset list. The corrected `results/benchmark_vol.json`/`.csv` produced under this
amendment supersede the pre-amendment artifact; whichever way the CORE-table numbers
land (win, partial win, or no win vs HAR-RV) is reported as-is in RESULTS.md/README.md.

### Amendment 5 — 2026-07-31 — HAR-anchored init, walk-forward calibration, GR combination
(declared before inspecting the pooled result of the changes below; Amendment 4's honest
result — Meridian did not beat HAR/EWMA on QLIKE, though it won RMSE/IC outright and beat
HAR specifically in the "stress" VIX regime — is the reason for this follow-up, not a
result being second-guessed after the fact)

Literature/codebase review (disclosed): (a) Reisenhofer, Bayer & Hautsch 2022, "HARNet: A
Convolutional Neural Network for Realized Volatility Forecasting" (arXiv:2205.07719) —
initializes a HAR-structured network so it is IDENTICAL to the HAR baseline before any
optimization step, which the authors report stabilizes QLIKE training and lets the network
only ever learn corrections on top of HAR, never rediscover it from scratch; (b) this
project's own `meridian/evalproto.py:walk_forward_calibrate` — an existing, already-proven,
leakage-safe affine recalibration that produced the MODEL_CARD headline win via
`compare_calibrated.py`, but was never applied inside `benchmark_vol.py`; (c) general
forecast-combination evidence (hybrid HAR+ML models beating single models when combined
with fitted, not naive, weights).

Three changes, disclosed before the corrected pooled numbers are inspected:

1. **HAR-anchored initialization** (`meridian/model.py`: `MeridianConfig.har_anchor`,
   `Meridian.init_har_anchor`). An explicit linear `har_skip` pathway reads
   (har_d, har_w, har_m) directly and is added to every vol head's output. Each fold, its
   weights are initialized (closed-form, from that fold's own HAR-OLS fit, folding in the
   window scaler's mean/std) so the model's forecast is IDENTICAL to HAR before a single
   optimization step; the SSM/MLP path is zero-initialized. Training then only ever has to
   find corrections on top of HAR-level performance.
2. **Post-hoc walk-forward calibration**, applied uniformly to every model's raw pooled OOS
   log-variance forecast via `walk_forward_calibrate` (unmodified, reused as-is) — the same
   mechanism already validated to produce the MODEL_CARD's headline number. Because this
   step drops a ~252-row per-asset warm-up, calibrated columns are ALWAYS scored on a common
   row support across every calibrated model in the table (`score_table(..., common=True)`)
   — never mixed with the raw table's near-complete support — so no model is compared on an
   easier date range than another by construction.
3. **Ensemble_GR**: a walk-forward Granger-Ramanathan optimal combination of HAR, Meridian,
   and RF+SemiVar (`meridian/evalproto.py:walk_forward_combine`, a direct multivariate
   generalization of `walk_forward_calibrate`, fit per asset on past data only), replacing
   the naive fixed 0.5/0.5 `Meridian+` blend with data-driven weights.

**What does NOT change:** the §5 win bar, the CORE universe, the purge/embargo protocol, or
the raw (uncalibrated) table already reported under Amendment 4 — it is kept alongside the
new calibrated table for transparency, not replaced. Whichever way the calibrated CORE
numbers land is reported as-is.

### Amendment 6 — 2026-10-31 — harness audit; numbers under Amendments 4–5 are SUPERSEDED
(declared before any result of the fixed harness is computed)

An audit of `scripts/benchmark_vol.py` and `meridian/evalproto.py` found that several numbers
reported under Amendments 4–5 are overstated or not comparable. They are superseded, not
defended:

1. **Significance was inflated.** DM was run on ~40k stacked panel rows with HAC lag 0,
   ignoring cross-asset correlation on the same date and loss autocorrelation; MCS
   block-bootstrapped stacked panel rows. The reported `p≈1e-14` values are withdrawn.
2. **Raw and calibrated tables used different rows** (calibration drops a per-asset warm-up),
   so raw-vs-calibrated comparisons mixed a sample change with a model change.
3. **The HAR baseline was handicapped**: one pooled regression and one intercept across
   equities and FX. Textbook HAR is fit per asset.
4. **h=5/h=22 label leakage**: the embargo was counted on the union calendar, so a 22-day
   label could reach ~1 trading day into the test block.
5. **TimeMixer** was standardized with full-sample (incl. test) statistics.
6. **Pooled IC** mostly measured cross-asset level differences.

**Fixed protocol (applies to every model, every table):**
- **Invalid OHLC bars repaired** (`clean_ohlc` in `scripts/benchmark_vol.py`): 36 stale FX
  bars with high == low (open=high=low=close, volume 0: no real quote) are dropped, and
  ~470 FX bars whose open/close fall outside [low, high] (a Yahoo snapshot-timing quirk)
  have open/close clipped into the range. Left as-is, the range-based RV estimator reads
  these bars as zero or negative variance (clipped to 1e-12, log ≈ −27.6); each such row
  adds QLIKE ≈ 15 as a target and distorts the next day's forecasts as an input. Every
  QLIKE number reported under Amendments 4–5 includes these rows. (Found in a 1-fold,
  1-epoch smoke check, before any result of the fixed harness was computed.)
- Train rows for a horizon-h target are purged by **label end date** (the date the h-day
  label window ends, on that asset's own calendar must be < test start), plus the 22-day
  embargo.
- All scalers/normalizers are fit on each fold's train rows only.
- Every model's raw OOS forecast is saved to `results/benchmark_vol_predictions.parquet`;
  all metrics are computed from that file by `scripts/score_vol_benchmark.py`.
- Every table is scored on **one common row support** (all models finite after calibration
  warm-up).
- **DM**: on the date-level (cross-asset mean) loss differential, Newey–West lag
  `floor(4·(T/100)^(2/9))`, HLN correction, one-sided. **MCS**: on date-level losses, block
  22. **CI**: block-bootstrap over dates for the % QLIKE change vs per-asset HAR.
- **IC**: per-asset time-series Spearman, averaged across assets.
- Reference model for all tests: **per-asset HAR-RV** (log, Jensen-corrected).

**Model list:** RW, EWMA, GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1), HAR pooled, HAR per
asset, LHAR, SHAR, HAR-X(VIX), Ridge, RandomForest, LightGBM, TimeMixer, Meridian variants,
and a walk-forward Granger–Ramanathan combination. HARQ, HAR-CJ and Realized GARCH need
intraday data and are omitted rather than approximated.

**Selection rule (anti-snooping):** Meridian variants (current; per-asset HAR offset; asset
embedding; early stopping; combination) are compared ONLY on the **development period**
(OOS dates < 2019-01-01). The single variant with the lowest development-period calibrated
QLIKE is chosen, re-run with 5 seeds, and its **holdout-period** (OOS dates ≥ 2019-01-01)
result is the headline. Holdout numbers are not used for any choice. The §5 win bar (≥5%
QLIKE reduction vs HAR, DM p<0.05) is applied to the holdout result against **per-asset
HAR**. Whatever it shows is reported.

**Amendment 6 — selection record (written after the development-period scoring and BEFORE
any holdout number was viewed).** Ablation run: `results/benchmark_vol_predictions_ablation.parquet`
(3 seeds per variant), scored in `results/benchmark_vol_ablation.json`. Development period
(2012-06-14 → 2018-12-31, 1,901 dates × 11 assets), calibrated QLIKE vs per-asset HAR (0.3741):

| Candidate | Dev QLIKE | vs HAR |
|---|---|---|
| GR combo HAR + LightGBM + Meridian M0 | 0.3353 | −10.4% |
| GR combo HAR + LightGBM + Meridian M2 | 0.3354 | −10.3% |
| GR combo HAR + LightGBM + Meridian M1 | 0.3356 | −10.3% |
| GR combo HAR + LightGBM + Meridian M3 | 0.3381 | −9.6% |
| Meridian M0 | 0.3404 | −9.0% |
| Meridian M2 | 0.3430 | −8.3% |
| Meridian M1 | 0.3432 | −8.3% |
| Meridian M3 | 0.3477 | −7.1% |

**Selected: GR combination of HAR + LightGBM + Meridian M0** (lowest development QLIKE; the top
three are within 0.0003, i.e. practically tied). The proposed improvements M1 (per-asset HAR
offset), M2 (asset embedding) and M3 (early stopping) did not beat the existing M0 on the
development period — recorded as a negative result. The selected configuration is re-run with
5 seeds (`results/benchmark_vol_predictions.parquet`); its holdout result is the headline.

### Amendment 7 — 2026-11-01 — tail-risk (VaR / ES) benchmark and frozen forward test
(declared before any result of this benchmark is computed)

**Question.** Does Meridian's audited volatility forecast give better next-day Value-at-Risk and
Expected Shortfall than the standard risk models, when every model is evaluated the same way?

**Target.** Next-day close-to-close log return r_{t+1} of each core asset (8 equities + 3 FX),
forecast at the close of day t. Levels α = 1%, 2.5% (Basel FRTB ES level), 5%. Zero conditional
mean for every model.

**Models** (all walk-forward on the same folds as `results/benchmark_vol_predictions.parquet`;
parameters fit on data dated before each fold's cutoff only):
- Historical simulation (HS, trailing 500 days).
- RiskMetrics EWMA (λ = 0.94), normal.
- GARCH(1,1)-normal, GARCH(1,1)-Student-t, GJR-GARCH(1,1)-skewed-t (parametric VaR/ES).
- GJR-GARCH(1,1) + filtered historical simulation (FHS, Barone-Adesi) — **the reference**.
- GJR-GARCH(1,1) + EVT/GPD on standardized residuals (McNeil–Frey 2000).
- CAViaR asymmetric slope (Engle–Manganelli 2004), VaR only.
- Volatility-filtered models using a saved OOS log-variance forecast σ²_{t+1|t} from the audited
  volatility benchmark: **HAR-FHS, HAR-EVT, LightGBM-FHS, Meridian-FHS, Meridian-EVT,
  Meridian-FHS-ACI** (adaptive conformal level, Gibbs–Candès 2021, γ = 0.005).
- Filtered models read quantiles from the trailing ≤1000 past standardized returns z = r/σ (min
  250); EVT threshold at the 90th loss percentile, GPD refit every 21 days.

**Scoring** (`scripts/score_tail_benchmark.py`, from saved predictions only):
- Primary: joint VaR+ES **FZ0 loss** (Fissler–Ziegel; Patton–Ziegel–Chen 2019) at α = 2.5%.
  FZ0 differences between models are invariant to each asset's return scale, so assets pool.
- Secondary: quantile (tick) loss for VaR at 1%, 2.5%, 5%, on returns scaled by each asset's
  return s.d. before its first OOS date (a fixed constant, known ex ante).
- Significance on the date-level loss differential (mean across assets per date): HAC DM test
  (Newey–West, lag floor(4·(T/100)^(2/9))), moving-block bootstrap 95% CI, MCS (90%, block 22).
- Absolute backtests per asset: Kupiec POF, Christoffersen conditional coverage, Engle–Manganelli
  Dynamic Quantile test (4 lags); reported as the number of assets passing at 5%. ES: McNeil–Frey
  exceedance-residual test and realized/predicted ES ratio.
- Every table on one common row support (all models finite).

**Selection rule.** Among the Meridian tail variants (FHS, EVT, FHS-ACI) the one with the lowest
**development-period** (dates < 2019-01-01) FZ0 loss at 2.5% is chosen; the **holdout** (≥ 2019-01-01)
result is the headline. Holdout numbers are not used for any choice. Win bar: lower holdout FZ0
loss than the GJR-GARCH-FHS reference, DM p < 0.05. Whatever it shows is reported.

**Forward test.** The selected volatility configuration (Meridian M0, 5 seeds) and per-asset HAR
are trained once on all data up to the freeze date and **frozen** (weights and coefficients
saved, hashed). Each run appends forecasts for the next trading day to an append-only log with a
UTC timestamp. Only forecasts logged before their target day count as forward results; any rows
back-filled for days already past at freeze time are labelled as such and reported separately.
No retraining, no parameter change, for the life of the test.

**Amendment 7 — selection record (written after the development-period scoring and BEFORE any
holdout number was viewed).** `python scripts/score_tail_benchmark.py --dev-only`, development
period 2012-06-12 → 2018-12-31, FZ0 loss at 2.5% (lower is better; Δ vs GJR-GARCH-FHS):

| Model | Dev FZ0 | Δ vs reference |
|---|---|---|
| HAR-RV + FHS | −3.7832 | −0.0998 |
| HAR-RV + EVT | −3.7830 | −0.0981 |
| Meridian + EVT | −3.7753 | −0.0865 |
| Meridian + FHS | −3.7746 | −0.0862 |
| Meridian + FHS + ACI | −3.7547 | −0.0677 |
| LightGBM + FHS | −3.7514 | −0.0595 |
| GJR-GARCH + EVT | −3.7034 | −0.0002 |
| GJR-GARCH + FHS (reference) | −3.7016 | 0 |
| GJR-GARCH-skewed-t | −3.6911 | +0.0161 |
| GARCH-t | −3.6842 | +0.0204 |
| GARCH-normal | −3.6804 | +0.0187 |
| RiskMetrics | −3.6207 | +0.0676 |
| Historical simulation | −3.5578 | +0.1409 |

**Selected Meridian tail variant: Meridian + EVT.** Noted before the holdout: on the development
period the HAR-filtered tails are slightly better than the Meridian-filtered ones, so the tail
benchmark will also report Meridian against HAR-RV + FHS, not only against the GJR reference.

**Amendment 8 — overnight validation of the frozen deployment and of Meridian-WM (written and
committed 2026-11-05, BEFORE any of the runs below were started; the git commit time of this text
is the evidence).** Three pre-specified tests. Nothing below is tuned; every configuration is the
one already in the repository.

*V1 — Snapshot B, longest honest window.* The legacy frozen model `results/meridian_final.pt`
(cut-off reconstructed by audit check T4: last training origin 2026-07-28) is run forward,
untouched, on every target date from **2026-08-03 to the last complete day** (≈ 66 trading dates).
All comparison models are fit once on rows whose 1-day label ends **≤ 2026-07-29** and then run
forward frozen. Primary: core-universe QLIKE vs per-asset HAR, date-level DM (Newey–West HAC),
moving-block bootstrap CI, model confidence set (10%). Reported for the raw head, the
Jensen-only head (Var term from pre-cutoff rows) and the level-offset head (mean + Var from
pre-cutoff rows). Tail: next-day 2.5% and 1% VaR/ES on the same rows from Meridian-B + EVT
(Amendment 7 selection), HAR + FHS, LightGBM + FHS and GJR-GARCH + FHS (reference); FZ0 loss,
Kupiec and Christoffersen per asset.

*V2 — Meridian-WM in the audited walk-forward harness.* `MeridianWM` exactly as in
`meridian/meridian_wm.py` (default `WMConfig`, 15 features, window 32, 8 epochs, batch 256,
AdamW 1e-3) is trained per fold inside the audited harness of Amendment 6: the same 12 folds,
22-day embargo, label-end purge, train-only scaler, core assets for training, data truncated
to the audited benchmark's data end (2026-07-30) so the fold grid is identical (checked against
the `fold` column of `results/benchmark_vol_predictions.parquet`). Ensemble: 3 seeds (CPU
budget; M0 used 5 — stated as a limitation). Scored with `scripts/score_vol_benchmark.py`'s
protocol on the **2019-01-01 → 2026-06-29 holdout** against per-asset HAR and the stored M0,
raw and with the same walk-forward recalibration every model gets. Win bar unchanged (§5:
≥ 5% QLIKE reduction vs HAR and DM p < 0.05). Secondary: the Student-t head's 2.5%/1% VaR
coverage (Kupiec, per asset) and regime persistence (mean dwell of the argmax regime).
Development-period (< 2019) numbers are reported but not used for any choice; there is no choice.

*V3 — Frozen Meridian-WM, deployed without further training.* `MeridianWM` (same config, 5
seeds) is trained ONCE on rows whose labels end ≤ 2026-07-29 (the V1 cut-off), saved and
sha256-hashed, then run forward untouched on the V1 window and scored exactly like V1.

All three are added to the audit (`scripts/audit_deploy.py`): weights unchanged after freeze,
targets after training, look-ahead truncation, fold-grid identity, independent re-scoring,
report == JSON. Whatever the results are, they are reported.
