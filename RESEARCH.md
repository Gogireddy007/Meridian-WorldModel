# Meridian — JEPA / World-Model Landscape (deep research, 2026-07-30)

Fan-out deep research (105 agents, 23 primary sources, 103 claims extracted, 25
adversarially verified → 24 confirmed / 1 refuted). Question: *which JEPA / EBM /
world-model variant has a concrete mechanism to beat a plain SSM+QLIKE core at
daily financial volatility or regime detection, and how to adapt it.*

## The headline finding (a negative-space result)

**No surveyed variant has any published out-of-sample evidence of beating
HAR-RV/GARCH/HMM on daily finance.** The most on-point benchmark — **Brini 2026**
(arXiv 2607.05291, 50 assets: 40 equities, 5 FX, 5 futures) — finds **9 zero-shot
time-series foundation models fail to uniformly beat a well-specified Log-HAR**;
only Tiny Time Mixers edges it (~1.3–1.8%), and a Mincer–Zarnowitz recalibration
shows "much of the short-horizon advantage reflects better-scaled forecasts rather
than better prediction of volatility dynamics." The HAR family forms a 90% Model
Confidence Set. **This independently corroborates Meridian's own ablation:** a plain
SSM+QLIKE core is near the achievable frontier for daily vol; latent-prediction JEPA
+ SIGReg add ~nothing.

## What the evidence de-prioritizes

- **Hierarchical / H-JEPA and hyperbolic / non-Euclidean JEPA** — requested, but
  produced **zero surviving verified claims** with a concrete daily-vol mechanism.
  (Non-Euclidean SPDNet/U-SPDNet regime models even *underperform* an equal-weight
  benchmark OOS under purged+embargo backtesting.) Treat as unsupported here.
- **EB-JEPA** (Apache-2.0) — clean reference implementation of the *same* JEPA+SIGReg
  idea Meridian already has; vision/navigation only; no new vol mechanism.
- **TS-JEPA / LaT-PFN** — representation or generic forecasting; LaT-PFN is univariate
  and doesn't uniformly beat ARIMA; no financial backtests.
- The claim "the JEPA *objective* is the primary driver of gains" was **REFUTED**
  (1-2 vote) — convergent with Meridian's ablation and Fin-JEPA's weak surprise.

## Ranked shortlist (the only two that survived as actionable)

### #1 — Switching / deep state-space (DS3M, Kalman-VAE)  ·  axis: REGIME  ·  highest expected value
Mechanism: keep the **discrete regime latent strictly Markov** and push non-Markov
dynamics into RNN-driven **continuous** latents (DS3M, arXiv 2106.02329, Int'l J.
Forecasting 2025). This yields "longer regime durations… compared to the chaotic
switching seen in other models" — i.e. it **directly targets Meridian's FAILED
regime-persistence-vs-HMM claim**, which is a *duration* problem, not a vol-accuracy
problem. Kalman-VAE (arXiv 1710.05741) is the soft-mixture-of-K-LGSSMs alternative
with exact Kalman inference. **Not a JEPA** — a world-model. Evidence is
forecasting-accuracy / qualitative, not a head-to-head regime win vs HMM on daily
equity+FX (that experiment is unpublished — an opportunity).
Experiment: replace the post-hoc HMM regime layer with a DS3M (2–4 regimes) on the
same purged walk-forward; evaluate regime persistence/duration AND economic value
vs HMM; keep the SSM+QLIKE vol head unchanged.

### #2 — CF-JEPA EMA-encoder routing  ·  axis: VOL  ·  cheap / low-risk  ·  **is a JEPA**
Mechanism (CF-JEPA, arXiv 2606.07031, June 2026): mask-free *forward* prediction,
plus an observed **online-vs-EMA encoder asymmetry** — the online encoder learns
high-rank discriminative features, the EMA target encoder is smoother and better for
*regression*; routing forecasting to the EMA encoder cut multivariate MSE 27% "at no
additional training cost." Confidence **medium** (single recent preprint, generic
benchmarks, unreplicated). For Meridian this is a near-zero-cost test: read the vol
forecast off the EMA target encoder vs the online encoder.
**Status: IMPLEMENTED & TESTED → REPLICATES.** Dual vol head (online vs EMA-target
readout), `MERIDIAN_DUALVOL=1`. Result: EMA-target readout QLIKE 0.3361 vs online
0.3403 = **+1.21%, DM p=0.0027**. The CF-JEPA asymmetry replicates on daily
equity+FX vol — modest but a genuine, cheap, JEPA-specific win. See RESULTS.md.

## Honest caveats (from the research)
1. Beating a well-specified HAR/SSM+QLIKE on daily vol is genuinely hard; most "wins"
   in the literature are calibration or outlier-driven. Expect little QLIKE lift from
   any JEPA.
2. No surveyed variant has ANY daily equity+FX vol/regime backtest vs HAR/GARCH/HMM —
   every finance judgment is analogical transfer.
3. CF-JEPA's 27% asymmetry and TTM's narrow win are unreplicated recent preprints.
4. Genuine foundation-model informational gains in Brini survive **only at the
   monthly horizon** → a real lever may be targeting weekly/monthly RV, not 1-day.

## Mapping to Meridian experiments
| research item | Meridian action | status |
|---|---|---|
| CF-JEPA EMA routing (#2, JEPA) | dual online/target vol head | **done → +1.21%, p=0.003** |
| DS3M switching-SSM (#1, regime) | replace HMM regime layer | proposed |
| Monthly-horizon target (caveat 4) | add h=5/22 RV targets | proposed |
| EB-JEPA/H-JEPA/hyperbolic | — | de-prioritized (no evidence) |

Sources: arXiv 2607.05291 (Brini), 2106.02329 (DS3M), 1710.05741 (KVAE),
2606.07031 (CF-JEPA), 2602.03604 (EB-JEPA), 2008.12595 (DVAE review),
2007.11887 (D2FM), 2405.10093 (LaT-PFN), 2509.25449 (TS-JEPA).

---

# Part 2 — Generative latent predictors (deep research #2, 2026-07-30)

106-agent verified survey of generative-latent models (DVAEs, latent diffusion,
distributional heads, generative world models) for daily vol/regime/tail.

**Headline:** NOT ONE generative-latent method has a published OOS win over
HAR-RV/GARCH/HMM on daily realized vol. Add the generative branch **surgically**
(regime inference + distributional head), never as a reconstruction-based
world-model replacement for the SSM+JEPA core — every DVAE optimizes an ELBO whose
data term is a reconstruction likelihood, the exact high-variance-noise pathology
JEPA was built to avoid. Run generative objectives in the SSM's LATENT space, not
on raw returns.

**Tier 1 — build (real, transferable value):**
- **Small GMM distributional head** ("heads not backbones", verified high): on S&P
  monthly, CRPS-skill point −0.09% → Gaussian +1.18% → **GMM +3.59%**; head effect
  (3.7pp) dominates backbone spread. OOS tail gains: −8.9% 5%-VaR pinball at h=1,
  coverage 94.0% vs 92.5%, better crisis tail capture. It **complements** the QLIKE
  point core (a Model Confidence Set on squared error excludes no backbone-head — the
  head separates models on CRPS/pinball/coverage, NOT point QLIKE). **Use a SMALL,
  strongly-regularized GMM — full LSTM-MDN VaR heads are fragile (negative flag).**
- **DS3M switching state-space** for regimes (peer-reviewed, IJF 2025; repo
  Sherry-Xu/Deep-Switching-State-Space-Model): RNN + nonlinear switching SSM, discrete
  Markov regime latents, ELBO; beats GRU/SRNN/DSARF/SNLDS with longer, realistic
  regime durations. **Zero financial evidence** → the daily equity+FX DS3M-vs-HMM test
  is the single highest-value unrun experiment (the exact test Meridian's HMM claim
  failed).

**Tier 2 — promising, unproven on the right baselines (skip for now):** latent
diffusion — D3VAE (NeurIPS'22, low-sample design, but beats only generative baselines),
Diffolio (beats DCC-GARCH on portfolio Sharpe, not vol QLIKE), conditional IV-surface
diffusion (beats only VolGAN). None benchmarked vs HAR/GARCH/HMM on daily RV.

**Constraints:** (a) validate tails with **VaR backtests + fixed-level quantiles**, not
just CRPS — a proper scoring rule provably cannot separate extreme-tail (max-functional)
values. (b) The distributional head must not degrade the QLIKE point win.

Sources #2: arXiv 2106.02329 (DS3M), 2512.03298 (AgACI), 2008.12595 (DVAE review),
2210.xxxx/NeurIPS'22 (D3VAE), Diffolio (2025), EJS 2019 (tail impossibility),
"heads not backbones" (2025-26 preprint).

---

# Part 3 — Modular/interpretable architecture & alpha modules (deep research #3, 2026-07-30)

105-agent verified survey of modular-interpretable architectures and, crucially,
which specialist modules actually generate ALPHA.

**Top-tier confirmation of our honest finding:** vol-timing / vol-management on a
single signal does NOT generate reliable OOS net alpha — the ~1.1 Sharpe ceiling is
real (Cederburg et al. 2020 JFE; DeMiguel et al. 2024 JF; best conditional
multifactor vol-timing combiner ~1.06 net). **A better vol forecast is not alpha.**

**The only evidence-backed path to Sharpe ~1.5: BREADTH + diversification.** Carry
within one asset class earns 0.6–0.9, but diversified across
equities/bonds/FX/commodities/credit reaches ~1.5 *purely from diversification*
(Koijen-Moskowitz-Pedersen-Vrugt 2018). Many modestly-profitable ORTHOGONAL modules
combined is the only route — not a better single signal.

**Alpha modules, ranked by OOS evidence (net Sharpes realistically 0.4–0.9/module):**
- **Time-series momentum** (Moskowitz-Ooi-Pedersen 2012 JFE): strongest single-signal
  evidence; needs only daily prices. Deep momentum nets improve it but ONLY
  frictionlessly (cost cliff at 2–3bps) — again "better forecast ≠ tradeable alpha".
- **Cross-sectional factor/ranking** (Gu-Kelly-Xiu 2020 RFS): ~1.35 gross long-short
  decile; needs cross-sectional breadth.
- **Carry**: needs forward/futures curves + short rates; breadth is the lever.
- **Variance risk premium**: needs options/implied-vol (we lack it).

**Interpretable bridge:** Bernstein Online Aggregation (BOA) / Interpretable
Mixture-of-Experts (IME) / glass-box additive (EBM/GAM) — match black boxes on fit
with exact per-expert attribution; BOA's proven value is HALVING DRAWDOWN, not
headline Sharpe; weights shift transparently at regime breaks.

**What is NOT achievable from daily free OHLC alone (verified):** Sharpe 1.5 on a
single asset class or single signal. Requires (a) cross-asset breadth
(equities+bonds+FX+commodities), (b) forward/futures curves + short rates for carry,
(c) options/implied-vol for VRP, (d) cross-sectional breadth for ranking.

**We built and MEASURED this** (`scripts/modular_stack.py`): TS-momentum + vol-managed
+ regime-overlay bridged by BOA. Momentum on our 11 assets: Sharpe 0.48 (thin breadth,
as predicted); BOA correctly down-weighted it to 0.07; combined Sharpe 0.93. Confirms
empirically: the ceiling is DATA/BREADTH, not architecture.

Sources #3: Moskowitz-Ooi-Pedersen 2012 (JFE), Koijen et al. 2018 (JFE),
Gu-Kelly-Xiu 2020 (RFS), Cederburg et al. 2020 (JFE), DeMiguel et al. 2024 (JF),
BOA (arXiv 2111.15365), IME (arXiv 2206.02107), EBM (NBER w33320).

---

# Part 4 — EBM foundations & decision-focused learning (deep research #4, 2026-07-30)

104-agent verified survey, read from the foundations (LeCun 2006 energy-based learning;
Donti-Amos-Kolter; Elmachtoub-Grigas SPO+; Lim-Zohren-Roberts DMN).

**Confirms the empirical ceiling from theory + top-tier evidence:**
- Decision-focused / end-to-end economic-loss training produces better DECISIONS than
  predict-then-optimize when the forecaster is misspecified/data-limited — but the OOS
  **net-of-cost gain on daily multi-asset data is modest: ~0.1–0.3 Sharpe over tuned
  PtO**, not transformational. Real decision-focused ETF Sharpes cluster **< 1.0**
  (SPO+/RobustSPO 0.55–0.79 vs Markowitz 0.66); 1.33 only in a single un-deflated
  7-ETF backtest.
- **DMN's headline ~2.8 Sharpe is GROSS, on FUTURES, and collapses to NEGATIVE past a
  ~2–10bps cost cliff** — mirrors our own capped-vs-uncapped result (0.32 capped;
  0.62 uncapped blew up −100% DD).
- **Energy-collapse degeneracy (LeCun 2006 §2.2.1):** a single-term "push-down"
  economic/Sharpe loss with no contrastive term collapses to a flat energy — exactly
  why our flexible MLP overfit to −0.31. The fix is a contrastive/regularizing term +
  turnover control, or minimal capacity (our linear policy).
- **EBMs** give real architectural value (un-normalized economic energy over
  (state,action); energy-min inference IS the decision; native multimodality) but have
  **NO published financial net-of-cost OOS evidence** — only robotics (Implicit BC) and
  generic forecasting (ScoreGrad).
- **Optimizers:** on tiny low-SNR financial data, ensembling + strong regularization +
  (sometimes) SAM help generalization; exotic optimizers (Lion/Sophia/Shampoo) are
  largely irrelevant to the OOS result.

**Verdict (foundations-level, matching all 6 of our experiments):** NO method credibly
and reproducibly reaches ~1.5 net Sharpe on daily data without futures/options/intraday.
The decision-focused edge is real but ~0.1–0.3 Sharpe. The binding constraint is DATA
MODALITY, confirmed independently at the foundations.

**Best honest model design (from the research):** a decision-focused EBM — economic
energy E(state, action) + a CONTRASTIVE term (prevents collapse) + explicit turnover
penalty, minimal capacity, seed-ensembled, deflated-Sharpe (Bailey–López de Prado)
evaluated on purged walk-forward. Expected honest net Sharpe: ~0.5–0.9 on this data.

Sources #4: LeCun et al. 2006 (EBM tutorial), Donti-Amos-Kolter 2017, Elmachtoub-Grigas
2022 (SPO+), Lim-Zohren-Roberts 2019 (DMN), Zhang-Zohren-Roberts 2020, Bailey-López de
Prado 2014 (deflated Sharpe).

---

# Part 5 — Intraday frontier (REAL new experiment, 2026-07-30)

First test of higher-frequency real data: Yahoo 1-hour bars, ~3yr, 13 liquid ETFs.
`scripts/intraday_research.py`.

**Finding 1 — data modality reveals real new structure (thesis validated):** the
overnight vs intraday decomposition (impossible from daily bars) shows equities earn
their premium OVERNIGHT (Sharpe ~0.9–1.4) while intraday is flat (~0.0–0.4). A genuinely
different signal exists at higher frequency. This is real evidence that richer data
carries structure daily bars destroy.

**Finding 2 — but signal ≠ tradable alpha (cost test):** the overnight anomaly is NOT
tradable net of costs — market-neutral overnight-minus-intraday is +0.54 GROSS but
NEGATIVE at ≥1bp/trade (it flips twice daily); overnight-only is 1.1 gross → 0.60 (1bp)
→ 0.09 (2bp) → negative (3bp). Turnover destroys it (confirms Lachance 2021).

**Sharpened conclusion:** reaching 1.5 net needs a signal that is BOTH in richer data
AND low-turnover / cost-efficient. The overnight anomaly has the signal, fails on cost.
This is precisely why **futures carry/trend** is the evidence-backed path: same richer
data, but low-turnover and ~10× cheaper to trade, so the premium survives costs. Intraday
RV is also now buildable (proper realized variance) for a cleaner vol target — a real
upgrade for the vol module when a longer intraday history is available.
