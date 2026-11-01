# Meridian Enterprise World Model — a linked bank of interpretable specialist modules

> **Numbers policy (PREREGISTRATION.md Amendment 6).** The only validated numbers in this
> project are those produced by the audited benchmark harness and stored in
> `results/benchmark_vol.json` — currently the **volatility module only**. Earlier versions of
> this document quoted figures for every module (regime p-values, VaR/ES calibration, factor R²,
> propagation correlations, combiner and continual-learning QLIKE gains, portfolio Sharpes).
> Those were computed before the audit, with the same evaluation weaknesses it found (stacked-row
> significance tests, invalid FX price bars, handicapped baselines), and are **withdrawn** here.
> The experiments still exist in `scripts/` and their history is in `RESULTS.md`; each module
> needs saved predictions, date-level tests and the cleaned data before a number is quoted again.

**Interactive, any-entity, continually-current.** A user asks about *any* entity in natural
language ("Apple", "bitcoin", "the yen", "Reliance Industries", "Apple vs Tesla", "what if
the market drops 5%") and the engine resolves it to a market symbol (Yahoo global search —
equities, ETFs, FX, crypto, futures, indices worldwide), fetches its history **live** (always
current — no stale training window), runs the module bank, and **explains** the
result in plain language. Unknown/private entities trigger an honest "share data or a source"
request rather than a fabricated answer. Engine: `meridian/engine.py`; CLI: `scripts/ask.py`
(single-entity analysis, side-by-side comparison, and first-order "what-if" scenario
propagation through the connectedness/beta link). This is the user-facing surface of the
world model — in-depth analysis + simulation on demand, auditable number by number.


**Design law (observed in pre-audit experiments, not re-validated):** a single model that tries
to *generalize* across tasks degraded — collapsing the regime module into the vol backbone
regressed vol (v1); starving a decoupled module regressed regime (v2). Only **decoupled
modules, each with its own objective, LINKED by interpretable combiners** kept every win
(v3). So the world model is a *bank of specialists*, never one network.

## The two integration principles (the "link without degradation" guarantee)

1. **No shared trainable backbone.** Modules never share weights that one task's gradient
   can corrupt. They may share the *data* and a *read-only state*, never a *trainable*
   representation. → one module's training can never degrade another's.
2. **Interpretable links only.** Modules connect through objects a human can read and audit:
   - a **combiner** with inspectable weights (BOA / stacking) — shows which module the
     system trusts, and when;
   - the **connectedness graph** (Diebold-Yilmaz) — an inspectable edge-weighted network
     of who transmits/absorbs shocks;
   - **regime-conditioning** — a discrete, named state (Calm/Transition/Stress) that gates
     module behavior transparently.
   No opaque cross-attention; every link is a number you can point at and explain.

## The modules (each interpretable from foundations)

| Module | Foundation (why interpretable) | Output | Status |
|---|---|---|---|
| **Volatility** | SSM belief core, QLIKE-trained, HAR-anchored ("M0"); HAR-style multi-scale RV + leverage channel as baselines | log-variance forecast | ✅ **audited**: −7.5% QLIKE vs per-asset HAR on the 2019–2026 holdout, 11/11 assets (`results/benchmark_vol.md`) |
| **Regime** | interpretable-by-design vol-percentile state (production) | named Calm/Transition/Stress | not re-validated — no regime-existence claim is made; the vol-percentile partition is a transparent labeling convention, not a discovered latent state (`regime_hardened.py`, `regime_faithful.py`) |
| **Tail / risk** | regime-conditional ACI (VaR) + conditional EVT/GPD (Expected Shortfall) | conditional-coverage VaR + coherent ES | not re-validated (`conformal_var.py`, `evt_tail.py`) |
| **Factor-state** (sparse DFG, Mirowski-LeCun) | energy-based dynamic factor graph + L1 sparse loadings | named factors + systemic-surprise energy | not re-validated (`dfg_sparse.py`) |
| **Connectedness** | VAR + generalized FEVD (Diebold-Yilmaz) | shock source/sink graph, systemic % | descriptive; not re-validated |
| **Network propagation** | VAR + generalized IRF (Pesaran-Shin), order-invariant | multi-entity "what-if" shock propagation matrix | not re-validated as a forecast; narrate stress links as vol-amplified interdependence, not contagion (`network.py`, `network_scenario.py`, `forbes_rigobon.py`) |
| **Causal impulse-response** | Local Projections (Jorda), recursive ID, Newey-West | dynamic shock propagation over horizons | descriptive; identification stated, not manufactured (`local_projections.py`) |
| **Covariance / portfolio** | rolling sample / EWMA(RiskMetrics) / Ledoit-Wolf shrinkage; GMV + risk-parity | forecast covariance matrix → min-var weights, portfolio VaR | not re-validated (`covariance.py`) |
| **Interactive engine** | entity resolution + module bank + NL explanation; analyze/compare/scenario/world/portfolio | in-depth answer to any user query, live data | functional (not a forecasting claim) — `engine.py`, `ask.py` |
| **Conversational contract** | tool-call registry + provenance ledger; LLM routes/explains, modules own every number | trustworthy interactive answers | demo only, not a benchmark — `tools.py`, `conversational_demo.py` |
| **Combiner** | glass-box additive GAM (monotone shape functions) + Bernstein online aggregation | inspectable per-module weights + readable per-feature curves | not re-validated (`ebm_combiner.py`) |
| **Continual-learning wrapper** | online test-time adaptation + rehearsal; regime = memory index | BWT, retention, online vs retrain | not re-validated (`continual.py`, `online_vs_retrain.py`, `leakage_audit.py`) |

## Interpretability standard — the bar we hold ourselves to (not just a claim)

"Interpretable from foundations" is a testable claim, so we define the bar (ascending strength)
and audit every module against it honestly (Rudin 2019; Jain-Wallace 2019; SAE/causal-intervention
literature — verified deep-research pass w6sf0th0s):

1. **Faithfulness (minimum):** the output must *provably change when the explanation changes*.
   Attention weights, gate activations, and post-hoc SHAP/LIME are **decorative until proven faithful**
   — we do not present them as "why".
2. **Causal necessity (ablation):** removing/zeroing a module must move the forecast in the
   *pre-registered* direction — a component earns "interpretable" only when intervention confirms it.
3. **Causal sufficiency (clamp), bounded & OOS-tested:** forcing a state reproduces the target behavior,
   validated out-of-sample (our leakage-audit discipline).

| Module | Meets the bar? |
|---|---|
| HAR + leverage vol | ✅ coefficients on measured RV *are* the model; grounded in realized-vol theory (ABDL 2003) |
| Generalized-IRF propagation | ⚠️ interpretable by construction; its earlier OOS validation is not re-validated |
| Conditional-EVT tail | ✅ inspectable parametric tail; backtest (Kupiec/Christoffersen) is its causal test — to be re-run |
| Ledoit-Wolf covariance | ✅ closed-form shrinkage, no black box |
| Diebold-Yilmaz connectedness | ✅ faithful; one edge-ablation necessity test still to add |
| **Regime** | ⚠️ production ships the **interpretable-by-design vol-percentile regime** (faithful by construction, stable on every asset); no claim that hidden regimes exist |
| Conversational / provenance layer | ⚠️ hard rule: surface **only ablation-verified attributions**, never gate/attention weights |

**What we CAN claim** (financial-theory-bounded, once each module is re-validated): conditional
volatility levels & term structure; tail VaR/ES with backtested coverage; variance connectedness &
shock *propagation* (co-movement); covariance-based min-variance portfolios; which variance regime we
are in (as a pre-registered *convention*). **What we CANNOT claim:** return direction/level; any
latent coordinate "is" a named factor without a similarity-transform invariance test; a
"discovered" (vs imposed) crisis regime; HAR beyond the ~22-day cascade (Corsi's own
mis-extrapolation flag); any attribution not verified by ablation; personalized investment advice.
Today, only the volatility claim is backed by an audited number.

## Continual learning without catastrophic forgetting (the modular advantage)

Because modules are decoupled, forgetting is solved **per module, cheaply**, and drift is
absorbed **at the link, transparently** — a genuinely new-for-this-domain composition:
- **Within a module:** online/incremental update with a forgetting guard (EWC-style
  penalty anchoring parameters important to past regimes, and/or a bounded replay buffer
  of exemplar days per regime). Measured by **backward transfer** — performance on old
  regimes must not drop after learning new data (target ≈ 0 forgetting).
- **Across modules (drift):** the BOA combiner reweights live — when a regime shifts, the
  combiner shifts trust to the module handling it, without any module unlearning.
- **Regime memory:** the switching module's states act as an index — the system *remembers*
  distinct market regimes as named, persistent latents, not as an averaged blur.

This is the key thesis: **modularity makes "keep learning + never forget" tractable** —
a monolith must trade plasticity against stability in one weight set; a linked bank lets
each module stay stable while the combiner supplies plasticity. That separation is the
new concept worth proving.

## Evaluation (every capability measurable, nothing aspirational)
- Prediction: OOS QLIKE / MZ-R2 / date-level DM / MCS vs 15 models (done for 1-day volatility —
  `results/benchmark_vol.md`).
- Systemic/higher-order: impulse-response & connectedness checked against known crises
  (to be re-run under the audited harness).
- Continual learning: **backward transfer ≥ 0** (no forgetting) AND online forecast
  improvement vs periodic retraining — both on purged walk-forward (to be re-run).
- Interpretability: every link (combiner weights, graph edges, regime labels) is a
  human-readable artifact.

## Integration harness (`scripts/world_model.py`)
One interface runs every module on the latest data → a single coherent **world state**:
system-level (connectedness graph + DFG factor-state + systemic-surprise energy) and
asset-level (vol forecast + regime + VaR), emitted as `results/world_state.json`. The
harness only READS interpretable module outputs and links them — no shared backbone,
per the no-degradation law. All modules are integrated in first form; only the volatility
module is validated under the audited harness.

## Portfolio module — honest scope
The connectedness graph is an **analytical / risk-monitoring** tool, NOT an allocation
signal: in a pre-audit test a connectedness-aware tilt underperformed plain inverse-vol
(high-connectedness assets are also high-premium, and connectedness becomes universal in
crises). Any portfolio pillar is **risk management** (vol-targeting + regime de-risking + VaR
constraints), not connectedness-driven alpha. No Sharpe figure is claimed. `scripts/portfolio.py`.

## Honest boundary
Prediction/risk/systemic/continual-learning are measurable capabilities; volatility
prediction is the one measured under the audited harness so far. What is NOT claimed: that
better prediction equals trading alpha (vol-timing on a single signal does not reliably
generate net alpha — Cederburg et al. 2020; DeMiguel et al. 2024), or that any module
"understands" markets. The world model is an interpretable **measurement-and-forecast
system** — not an oracle. That honesty is itself a feature in a market that distrusts black boxes.
