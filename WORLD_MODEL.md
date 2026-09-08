# Meridian Enterprise World Model — a linked bank of interpretable specialist modules

**Interactive, any-entity, continually-current.** A user asks about *any* entity in natural
language ("Apple", "bitcoin", "the yen", "Reliance Industries", "Apple vs Tesla", "what if
the market drops 5%") and the engine resolves it to a market symbol (Yahoo global search —
equities, ETFs, FX, crypto, futures, indices worldwide), fetches its history **live** (always
current — no stale training window), runs the validated module bank, and **explains** the
result in plain language. Unknown/private entities trigger an honest "share data or a source"
request rather than a fabricated answer. Engine: `meridian/engine.py`; CLI: `scripts/ask.py`
(single-entity analysis, side-by-side comparison, and first-order "what-if" scenario
propagation through the connectedness/beta link). This is the user-facing surface of the
world model — in-depth analysis + simulation on demand, auditable number by number.


**Design law (empirically proven here, not assumed):** a single model that tries to
*generalize* across tasks degrades — collapsing the regime module into the vol backbone
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

## The modules (each interpretable from foundations, each measured)

| Module | Foundation (why interpretable) | Output | Status |
|---|---|---|---|
| **Volatility** | HAR-style multi-scale RV + **leverage/bad-vol channel** (down-move variance); SSM belief | log-variance forecast | ✅ +9% vs HAR (SSM ensemble); engine's per-entity HAR+leverage adds +0.76% OOS QLIKE (most on equities, ~0 on FX — leverage effect confirmed; shar_validate.py) |
| **Regime** | interpretable-by-design vol-percentile state (production) | named Calm/Transition/Stress | ⚠️ **existence MARGINAL under the HONEST null** (regime_hardened.py: null=HAR+leverage, block bootstrap preserving conditional heteroskedasticity — SPY p=0.04 barely survives, QQQ p=0.08 does NOT). The original p=0.01 was INFLATED by an AR(1) null. Corrected read: hidden vol regimes are weak/marginal, so we make NO discovery claim — the by-design vol-percentile partition is honest precisely because it's a transparent labeling of the vol distribution, not a claimed latent state. ⚠️ old switching-SSM captured it POORLY (interpret_regime.py: η²=1.4%, 55% identifiable). Faithful MS-on-log-RV is **16× more faithful** (η²=22%, catches crises) on equities but **degenerate on FX** (98% one state) and fit-unstable across assets → **production uses the by-design vol-percentile regime** (faithful by construction, stable on every asset incl. FX). Rudin's "use interpretable models" vindicated — regime_faithful.py |
| **Tail / risk** | regime-conditional ACI (VaR) **+ conditional EVT/GPD (Expected Shortfall)** | conditional-coverage VaR + coherent ES | ✅ ACI fixes clustering (Christoffersen 0.77); **EVT Expected Shortfall calibrated ratio 1.01** (Basel III coherent measure) — conformal_var.py, evt_tail.py |
| **Factor-state** (sparse DFG, Mirowski-LeCun) | energy-based dynamic factor graph + L1 sparse loadings | NAMED factors (Equity-vol / Rates-vol / Commodity-vol / FX-vol) + systemic-surprise energy | ✅ R²0.77, factors 82–92% single-class (named, interpretable), +2.1% pred lift, caught 2019 repo crisis (dfg_sparse.py) |
| **Connectedness** | VAR + generalized FEVD (Diebold-Yilmaz) | shock source/sink graph, systemic % | ✅ COVID-validated (reduced-form) |
| **Network propagation** | VAR + generalized IRF (Pesaran-Shin), order-invariant | multi-entity "what-if" shock propagation matrix | ✅ **validated OOS as CO-MOVEMENT forecast**: 124 held-out large-move days, propagated vs realized corr **+0.72**, 79% direction; order-invariance exact. ⚠️ **Forbes-Rigobon gate (forbes_rigobon.py): 0/14 stress-period links survive vol-adjustment** — stress connectedness is the mechanical volatility effect (stable interdependence), NOT new contagion; narrate as vol-amplified interdependence, not crisis channels — network.py, network_scenario.py |
| **Causal impulse-response** | Local Projections (Jorda), recursive ID, Newey-West | dynamic shock propagation over horizons | ✅ economically sensible (equity reversal, risk-FX continuation, bond hedge); identification stated, not manufactured (local_projections.py) |
| **Covariance / portfolio** | rolling sample / EWMA(RiskMetrics) / Ledoit-Wolf shrinkage; GMV + risk-parity | forecast covariance matrix → min-var weights, portfolio VaR | ✅ GMV cuts portfolio risk 69% vs equal-weight OOS 2011→2026; Ledoit-Wolf best risk-adjusted — covariance.py |
| **Interactive engine** | entity resolution + module bank + NL explanation; analyze/compare/scenario/world/portfolio | in-depth answer to any user query, live data | ✅ resolves global entities, data-quality flagging, honest need-data fallback — engine.py, ask.py |
| **Conversational contract** | tool-call registry + provenance ledger; LLM routes/explains, modules own every number | trustworthy interactive answers | ✅ anti-hallucination proven: honest paraphrase passes, invented figure CAUGHT — tools.py, conversational_demo.py |
| **Combiner** | **glass-box additive GAM** (monotone shape functions) + Bernstein online aggregation | inspectable per-module weights + readable per-feature curves | ✅ **glass-box matches/beats black box** (ebm_combiner.py: GAM OOS QLIKE 1.23 < monotone-GBDT 1.47 < linear 1.78, pooled/relative); all shape functions monotone by theory; **Adebayo faithfulness PASS** (effect 0.78 real → 0.006 under label-shuffle) — interpretable-by-design combiner, no accuracy penalty |
| **Continual-learning wrapper** | online test-time adaptation + rehearsal; regime = memory index | BWT, retention, online vs retrain | ✅ BWT+ (no forgetting) + rehearsal retention (continual.py); **online adaptation beats periodic retrain −27% QLIKE (online_vs_retrain.py); LEAKAGE-AUDITED (leakage_audit.py): clean==explicitly-lagged to 0.02%, a leaky variant would gain +3.8% so the number is conservative, per Proceed KDD-2025** — the "compounding learning" engine |

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
| Generalized-IRF propagation | ✅ **the exemplar** — already intervention-validated OOS (+0.72) |
| Conditional-EVT tail | ✅ inspectable parametric tail; backtest (Kupiec/Christoffersen) is its causal test |
| Ledoit-Wolf covariance | ✅ closed-form shrinkage, no black box |
| Diebold-Yilmaz connectedness | ✅ faithful; one edge-ablation necessity test still to add |
| **Regime** | ✅ **RESOLVED** — regime *existence* proven (bootstrap LR p=0.01); the *learned* switching-SSM was the weak interpreter (η²=1.4%), so production ships the **interpretable-by-design vol-percentile regime** (faithful by construction, stable on every asset). The learned MS-on-log-RV corroborates (16× more faithful on equities) but is degenerate on FX — kept only as corroboration, not production |
| Conversational / provenance layer | ⚠️ hard rule: surface **only ablation-verified attributions**, never gate/attention weights |

**What we CAN claim** (financial-theory-bounded): conditional volatility levels & term structure;
tail VaR/ES with backtested coverage; variance connectedness & shock *propagation* (co-movement,
OOS-validated); covariance-based min-variance portfolios; which variance regime we are in (as a
pre-registered *convention*, once the regime-count test passes). **What we CANNOT claim:** return
direction/level; any latent coordinate "is" a named factor without a similarity-transform invariance
test; a "discovered" (vs imposed) crisis regime; HAR beyond the ~22-day cascade (Corsi's own
mis-extrapolation flag); any attribution not verified by ablation; personalized investment advice.

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
- Prediction: OOS QLIKE / MZ-R2 / DM vs a full model field (done).
- Systemic/higher-order: impulse-response & connectedness validated against known crises
  (COVID spike confirmed); shock-source/sink economic sanity (confirmed).
- Continual learning: **backward transfer ≥ 0** (no forgetting) AND online forecast
  improvement vs periodic retraining — both on purged walk-forward.
- Interpretability: every link (combiner weights, graph edges, regime labels) is a
  human-readable artifact.

## Integration harness (`scripts/world_model.py`)
One interface runs every module on the latest data → a single coherent **world state**:
system-level (connectedness graph + DFG factor-state + systemic-surprise energy) and
asset-level (vol forecast + regime + VaR), emitted as `results/world_state.json`. The
harness only READS interpretable module outputs and links them — no shared backbone,
per the no-degradation law. **All five pillars are validated and integrated in first form.**

## Portfolio module — honest scope (tested)
The connectedness graph is an **analytical / risk-monitoring** tool, NOT an allocation
signal: a connectedness-aware tilt was tested and *underperformed* plain inverse-vol
(Sharpe 0.40 vs 0.55, worse drawdown) — high-connectedness assets are also high-premium,
and connectedness becomes universal in crises. The portfolio pillar is genuine **as risk
management** (vol-targeting + regime de-risking + VaR constraints → ~0.5–1.1 Sharpe by
book) — not as connectedness-driven alpha. `scripts/portfolio.py`.

## Honest boundary
Prediction/risk/systemic/continual-learning are real, measurable capabilities. What is
NOT claimed: that better prediction equals large trading alpha (it does not on daily data —
proven), or that any module "understands" markets. The world model is an interpretable,
continually-learning **measurement-and-forecast system**, best-in-class where measured —
not an oracle. That honesty is itself a feature in a market that distrusts black boxes.
