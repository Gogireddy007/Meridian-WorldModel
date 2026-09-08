# Meridian-WM — a bridge of interpretable specialist modules for financial markets

**Design philosophy (fixed): modular + interpretable, not monolithic.** The model is a
**bridge of specialist modules**, each doing ONE interpretable job with an inspectable
output — not one generalized black box. This is both the user's stated preference and
what our ablations empirically show (different components own different jobs). Modules
are **decoupled** so none can degrade another (v1 violated this — the switching core as
shared vol backbone regressed vol; v2 decouples). New capabilities (alpha, carry) are
added as NEW bridged modules, never by making one module "generalize."

| module | one job | interpretable output | measured status |
|---|---|---|---|
| Vol module (SSM+QLIKE+EMA readout) | next-day RV | log-variance forecast | +6.8% vs HAR ✅ |
| Regime module (switching SSM) | market regime | sticky posterior α, dwell | beats HMM econ value ✅ |
| Tail module (Student-t head) | return VaR/tails | df, scale → VaR quantiles | calibrated exceedance ⚠ |
| Surprise (JEPA energy) | novelty signal | per-day energy z-score | descriptive |
| *Alpha module(s)* (roadmap) | risk-adjusted P&L | position/signal | NOT built — needs new data |
| Interpretable bridge | compose modules | per-module attribution | to design |

**Status: design proposal (living doc).** Every choice is traceable to a *measured*
result or a *verified* research finding. Components are ablated individually; nothing is
claimed "best" until benchmarked.

## Evidence the design is built on

| Finding | Source | Design consequence |
|---|---|---|
| SSM belief core + **direct QLIKE** training + seed-ensembling beats HAR-RV (+6.3%, generalizes) | our runs | Keep as the vol engine — do not replace it |
| JEPA latent-prediction loss + SIGReg add **~nothing** to vol (backbone≈full) | our ablation | Demote JEPA loss to a low-weight auxiliary; it is not the vol driver |
| JEPA **EMA target-encoder readout** gives +1.2% (CF-JEPA replicated) | our CF-JEPA run | Read heads off the EMA target encoder |
| Beating HAR on daily vol is near-frontier; gains survive mainly at **monthly** horizon | Brini 2026 (verified) | Expect vol ≈ current best; add weekly/monthly RV targets |
| Regime is the open value; **HMM fails on persistence**; DS3M's Markov-discrete + RNN-continuous gives persistent regimes | our regime result + verified research #1 | Add an explicit **sticky discrete switching latent** |
| Hierarchical / hyperbolic JEPA: **no surviving evidence** | verified research | Excluded |
| Generative/distributional latent head for tail/vol-of-vol + likelihood surprise | Track B (pending) | Add a distributional head; validate it earns its keep |

## Architecture (Meridian-WM)

```
 daily feature window  x_{t-L+1 .. t}
          │
   ┌──────┴───────┐        EMA copy (τ)        ┌──────────────┐
   │  SSM belief  │ ───────────────────────────▶│ target enc.  │  h_t^EMA (smoothed)
   │  core (online)│  h_t (continuous)           └──────┬───────┘
   └──────┬───────┘                                     │  ← heads read from EMA (CF-JEPA)
          │                                             │
   ┌──────┴────────────────┐                            │
   │ discrete regime s_t∈K │  sticky Markov transition  │
   │ (DS3M-style) gates the│  (min-dwell / entropy prior)│
   │ SSM step params       │  → PERSISTENT regimes       │
   └───────────────────────┘                            │
                                                         ▼
   Heads (regime-conditioned, read off h_t^EMA):
     • vol head        → log-variance      L_qlike   [proven engine]
     • distributional  → Student-t / mixture over logRV   L_nll / L_crps   [tail, vol-of-vol]
     • regime posterior→ p(s_t | x_{≤t})   L_reg + stickiness prior
   Auxiliary (belief/surprise only, low weight):
     • JEPA predictor + SIGReg  → energy = surprise score
```

### Objective
`L = L_qlike(vol) + λ_dist·L_nll(dist) + λ_reg·(regime NLL + stickiness) + λ_jepa·L_jepa + λ_sig·SIGReg`,
seed-ensembled. Start weights: λ_jepa, λ_sig small (they don't drive vol); λ_dist,
λ_reg tuned on an inner split, never on the pre-registered OOS.

### Why this is a genuine bridge (and plausibly novel)
It composes three separately-evidenced ideas that, to our knowledge, have **not** been
combined for financial vol/regime: (a) a **switching state-space** latent for persistent
regimes (DS3M), (b) a **JEPA EMA-target readout** for the cheap smoothing gain (CF-JEPA),
and (c) a **generative distributional head** for tails/vol-of-vol and a likelihood
surprise — all anchored on a **QLIKE-trained SSM** backbone that we proved is the vol
workhorse. It is a *switching-state JEPA world model*.

### Honest expectations (no overclaim)
- **Vol QLIKE:** likely ≈ current best (near HAR frontier). Main vol upside is the
  monthly-horizon target, not architecture.
- **Regime:** the real expected win — persistent, economically-useful regimes vs HMM.
- **Tail / distributional:** new capability (CRPS, VaR coverage) the point head lacks.
- **Risk:** added capacity can *hurt* the near-frontier vol number and overfit low-sample
  daily data → each component ablated; sticky prior + entropy floor guard regime collapse.

## Build order (highest evidence-value first)
1. **Switching regime latent** (DS3M-style) on the existing SSM core → fix the regime axis.
2. **Distributional head** (Student-t / mixture over logRV, CRPS) → tails + likelihood surprise.
3. **Monthly/weekly RV targets** → the one evidence-backed vol lever.
4. Integrate + ensemble → Meridian-WM; full ablation table.
