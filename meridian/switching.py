"""Switching-state belief core (Meridian-WM build #1) — a soft, sticky
mixture-of-K-SSMs, the differentiable Kalman-VAE-style cousin of DS3M.

Mechanism (research #1): K parallel diagonal-SSM dynamic modes; a recurrent gate
produces a regime posterior alpha_t over modes with an explicit STICKINESS term
(alpha_t leans on alpha_{t-1}), which yields PERSISTENT regimes — the fix for the
failed HMM-persistence claim. The belief step is the alpha-weighted mixture of the
K linear dynamics; the regime label is argmax_k alpha_t.

Trained end-to-end (no discrete inference / Viterbi). Regimes emerge from the vol/
QLIKE objective plus a stickiness prior, not from post-hoc clustering.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SwitchConfig:
    n_features: int = 11
    d_model: int = 64
    d_state: int = 64
    n_regimes: int = 3
    stick: float = 2.0        # stickiness: logit bonus toward previous regime
    dropout: float = 0.1
    seed: int = 0


class SwitchingSSM(nn.Module):
    """Sticky soft mixture of K diagonal-SSM modes.

    Returns the belief sequence and the per-step regime posterior alpha (B,L,K).
    """

    def __init__(self, cfg: SwitchConfig):
        super().__init__()
        K, dst, dm = cfg.n_regimes, cfg.d_state, cfg.d_model
        self.cfg = cfg
        self.inp = nn.Linear(cfg.n_features, dm)
        # K dynamic modes: decay a_k in (0,1) and input gain b_k
        self.a_logit = nn.Parameter(torch.stack([
            torch.linspace(0.5 + 0.5 * k, 3.5 + 0.5 * k, dst) for k in range(K)]))  # (K,dst)
        self.b = nn.Parameter(torch.ones(K, dst) * 0.5)
        self.in_proj = nn.Linear(dm, dst)
        self.out_proj = nn.Linear(dst, dm)
        self.gate = nn.Sequential(                       # regime gate from (x_t, h_{t-1})
            nn.Linear(dm + dst, dm), nn.GELU(), nn.Linear(dm, K))
        self.norm = nn.LayerNorm(dm)

    def forward(self, x):                                # x: (B,L,F)
        B, L, _ = x.shape
        K, dst = self.cfg.n_regimes, self.cfg.d_state
        u = self.in_proj(self.inp(x))                    # (B,L,dst)
        xin = self.inp(x)                                # (B,L,dm) for gate
        a = torch.sigmoid(self.a_logit)                  # (K,dst)
        h = torch.zeros(B, dst, device=x.device, dtype=x.dtype)
        prev_alpha = torch.full((B, K), 1.0 / K, device=x.device, dtype=x.dtype)
        beliefs, alphas = [], []
        for t in range(L):
            g = self.gate(torch.cat([xin[:, t], h], -1))          # (B,K)
            g = g + self.cfg.stick * prev_alpha                    # stickiness
            alpha = F.softmax(g, -1)                               # (B,K) regime posterior
            # mixture of K linear dynamics: h_t = sum_k alpha_k (a_k h + b_k u_t)
            hk = a.unsqueeze(0) * h.unsqueeze(1) + self.b.unsqueeze(0) * u[:, t].unsqueeze(1)  # (B,K,dst)
            h = (alpha.unsqueeze(-1) * hk).sum(1)                  # (B,dst)
            beliefs.append(self.out_proj(h))
            alphas.append(alpha)
            prev_alpha = alpha
        belief = self.norm(torch.stack(beliefs, 1))               # (B,L,dm)
        alpha = torch.stack(alphas, 1)                            # (B,L,K)
        return belief, alpha


def switching_regularizer(alpha: torch.Tensor, w_switch: float = 1.0,
                          w_balance: float = 1.0) -> torch.Tensor:
    """Persistent BUT non-degenerate regimes. alpha: (B,L,K).

    * persistence: penalize per-step regime-posterior change (switch rarely).
    * load balance: encourage the MARGINAL regime usage across the whole batch/time
      to spread over all K (maximize its entropy) — this is what prevents the
      collapse-to-one-regime failure that per-step entropy does not.
    Both terms are minimized: small switching + small negative-marginal-entropy.
    """
    switch = (alpha[:, 1:] - alpha[:, :-1]).abs().sum(-1).mean()
    marg = alpha.mean(dim=(0, 1)).clamp_min(1e-8)                  # (K,) marginal usage
    neg_marg_entropy = (marg * marg.log()).sum()                  # = -H(marg); want small
    return w_switch * switch + w_balance * neg_marg_entropy


if __name__ == "__main__":
    cfg = SwitchConfig()
    m = SwitchingSSM(cfg)
    x = torch.randn(4, 32, cfg.n_features)
    b, al = m(x)
    print("belief", b.shape, "alpha", al.shape, "alpha sums to 1:",
          torch.allclose(al.sum(-1), torch.ones(4, 32)))
    print("switching regularizer:", float(switching_regularizer(al).detach()))
