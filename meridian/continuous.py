"""Continuous-time belief cores — validating the Neural-ODE / continuous-flow idea
on the vol task before committing to it (per the modular, evidence-first discipline).

ODERNNEncoder: between observations the belief evolves along a learned ODE (Euler
integration); at each observation it is updated by a GRU cell (Rubanova et al. 2019,
"Latent ODEs for Irregularly-Sampled Time Series"). On IRREGULAR streams this is the
right tool; on REGULAR daily bars it reduces to a fancier RNN — so this is a fair,
honest test of whether continuous-time helps here at all.

Matches the Encoder interface: forward(x:(B,L,F)) -> (B,L,d_model).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ODERNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.inp = nn.Linear(cfg.n_features, d)
        self.f = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))   # ODE dynamics
        self.cell = nn.GRUCell(d, d)
        self.head = nn.LayerNorm(d)
        self.ode_steps = 2
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):                                    # (B,L,F)
        B, L, _ = x.shape
        u = self.inp(x)
        h = torch.zeros(B, u.shape[-1], device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            for _ in range(self.ode_steps):                 # continuous evolution (Euler)
                h = h + (1.0 / self.ode_steps) * torch.tanh(self.f(h))
            h = self.cell(u[:, t], h)                        # observation update
            outs.append(h)
        return self.head(self.drop(torch.stack(outs, 1)))
