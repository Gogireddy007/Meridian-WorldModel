"""Meridian core: SSM belief state + JEPA prediction + SIGReg regularization.

Architecture
------------
  x_{t-L+1:t}  --(diagonal SSM belief core)-->  belief state h_t
       |                                              |
       |                                     +--------+--------+
       |                                     |                 |
   JEPA predictor g(h_t) -> z_hat_{t+1}   vol head        regime head
                                          log RV_{t+1}    K-state logits

Objectives (jointly trained):
  * supervised volatility: MSE( vol_head(h_t), log RV_{t+1} )
  * JEPA latent prediction: || g(h_t) - sg(target_enc(future)) ||^2   (energy / surprise)
  * SIGReg: push embeddings toward isotropic Gaussian (anti-collapse, LeJEPA-style)

The JEPA "energy" (latent prediction error) is the live **surprise score**.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MeridianConfig:
    n_features: int = 11
    window: int = 32
    d_model: int = 64
    d_state: int = 64
    n_layers: int = 2
    n_regimes: int = 3
    dropout: float = 0.1
    lambda_jepa: float = 1.0
    lambda_sig: float = 0.5
    ema: float = 0.99
    seed: int = 0
    loss_mode: str = "mse"     # "mse" (head=E[log RV]) or "qlike" (head=log variance)
    dual_vol: bool = False     # CF-JEPA test: 2nd vol head reads the EMA target encoder
    core_type: str = "ssm"     # "ssm" (DiagonalSSM) or "ode" (Neural-ODE / ODE-RNN)
    aux_horizons: tuple = ()   # e.g. ("y_5", "y_22"): auxiliary multi-horizon vol heads
    lambda_aux: float = 0.3    # weight of each auxiliary-horizon loss in the total
    har_anchor: bool = False   # add an explicit linear HAR skip pathway (HARNet-style)
    har_idx: tuple = (0, 1, 2) # feature indices of (har_d, har_w, har_m) in the input window
    offset_mode: bool = False  # vol = external per-row offset (e.g. per-asset HAR) + NN correction
    n_assets: int = 0          # >0 with asset_emb_dim>0: learned per-asset embedding
    asset_emb_dim: int = 0     # embedding size added (via a zero-init projection) to the belief


# --------------------------------------------------------------------------- #
class DiagonalSSM(nn.Module):
    """Diagonal linear state-space layer (S4/Mamba-lite).

    h_t = a ⊙ h_{t-1} + b ⊙ x_t ;  y_t = C h_t + D x_t, with a = sigmoid(logit)
    giving stable decay in (0,1). Runs as a sequential scan (windows are short).
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.in_proj = nn.Linear(d_model, d_state)
        self.a_logit = nn.Parameter(torch.linspace(1.0, 4.0, d_state))  # decay ~sigmoid
        self.b = nn.Parameter(torch.ones(d_state) * 0.5)
        self.out_proj = nn.Linear(d_state, d_model)
        self.skip = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: (B, L, d_model)
        B, L, _ = x.shape
        a = torch.sigmoid(self.a_logit)               # (d_state,)
        u = self.in_proj(x) * self.b                   # (B, L, d_state)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = a * h + u[:, t]
            outs.append(h)
        h_seq = torch.stack(outs, dim=1)               # (B, L, d_state)
        return self.out_proj(h_seq) + self.skip(x)


class Encoder(nn.Module):
    """Feature projection + stacked SSM belief core. Returns full belief sequence."""

    def __init__(self, cfg: MeridianConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.n_features, cfg.d_model)
        self.layers = nn.ModuleList(
            [DiagonalSSM(cfg.d_model, cfg.d_state) for _ in range(cfg.n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)])
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.LayerNorm(cfg.d_model)

    def forward(self, x):                              # (B, L, F) -> (B, L, d_model)
        h = self.inp(x)
        for ssm, norm in zip(self.layers, self.norms):
            h = h + self.drop(ssm(norm(h)))
        return self.head(h)


class Predictor(nn.Module):
    """JEPA predictor: belief state -> predicted next-window embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, h):
        return self.net(h)


# --------------------------------------------------------------------------- #
def sigreg_loss(z: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (LeJEPA-style, practical form).

    Pushes the embedding distribution toward an isotropic Gaussian by matching,
    along `n_proj` random 1-D projections, the first four standardized moments
    to those of N(0,1). Prevents representation collapse without stop-grad.
    """
    z = z - z.mean(0, keepdim=True)
    d = z.shape[1]
    P = torch.randn(d, n_proj, device=z.device, dtype=z.dtype)
    P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
    p = z @ P                                          # (N, n_proj)
    p = p / (p.std(0, keepdim=True) + 1e-6)
    m2 = (p ** 2).mean(0)                              # ->1
    m3 = (p ** 3).mean(0)                              # ->0
    m4 = (p ** 4).mean(0)                              # ->3
    return ((m2 - 1) ** 2 + m3 ** 2 + (m4 - 3) ** 2).mean()


class Meridian(nn.Module):
    def __init__(self, cfg: MeridianConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        _Enc = Encoder
        if cfg.core_type == "ode":
            from .continuous import ODERNNEncoder as _Enc
        self.encoder = _Enc(cfg)
        self.target_encoder = _Enc(cfg)                  # EMA target (no grad)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg.d_model)
        self.vol_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        # CF-JEPA test: a second vol head that reads the EMA target encoder.
        # It is a probe on the smoothed representation (stop-grad to the encoder),
        # trained on the same QLIKE loss so we can compare the two read-offs.
        self.vol_head_tgt = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        ) if cfg.dual_vol else None
        self.regime_head = nn.Linear(cfg.d_model, cfg.n_regimes)
        # Auxiliary multi-horizon vol heads (e.g. h=5, h=22 mean log-RV): cheap
        # extra readouts off the same shared belief state h_t, trained with a
        # low-weight auxiliary loss so they cannot dominate the primary h=1 head.
        self.aux_heads = nn.ModuleDict({
            h: nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                              nn.Linear(cfg.d_model, 1))
            for h in cfg.aux_horizons
        })
        # HARNet-style explicit HAR skip pathway (arXiv:2205.07719): a direct linear
        # readout of the (har_d, har_w, har_m) features at t, added to every vol head's
        # output. Gives gradient descent a short, easy path to at least HAR-level
        # forecasts from the first few steps, rather than having to route everything
        # through the SSM recurrence + MLP before it can match the linear baseline.
        if cfg.har_anchor:
            self.har_skip = nn.Linear(len(cfg.har_idx), 1)
            self.har_skip_aux = nn.ModuleDict({
                h: nn.Linear(len(cfg.har_idx), 1) for h in cfg.aux_horizons
            })
        # Offset mode: the forecast is offset + NN correction, where the offset (e.g. a
        # per-asset HAR forecast fit on the fold's train data) is passed in per row. The
        # correction heads start at exactly 0, so before training the model IS the
        # offset model and gradient descent only ever adds corrections on top of it.
        if cfg.offset_mode:
            for head in (self.vol_head, *self.aux_heads.values()):
                nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        # Asset embedding (index n_assets = "unseen asset": a fixed zero vector), added
        # to the belief state through a zero-initialized projection.
        if cfg.asset_emb_dim > 0 and cfg.n_assets > 0:
            self.asset_emb = nn.Embedding(cfg.n_assets + 1, cfg.asset_emb_dim, padding_idx=cfg.n_assets)
            self.asset_proj = nn.Linear(cfg.asset_emb_dim, cfg.d_model)
            nn.init.zeros_(self.asset_proj.weight); nn.init.zeros_(self.asset_proj.bias)
        else:
            self.asset_emb = None

    @torch.no_grad()
    def init_har_anchor(self, beta, mean_har, std_har, aux_beta=None):
        """HARNet-style initialization (Reisenhofer, Bayer & Hautsch 2022,
        arXiv:2205.07719): set weights so that, BEFORE any optimization step, the
        model's output is identical to the closed-form HAR-OLS forecast for this
        fold. `beta` = [intercept, beta_d, beta_w, beta_m] fit on RAW (log) HAR
        features; `mean_har`/`std_har` are the neural window scaler's stats for
        the har_d/w/m feature slots, used to fold the standardization into the
        skip layer's weights. The rest of the network (SSM + vol_head MLP) is
        zeroed so its initial contribution is exactly 0 — training then only ever
        has to find CORRECTIONS on top of HAR, never has to rediscover it from
        scratch, which is what makes QLIKE optimization stable for this class of
        model (per the HARNet paper).
        """
        beta = np.asarray(beta, dtype=np.float32)
        mean_har = np.asarray(mean_har, dtype=np.float32)
        std_har = np.asarray(std_har, dtype=np.float32)
        w = beta[1:] * std_har
        b = beta[0] + float(np.dot(beta[1:], mean_har))
        self.har_skip.weight.copy_(torch.tensor(w).view(1, -1))
        self.har_skip.bias.copy_(torch.tensor([b]))
        for lin in (self.vol_head[-1],):
            lin.weight.zero_(); lin.bias.zero_()
        for h, head in self.aux_heads.items():
            head[-1].weight.zero_(); head[-1].bias.zero_()
            skip = self.har_skip_aux[h]
            if aux_beta is not None and h in aux_beta:
                ab = np.asarray(aux_beta[h], dtype=np.float32)
                skip.weight.copy_(torch.tensor(ab[1:] * std_har).view(1, -1))
                skip.bias.copy_(torch.tensor([ab[0] + float(np.dot(ab[1:], mean_har))]))
            else:
                skip.weight.copy_(torch.tensor(w).view(1, -1))
                skip.bias.copy_(torch.tensor([b]))

    @torch.no_grad()
    def update_target(self):
        m = self.cfg.ema
        for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.mul_(m).add_(p, alpha=1 - m)

    def belief(self, x):
        """Return last belief state h_t for a window x: (B, L, F) -> (B, d_model)."""
        return self.encoder(x)[:, -1]

    def forward(self, x_ctx, x_fut=None, offset=None, aux_offset=None, asset_id=None):
        """x_ctx: (B, L, F) context window ending at t.
        x_fut:  (B, L, F) window ending at t+1 (for JEPA target); optional.
        offset / aux_offset: (B,) / {h: (B,)} per-row log-variance offsets (offset_mode).
        asset_id: (B,) long, used when an asset embedding is configured.
        """
        h_seq = self.encoder(x_ctx)
        h_t = h_seq[:, -1]                               # belief state
        if self.asset_emb is not None and asset_id is not None:
            h_t = h_t + self.asset_proj(self.asset_emb(asset_id))
        vol = self.vol_head(h_t).squeeze(-1)             # predicted log RV_{t+1}
        if offset is not None:
            vol = vol + offset
        regime_logits = self.regime_head(h_t)

        if self.cfg.har_anchor:
            har_x = x_ctx[:, -1, list(self.cfg.har_idx)]
            vol = vol + self.har_skip(har_x).squeeze(-1)

        out = {"h": h_t, "vol": vol, "regime_logits": regime_logits,
               "h_seq": h_seq}

        if self.cfg.aux_horizons:
            aux = {h: self.aux_heads[h](h_t).squeeze(-1) for h in self.cfg.aux_horizons}
            if self.cfg.har_anchor:
                aux = {h: v + self.har_skip_aux[h](har_x).squeeze(-1) for h, v in aux.items()}
            if aux_offset is not None:
                aux = {h: v + aux_offset[h] if h in aux_offset else v for h, v in aux.items()}
            out["aux"] = aux

        if self.cfg.dual_vol:
            # read a vol forecast off the EMA target encoder (stop-grad to encoder)
            with torch.no_grad():
                h_tgt = self.target_encoder(x_ctx)[:, -1]
            out["vol_tgt"] = self.vol_head_tgt(h_tgt).squeeze(-1)

        if x_fut is not None:
            z_pred = self.predictor(h_t)                 # predicted future embedding
            with torch.no_grad():
                z_tgt = self.target_encoder(x_fut)[:, -1]
            energy = (z_pred - z_tgt).pow(2).mean(-1)    # per-sample surprise
            out.update({"z_pred": z_pred, "z_tgt": z_tgt, "energy": energy})
        return out

    def loss(self, batch):
        cfg = self.cfg
        aux_off = {h: batch[f"off_{h}"] for h in cfg.aux_horizons if f"off_{h}" in batch} or None
        out = self.forward(batch["x_ctx"], batch["x_fut"], offset=batch.get("off_y"),
                           aux_offset=aux_off, asset_id=batch.get("asset_id"))
        y = batch["y"]                       # log RV_{t+1}

        def vol_objective(f):
            if cfg.loss_mode == "qlike":
                # f is the log-variance forecast; minimize exact QLIKE.
                # QLIKE = RV/s2 - log(RV/s2) - 1 = RV*exp(-f) + f - log RV - 1
                rv = torch.exp(y)
                return (rv * torch.exp(-f) + f - y - 1.0).mean()
            return F.mse_loss(f, y)          # f is E[log RV]

        vol_loss = vol_objective(out["vol"])
        jepa_loss = out["energy"].mean()
        sig = sigreg_loss(out["h"])
        total = vol_loss + cfg.lambda_jepa * jepa_loss + cfg.lambda_sig * sig
        logs = {"total": float(total), "vol": float(vol_loss),
                "jepa": float(jepa_loss), "sig": float(sig)}
        if cfg.dual_vol:
            vol_loss_tgt = vol_objective(out["vol_tgt"])
            total = total + vol_loss_tgt     # trains the target-readout head only
            logs["vol_tgt"] = float(vol_loss_tgt)
        for h in cfg.aux_horizons:
            if h not in batch:
                continue
            yh, fh = batch[h], out["aux"][h]
            if cfg.loss_mode == "qlike":
                aux_loss = (torch.exp(yh) * torch.exp(-fh) + fh - yh - 1.0).mean()
            else:
                aux_loss = F.mse_loss(fh, yh)
            total = total + cfg.lambda_aux * aux_loss
            logs[f"aux_{h}"] = float(aux_loss)
        return total, logs
