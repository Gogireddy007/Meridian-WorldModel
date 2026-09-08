"""Meridian-WM — the bridged world model (see ARCHITECTURE.md).

Composes only components with MEASURED value in this project or verified research:
  * SwitchingSSM belief core (sticky K-mode) ......... persistent regimes  (research #1)
  * EMA target readout .............................. +1.2% CF-JEPA (measured)
  * point vol head, trained on QLIKE ................ the vol engine (measured win)
  * Student-t distributional head over log-RV ....... tails/CRPS (research Tier-1;
                                                       small regularized head, not MDN)
  * light JEPA predictor + SIGReg (aux) ............. keeps the EMA target meaningful

Everything reads off the EMA target encoder. Trained end-to-end; regimes = argmax of
the switching posterior alpha (persistent by construction, no post-hoc clustering).
Distributional head is stop-grad on the QLIKE scale so it CANNOT degrade the point
vol win (research: heads complement, not replace, the point core).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder, MeridianConfig, Predictor, sigreg_loss
from .switching import SwitchingSSM, SwitchConfig, switching_regularizer


@dataclass
class WMConfig:
    n_features: int = 11
    window: int = 32
    d_model: int = 64
    d_state: int = 64
    n_regimes: int = 3
    stick: float = 1.0
    dropout: float = 0.1
    ema: float = 0.99
    lambda_jepa: float = 0.3
    lambda_sig: float = 0.3
    lambda_dist: float = 0.5
    lambda_regvol: float = 0.5    # regime module's OWN QLIKE head (gives it a task signal)
    lambda_stick: float = 0.3     # weight on the switching regularizer
    w_balance: float = 1.0        # load-balance strength (anti regime-collapse)
    n_layers: int = 2             # for the (decoupled) plain vol core
    decouple: bool = True         # vol/tail read a plain SSM core; switching = regime only
    seed: int = 0


def _switch_cfg(cfg: WMConfig) -> SwitchConfig:
    return SwitchConfig(n_features=cfg.n_features, d_model=cfg.d_model,
                        d_state=cfg.d_state, n_regimes=cfg.n_regimes,
                        stick=cfg.stick, dropout=cfg.dropout, seed=cfg.seed)


class MeridianWM(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.core = SwitchingSSM(_switch_cfg(cfg))       # regime inference branch
        # Decoupled vol/tail engine: the PROVEN plain SSM core (+EMA readout), so the
        # switching regime branch cannot contaminate the near-frontier vol forecast.
        mcfg = MeridianConfig(n_features=cfg.n_features, d_model=cfg.d_model,
                              d_state=cfg.d_state, n_layers=cfg.n_layers,
                              dropout=cfg.dropout, seed=cfg.seed)
        self.vol_core = Encoder(mcfg)
        self.target_vol_core = Encoder(mcfg)
        self.target_vol_core.load_state_dict(self.vol_core.state_dict())
        for p in self.target_vol_core.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg.d_model)
        self.vol_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                                      nn.Linear(cfg.d_model, 1))
        # small, regularized Student-t head over next-day RETURNS: outputs
        # [log_df (tail heaviness), log_scale (return scale)]. Scale is predicted
        # directly (well-conditioned) and biased toward a typical daily return std.
        self.dist_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                                       nn.Linear(cfg.d_model, 2))
        self.dist_head[-1].bias.data = torch.tensor([0.0, -4.6])   # scale ≈ exp(-4.6)=0.01
        # the regime module's OWN vol head: reads the switching-core belief, trained on
        # QLIKE, so the switching core gets a task gradient → economically meaningful
        # regimes WITHOUT being the main vol module's backbone (stays decoupled).
        self.regime_vol_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                                             nn.Linear(cfg.d_model, 1))

    @torch.no_grad()
    def update_target(self):
        m = self.cfg.ema
        for tp, p in zip(self.target_vol_core.parameters(), self.vol_core.parameters()):
            tp.mul_(m).add_(p, alpha=1 - m)

    def forward(self, x_ctx, x_fut=None):
        belief_s, alpha = self.core(x_ctx)                # switching core belief + posterior
        regime_vol = self.regime_vol_head(belief_s[:, -1]).squeeze(-1)   # regime module's own head
        h_on = self.vol_core(x_ctx)[:, -1]                # online vol belief (for JEPA/SIGReg)
        with torch.no_grad():
            h_tgt = self.target_vol_core(x_ctx)[:, -1]    # EMA-smoothed vol belief (readout)
        vol = self.vol_head(h_tgt).squeeze(-1)           # log-variance forecast of RV
        ds = self.dist_head(h_tgt)                        # (B,2): [log_df, log_scale]
        df = 3.0 + F.softplus(ds[:, 0])                   # >3 → stable finite variance
        scale = torch.exp(ds[:, 1]).clamp(1e-4, 0.5)      # StudentT return scale
        # Reads the stop-grad EMA belief ⇒ tail head params are disjoint from the vol
        # head/encoder path, so the tail objective can never degrade the QLIKE win.
        out = {"h": h_on, "alpha": alpha, "regime": alpha[:, -1].argmax(-1),
               "vol": vol, "df": df, "scale": scale, "regime_vol": regime_vol}
        if x_fut is not None:                            # JEPA aux (keeps EMA meaningful)
            z_pred = self.predictor(h_on)
            with torch.no_grad():
                z_tgt = self.target_vol_core(x_fut)[:, -1]
            out["energy"] = (z_pred - z_tgt).pow(2).mean(-1)
        return out

    def loss(self, batch):
        cfg = self.cfg
        out = self.forward(batch["x_ctx"], batch["x_fut"])
        y = batch["y"]                                   # log RV_{t+1}
        f = out["vol"]
        rv = torch.exp(y)
        qlike = (rv * torch.exp(-f) + f - y - 1.0).mean()          # QLIKE point loss

        # Student-t distributional NLL over the next-day RETURN r_{t+1}. df and the
        # scale-correction are the only trainable parts here (sigma is detached), so
        # this head adds tail/VaR structure without ever degrading the QLIKE win.
        t = torch.distributions.StudentT(out["df"], loc=0.0, scale=out["scale"])
        dist_nll = -t.log_prob(batch["r_next"]).mean()

        fr = out["regime_vol"]                            # regime module's own QLIKE head
        regvol = (rv * torch.exp(-fr) + fr - y - 1.0).mean()
        jepa = out["energy"].mean() if "energy" in out else torch.zeros((), device=y.device)
        sig = sigreg_loss(out["h"])
        stick = switching_regularizer(out["alpha"], w_balance=cfg.w_balance)
        total = (qlike + cfg.lambda_dist * dist_nll + cfg.lambda_regvol * regvol
                 + cfg.lambda_stick * stick + cfg.lambda_jepa * jepa + cfg.lambda_sig * sig)
        return total, {"total": float(total), "qlike": float(qlike), "regvol": float(regvol),
                       "dist": float(dist_nll), "stick": float(stick),
                       "jepa": float(jepa), "sig": float(sig)}


if __name__ == "__main__":
    cfg = WMConfig()
    m = MeridianWM(cfg)
    xb = torch.randn(8, cfg.window, cfg.n_features)
    yb = torch.randn(8) - 10.0
    rb = torch.randn(8) * 0.01
    loss, logs = m.loss({"x_ctx": xb, "x_fut": xb, "y": yb, "r_next": rb})
    loss.backward()
    print("loss ok:", {k: round(v, 4) for k, v in logs.items()})
    out = m.forward(xb)
    print("regime dist:", torch.bincount(out["regime"], minlength=cfg.n_regimes).tolist())
