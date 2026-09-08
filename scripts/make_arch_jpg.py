"""Render the Meridian architecture as a white-background raster image (PNG → JPG).
Self-contained (matplotlib only); deterministic layout."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "results"
INK = "#12203a"; MUT = "#5b6678"; TEAL = "#0d9488"; GOOD = "#1f8a5b"; WARN = "#b45309"
BANDBG = "#f3f6fb"; BANDBD = "#dde4ee"; LINK = "#0d9488"; LINKBG = "#e7f6f3"; ARROW = "#7d8aa5"

fig, ax = plt.subplots(figsize=(13, 11), dpi=200)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, sub=None, tag=None, fc="white", ec=BANDBD, tc=INK, bold=True, ts=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y + (h * 0.16 if (sub or tag) else 0), title, ha="center", va="center",
            fontsize=ts, color=tc, fontweight="bold" if bold else "normal", family="monospace")
    if sub:
        ax.text(x, y - h * 0.14, sub, ha="center", va="center", fontsize=ts - 2.2, color=tc, family="monospace")
    if tag:
        ax.text(x, y - h * 0.34, tag, ha="center", va="center", fontsize=ts - 2.4,
                color=GOOD if not tag.startswith("△") else WARN, fontweight="bold", family="monospace")


def band(y, h, label):
    ax.add_patch(FancyBboxPatch((3, y - h / 2), 94, h, boxstyle="round,pad=0.2,rounding_size=1.5",
                                fc=BANDBG, ec=BANDBD, lw=1))
    ax.text(4.6, y + h / 2 - 1.6, label, ha="left", va="center", fontsize=8.5,
            color=MUT, fontweight="bold", family="monospace")


def arrow(y0, y1, x=50):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                                 color=ARROW, lw=1.6, shrinkA=0, shrinkB=0))


ax.text(50, 98, "Meridian Enterprise World Model — Architecture", ha="center", fontsize=17,
        fontweight="bold", color=INK)
ax.text(50, 95, "a bank of decoupled, individually-validated specialist modules · linked by interpretable objects, never one black box",
        ha="center", fontsize=9, color=MUT)

# USER query
box(50, 90.5, 46, 4.6, "USER  ·  any question · any entity", fc=TEAL, ec=TEAL, tc="white", ts=11)
arrow(88.2, 86.2)

# 1 DATA
band(82.5, 7.5, "1 · DATA — any entity on a free feed")
box(37, 82.3, 24, 4.4, "Yahoo (global)", "equities·ETF·FX·crypto·futures·index", ts=9)
box(64, 82.3, 22, 4.4, "FRED (macro)", "VIX · yields", ts=9)
arrow(78.6, 74.8)

# 2 MEASUREMENT
band(70.5, 8, "2 · MEASUREMENT — causal, no lookahead")
box(27, 70.2, 21, 4.4, "Realized variance", "Garman-Klass", ts=9)
box(50, 70.2, 21, 4.4, "HAR cascade", "d/w/m + leverage", ts=9)
box(73, 70.2, 21, 4.4, "Returns panel", "cross-asset", ts=9)
arrow(66.3, 60.8)

# 3 MODULE BANK
band(53.5, 15.5, "3 · MODULE BANK — decoupled specialists, each validated out-of-sample")
mods = [
    ("Volatility", "HAR+leverage", "+6.3% vs HAR"),
    ("Regime", "vol-percentile", "△ marginal"),
    ("Tail · EVT", "VaR + ES", "ES calib ~1.0"),
    ("Covariance", "Ledoit-Wolf GMV", "-69% risk"),
    ("Connected.", "Diebold-Yilmaz", "COVID-valid"),
    ("Network prop.", "generalized IRF", "+0.72 OOS"),
    ("Continual", "online learning", "-27% retrain"),
]
xs = [9.5, 22.4, 35.3, 48.2, 61.1, 74.0, 86.9]
for (t, s, g), x in zip(mods, xs):
    ec = WARN if g.startswith("△") else GOOD
    box(x, 52.5, 12.2, 10.5, t, s, g, ec=ec, ts=8.4)
arrow(45.6, 42.2)

# 4 COMBINER
box(50, 39.5, 66, 5, "4 · COMBINER — glass-box GAM + Bernstein online aggregation",
    "interpretable links only · Adebayo-faithful", fc=LINKBG, ec=LINK, ts=9.5)
arrow(37, 33.5)

# 5 ENGINE
box(50, 30.5, 62, 5, "5 · INTERACTIVE ENGINE",
    "analyze · compare · scenario · world-sim · portfolio", fc=LINKBG, ec=LINK, ts=9.5)
arrow(28, 24.5)

# 6 CONTRACT
box(50, 21.5, 62, 5, "6 · CONVERSATIONAL CONTRACT",
    "provenance ledger — modules own every number", fc=LINKBG, ec=LINK, ts=9.5)
arrow(19, 15.2)

# answer
box(50, 12.3, 52, 4.6, "EXPLAINED ANSWER  ·  every number from a calibrated module",
    fc=TEAL, ec=TEAL, tc="white", ts=10)

# design law footer
ax.text(50, 5.6, "Design law:  (1) no shared trainable backbone   (2) interpretable links only   "
                 "(3) every claim gated by an OOS test AND an adversarial null",
        ha="center", fontsize=8.3, color=MUT, family="monospace")
ax.text(50, 3.0, "compiled 2026-07-30 · measurement-and-forecast system, not investment advice",
        ha="center", fontsize=7.5, color="#98a2b5")

fig.savefig(OUT / "architecture.png", facecolor="white", bbox_inches="tight", pad_inches=0.25)
print("wrote", OUT / "architecture.png")
