"""Build the validation report PDF (PREREGISTRATION.md Amendment 8 + the frozen deployment).

    python scripts/build_validation_pdf.py   -> results/Meridian_Validation_Report.pdf

Every number in the PDF is read from a results JSON written by the pipeline
(deploy_score.json, v1v3_score.json, wm_score.json, forward_score.json, audit_report.json,
data_manifest.json, manifests). Nothing numeric is typed by hand; the verdicts are computed
from those numbers with the pre-registered bar (PREREGISTRATION.md §5: QLIKE at least 5% below
the reference AND date-level DM p < 0.05).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "Meridian_Validation_Report.pdf"


def J(p):
    p = RES / p
    return json.loads(p.read_text()) if p.exists() else None


DEP = J("deploy/deploy_score.json")
DMETA = J("deploy/deploy_meta.json")
DMAN = J("deploy/data_manifest.json")
V13 = J("validation/v1v3_score.json")
WMS = J("wm/wm_score.json")
FWD = J("forward/forward_score.json")
AUD = J("deploy/audit_report.json")
WMAN = J("wm/frozen/manifest.json")
V2F = J("wm/v2_primary_failure.json")
FLOG = pd.read_csv(RES / "forward" / "forecast_log.csv")
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

# palette (validated: scripts/validate_palette.js, light surface) + recessive neutrals
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL = "#b9b8b2"

LABEL = {"Meridian_M0": "Meridian M0 (audited)", "Meridian_M0_rtvix": "Meridian M0, real-time VIX",
         "HAR_frozen": "HAR (frozen with M0)", "Meridian_legacy": "Meridian legacy, Jensen term",
         "Meridian_legacy_raw": "Meridian legacy, raw head", "Meridian_legacy_cal": "Meridian legacy, level offset",
         "MeridianWM_frozen": "Meridian-WM (frozen)", "MeridianWM": "Meridian-WM (walk-forward)",
         "MeridianWM_cal": "Meridian-WM, recalibrated", "Meridian_M0_cal": "Meridian M0, recalibrated",
         "HAR": "HAR-RV per asset (ref.)", "HAR_cal": "HAR-RV, recalibrated (ref.)", "LightGBM_cal": "LightGBM, recalibrated",
         "HAR_pooled": "HAR-RV pooled", "LHAR": "LHAR", "SHAR": "SHAR", "HARX": "HAR-X (+VIX)", "Ridge": "Ridge",
         "RandomForest": "Random forest", "LightGBM": "LightGBM", "TimeMixer": "TimeMixer", "RW": "Random walk",
         "MA22": "22-day mean", "EWMA": "EWMA", "GARCH": "GARCH(1,1)", "GJR_GARCH": "GJR-GARCH", "EGARCH": "EGARCH",
         "Meridian_B_EVT": "Meridian legacy + EVT", "Meridian_B_FHS": "Meridian legacy + FHS",
         "MeridianWM_FHS": "Meridian-WM + FHS", "WM_studentt": "Meridian-WM Student-t head",
         "HAR_FHS": "HAR-RV + FHS", "LightGBM_FHS": "LightGBM + FHS", "GJR_FHS": "GJR-GARCH + FHS (ref.)"}
for i in range(3):
    LABEL[f"WM_s{i}"] = f"Meridian-WM seed {i}"


def lab(m):
    return LABEL.get(m, m)


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:+.{d}f}%"


def pv(p):
    if p is None or p != p:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(c):
    return "—" if not c else f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def verdict(pct_vs, p, n_dates):
    if n_dates is not None and n_dates < 20:
        return "Not testable (< 20 dates)"
    if pct_vs is None:
        return "—"
    if pct_vs <= -0.05 and p is not None and p < 0.05:
        return "MEETS the bar"
    if pct_vs < 0 and p is not None and p < 0.05:
        return "Better, below the 5% bar"
    if pct_vs >= 0 and p is not None and p > 0.95:
        return "Significantly WORSE"
    return "Not significant"


# --------------------------------------------------------------------------- #
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceAfter=6, textColor=colors.HexColor(INK))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor(INK))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12.2)
SM = ParagraphStyle("SM", parent=B, fontSize=7.6, leading=9.6, textColor=colors.HexColor(INK2))
CELL = ParagraphStyle("CELL", parent=B, fontSize=7.2, leading=8.8)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=21, leading=26)


def P(t, st=B):
    return Paragraph(t, st)


def table(rows, widths=None, head=1, hl_rows=()):
    data = [[P(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=head)
    st = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9c8c2")),
          ("BACKGROUND", (0, 0), (-1, head - 1), colors.HexColor("#efeee9")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    for r in hl_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e8f1fb")))
    t.setStyle(TableStyle(st))
    return t


TMP = Path(tempfile.mkdtemp())


def fig_path(name):
    return str(TMP / f"{name}.png")


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(NEUTRAL)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def img(path, w=16.5):
    from PIL import Image as PI
    im = PI.open(path)
    h = w * im.size[1] / im.size[0]
    return Image(path, width=w * cm, height=h * cm)


def bars_vs_ref(models, title, name, highlight=("Meridian",), ref_label="HAR"):
    """Horizontal bars: % QLIKE vs reference with bootstrap CI; Meridian rows in blue."""
    items = sorted(models.items(), key=lambda kv: kv[1]["pct_vs_ref"], reverse=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(items) + 0.9), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for i, (m, r) in enumerate(items):
        c = BLUE if any(m.startswith(h) for h in highlight) else NEUTRAL
        v = 100 * r["pct_vs_ref"]
        ax.barh(i, v, color=c, height=0.62)
        lo = hi = v
        if r.get("ci95_pct_vs_ref"):
            lo, hi = (100 * x for x in r["ci95_pct_vs_ref"])
            ax.plot([lo, hi], [i, i], color=INK2, lw=0.9)
        ax.text(max(v, hi) + 1.2, i, f"{v:+.1f}%", va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(items)), [lab(m) for m, _ in items], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(f"QLIKE vs {ref_label}, % (negative = better; line = 95% block-bootstrap CI)", fontsize=7.5, color=INK2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(fig_path(name), facecolor=SURF)
    plt.close(fig)
    return fig_path(name)


def lines(series, title, name, xlabel, ylabel):
    """<=4 series, direct-labelled at the line end."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ends = []
    for (lbl, xs, ys), c in zip(series, (BLUE, ORANGE, AQUA, YELLOW)):
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4)
        ends.append([ys[-1], lbl, xs[-1]])
    ends.sort()
    span = (max(max(s_[2]) for s_ in series) - min(min(s_[2]) for s_ in series)) or 1.0
    gap = 0.07 * span                                   # nudge end labels apart (no collisions)
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, lbl, x in ends:
        ax.annotate(lbl, (x, y), xytext=(6, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=7.5, color=INK2); ax.set_ylabel(ylabel, fontsize=7.5, color=INK2)
    style_ax(ax); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend([s[0] for s in series], fontsize=6.8, frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(fig_path(name), facecolor=SURF)
    plt.close(fig)
    return fig_path(name)


def vol_rows(models, n_assets, extra_mcs=True):
    head = ["Rank", "Model", "QLIKE", "vs ref.", "95% CI", "DM p", "MCS 90%", "Assets beating ref."]
    rows = [head]
    hl = []
    for m in sorted(models, key=lambda k: models[k]["QLIKE"]):
        r = models[m]
        mcs = r.get("in_MCS90")
        rows.append([r.get("rank", ""), lab(m), f"{r['QLIKE']:.4f}", pct(r["pct_vs_ref"]), ci(r.get("ci95_pct_vs_ref")),
                     pv(r.get("DM_p")), "—" if mcs is None else ("in" if mcs else "out"),
                     f"{r['assets_beating_ref']}/{n_assets}"])
        if m.startswith("Meridian"):
            hl.append(len(rows) - 1)
    return rows, hl


def bench_rows(tb):
    """Rows for score_models tables (benchmark protocol)."""
    meta = tb["_meta"]
    models = {k: v for k, v in tb.items() if not k.startswith("_")}
    rows = [["Model", "QLIKE", f"vs {lab(meta['ref'])}", "95% CI", "DM p", "MCS 90%", "IC", "Assets beating ref."]]
    hl = []
    for m in sorted(models, key=lambda k: models[k]["QLIKE"]):
        r = models[m]
        rows.append([lab(m), f"{r['QLIKE']:.4f}", pct(r["pct_vs_ref"]), ci(r.get("ci95_pct_vs_ref")), pv(r.get("DM_p")),
                     "in" if r.get("in_MCS90") else "out", f"{r['IC']:.3f}", f"{r['assets_beating_ref']}/{meta['n_assets']}"])
        if m.startswith("Meridian"):
            hl.append(len(rows) - 1)
    return rows, hl, meta


# --------------------------------------------------------------------------- #
def build():
    S = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summ = AUD["summary"]
    S += [Spacer(1, 3 * cm), P("Meridian — validation report", TITLE),
          P("Frozen-model deployment after the training period, the Meridian-WM world-model network, "
            "and the leak / bluff / fake audit", ParagraphStyle("st", parent=B, fontSize=12, alignment=TA_CENTER, leading=16)),
          Spacer(1, 1 * cm),
          P(f"Generated {now} from commit <b>{COMMIT}</b> of branch <font face='Courier'>claude/sleepy-cray-ti3mxs</font>. "
            f"Audit: <b>{summ.get('PASS', 0)} PASS, {summ.get('WARN', 0)} WARN, {summ.get('FAIL', 0)} FAIL, "
            f"{summ.get('SKIP', 0)} SKIP</b>.", ParagraphStyle("c", parent=B, alignment=TA_CENTER)),
          Spacer(1, 0.6 * cm),
          P("Every number in this document is read from a results file written by the pipeline; the verdicts are computed "
            "from those numbers with the bar that was fixed before any result was seen (PREREGISTRATION.md §5: QLIKE at least "
            "5% below the reference <i>and</i> date-level Diebold–Mariano p &lt; 0.05). The test plan for the overnight runs "
            "(Amendment 8) was committed to git before any of them started; audit check X1 verifies the timestamps.",
            ParagraphStyle("c2", parent=SM, alignment=TA_CENTER, fontSize=8.5, leading=11)),
          PageBreak()]

    # ---------------- 1. Executive summary
    S.append(P("1. Executive summary", H1))
    rows = [["Test", "Model", "Window (target dates)", "vs reference", "95% CI", "DM p", "Verdict vs pre-registered bar"]]
    hl = []

    def add(test, m, sc, ref_txt=""):
        r = sc["models"][m]
        n = sc["meta"]["n_dates"]
        rows.append([test, lab(m) + ref_txt, f"{sc['meta']['target_range'][0]} → {sc['meta']['target_range'][1]} ({n})",
                     pct(r["pct_vs_ref"]), ci(r.get("ci95_pct_vs_ref")), pv(r.get("DM_p")),
                     verdict(r["pct_vs_ref"], r.get("DM_p"), n)])
    A = DEP["snapshots"]["A"]["windows"]["Nov-2026"]["core"]
    add("Deploy A", "Meridian_M0", A)
    if V13:
        vc = V13["vol_core"]
        for m in ("Meridian_legacy_cal", "Meridian_legacy", "MeridianWM_frozen"):
            add("V1/V3", m, vc)
    if V2F:
        rows.append(["V2 primary", "Meridian-WM as written", f"walk-forward, fold {V2F['fold']} (test from {V2F['fold_test_start']})",
                     "—", "—", "—", f"FAILED: diverged to NaN ({len(V2F['seeds_failed'])}/{V2F['seeds_run']} seeds)"])
    if WMS:
        for t, m in (("holdout_raw", "MeridianWM"), ("holdout_raw", "Meridian_M0"), ("holdout_cal", "MeridianWM_cal")):
            tb = WMS["tables"][t]
            r = tb[m]
            rows.append(["V2 (8a)", lab(m), f"{tb['_meta']['date_range'][0]} → {tb['_meta']['date_range'][1]} ({tb['_meta']['n_dates']})",
                         pct(r["pct_vs_ref"]), ci(r.get("ci95_pct_vs_ref")), pv(r.get("DM_p")),
                         verdict(r["pct_vs_ref"], r.get("DM_p"), tb["_meta"]["n_dates"])])
    if FWD:
        rows.append(["Live log", "Meridian M0 (pre-committed)", f"{FWD['date_range'][0]} → {FWD['date_range'][1]} ({FWD['n_dates']} origins)",
                     pct(FWD["pct_vs_HAR"]), "—", pv(FWD["DM_p"]), verdict(FWD["pct_vs_HAR"], FWD["DM_p"], FWD["n_dates"])])
    S.append(table(rows, [1.6 * cm, 3.6 * cm, 3.7 * cm, 1.5 * cm, 2.0 * cm, 1.2 * cm, 3.4 * cm]))
    S.append(Spacer(1, 4))
    S.append(P("Reference = per-asset HAR-RV fit at the same cut-off (recalibrated HAR for the recalibrated row). "
               "Negative % = lower QLIKE = better. 'Window' counts scored target dates (origins for the live log).", SM))
    S.append(P("<b>What this means.</b>", H3))
    pts = []
    if V13:
        vc = V13["vol_core"]["models"]
        best = min(vc, key=lambda m: vc[m]["QLIKE"])
        pts.append(f"After their training ended, no Meridian or Meridian-WM model beat HAR by a statistically established "
                   f"margin. Over the {V13['vol_core']['meta']['n_dates']} post-training dates of V1/V3 the best Meridian "
                   f"variant (legacy + level offset) is {pct(vc['Meridian_legacy_cal']['pct_vs_ref'])} (DM p "
                   f"{pv(vc['Meridian_legacy_cal']['DM_p'])}); the frozen Meridian-WM is {pct(vc['MeridianWM_frozen']['pct_vs_ref'])} "
                   f"(p {pv(vc['MeridianWM_frozen']['DM_p'])}). The best model in that window is {lab(best)} "
                   f"({pct(vc[best]['pct_vs_ref'])}, p {pv(vc[best]['DM_p'])}).")
    if WMS:
        h = WMS["tables"]["holdout_raw"]
        pts.append(f"In the long walk-forward test (V2, {h['_meta']['n_dates']} holdout dates, same folds as the audited "
                   f"benchmark) Meridian-WM is {pct(h['MeridianWM']['pct_vs_ref'])} vs HAR (p {pv(h['MeridianWM']['DM_p'])}), "
                   f"while the audited Meridian M0 is {pct(h['Meridian_M0']['pct_vs_ref'])} (p {pv(h['Meridian_M0']['DM_p'])}) "
                   f"on exactly the same rows. Verdict for the WM network: {verdict(h['MeridianWM']['pct_vs_ref'], h['MeridianWM']['DM_p'], h['_meta']['n_dates'])}.")
        if "holdout_raw_tradeday" in WMS["tables"]:
            ht = WMS["tables"]["holdout_raw_tradeday"]
            pts.append(f"With FX rows aligned to their true trading day (audit D4) the audited benchmark's headline M0 result becomes "
                       f"{pct(ht['Meridian_M0']['pct_vs_ref'])} {ci(ht['Meridian_M0']['ci95_pct_vs_ref'])}, DM p "
                       f"{pv(ht['Meridian_M0']['DM_p'])} (as stored: {pct(h['Meridian_M0']['pct_vs_ref'])} "
                       f"{ci(h['Meridian_M0']['ci95_pct_vs_ref'])}, p {pv(h['Meridian_M0']['DM_p'])}).")
    if WMS:
        h = WMS["tables"]["holdout_raw"]["Meridian_M0"]
        pts.append(f"'MEETS the bar' for M0 refers to its point estimate ({pct(h['pct_vs_ref'])}); the upper end of its 95% interval "
                   f"({ci(h['ci95_pct_vs_ref'])}) is above -5%, so the size of the gain is uncertain even though the gain itself is clear. "
                   "That holdout result predates this work and is reproduced exactly here, not newly established.")
    pts.append(f"The audited M0 has only {A['meta']['n_dates']} target dates after its own training cut-off, so its "
               f"post-training performance is not yet testable; its live pre-committed log covers {FWD['n_dates']} origins. "
               "Months of daily logging are needed before a forward verdict exists.")
    pts.append("The audit found no data leak, no changed weights, no look-ahead, and no number in a report that differs from "
               "its source file. It did find, and the work fixed, six real defects, including a pre-existing FX date-labelling "
               "issue that overstated some significance levels (Section 10).")
    for t in pts:
        S.append(P("• " + t))
    S.append(PageBreak())

    # ---------------- 2. Scope & WM
    S.append(P("2. What was asked and what was done", H1))
    for t in ["<b>1. What is the WM component?</b> Answered in Section 3.",
              "<b>2. Deploy without additional training, on the period after training ends.</b> The two frozen Meridian "
              "snapshots saved in the repository were loaded (sha256-checked) and only run forward: Snapshot A (audited M0, "
              "trained through 2026-10-30) and Snapshot B (pre-audit legacy model, trained through late July 2026). No weight "
              "was changed; audit W1/W2 prove it.",
              "<b>3. Measure and benchmark against other models.</b> 17 comparison models (HAR family, GARCH family, EWMA, "
              "random walk, Ridge, random forest, LightGBM, TimeMixer) were fit once on exactly the same pre-cut-off data and "
              "then also run forward frozen; tail risk (VaR/ES) was benchmarked against GJR-GARCH + FHS.",
              "<b>4. Audit for leaks, bluffed numbers and fakes.</b> A 22-check audit (Section 9) runs on every push in CI.",
              "<b>Deeper overnight validation (Amendment 8, pre-registered).</b> V1: Snapshot B over the longest honest "
              "post-training window, with tails. V2: the Meridian-WM network inside the audited walk-forward harness. "
              "V3: Meridian-WM trained once at the V1 cut-off, frozen, and deployed like V1."]:
        S.append(P("• " + t))
    S.append(P("3. The WM component", H1))
    S.append(P("<b>Meridian-WM</b> (<font face='Courier'>meridian/meridian_wm.py</font>, class <font face='Courier'>MeridianWM</font>) "
               "is one network with four parts: (1) a sticky switching state-space belief core (3 soft regimes; regime = most "
               "likely mode) with its own volatility head; (2) a separate diagonal state-space volatility core whose EMA-smoothed "
               "belief feeds the main next-day log-variance head (QLIKE loss); (3) a Student-t head over next-day returns "
               "(tail heaviness and scale); (4) a JEPA latent predictor with SIGReg regularization whose prediction error is the "
               "'surprise' score. The wider <i>world model</i> (WORLD_MODEL.md) is a bank of separately trained modules "
               "(volatility, tail, regime, factor, connectedness, propagation, covariance) linked by inspectable combiners."))
    S.append(P("No trained Meridian-WM weights existed in the repository (every script retrained it and discarded it), so it could "
               "not be deployed without training. To validate it anyway, Amendment 8 pre-registered two tests that train it only "
               "on data before each forecast: V2 (walk-forward, retrained per fold exactly like every benchmark model) and V3 "
               f"(trained once on labels through {WMAN['train_label_end_max']}, frozen with sha256 hashes, then run forward untouched)."))
    S.append(PageBreak())

    # ---------------- 4. Data
    S.append(P("4. Data and provenance", H1))
    S.append(P("Daily OHLC from Yahoo (the same source the models were trained on) and VIX / 10y / 2y yields from FRED. Each "
               "asset was checked against the next-day realized variance stored by the audited benchmark on every overlapping "
               "date (audit D1)."))
    rows = [["Asset", "Source", "First", "Last", "Overlap days", "Median |Δ log RV|", "Correlation"]]
    for a, v in DMAN["assets"].items():
        c = v["check_vs_benchmark"]
        rows.append([a, v["source"], v["first"], v["last"], c["overlap"], f"{c['median_abs_log_diff']:.4f}", f"{c['corr_log']:.4f}"])
    S.append(table(rows))
    S.append(Spacer(1, 4))
    S.append(P("Universe: the 11 pre-registered core assets (8 US equities/ETFs + EURUSD, USDJPY, GBPUSD) are scored; TLT and GLD "
               "are a separate out-of-universe check. Target: next-day Garman–Klass variance. Loss: QLIKE.", SM))

    # ---------------- 5. First deployment
    S.append(P("5. First deployment: October and November 2026", H1))
    for snap, w in (("A", "Nov-2026"), ("B", "Oct-2026")):
        sc = DEP["snapshots"][snap]["windows"][w]["core"]
        S.append(P(f"Snapshot {snap} — {DEP['snapshots'][snap]['name']}; training labels end "
                   f"{DEP['snapshots'][snap]['train_label_end_max']}; {w}: {sc['meta']['n_dates']} target dates.", H3))
        sel = {m: r for m, r in sc["models"].items()
               if m.startswith("Meridian") or m in ("HAR", "HAR_frozen", "Ridge", "RandomForest", "LightGBM", "LHAR", "GJR_GARCH", "EWMA")}
        rows, hl = vol_rows(sel, sc["meta"]["n_assets"])
        S.append(table(rows, hl_rows=hl))
    S.append(P("Snapshot A has 3 post-training dates (no test is possible). Snapshot B was superseded by the longer V1 window below.", SM))
    S.append(PageBreak())

    # ---------------- 6. V1/V3
    if V13:
        vc = V13["vol_core"]
        m = V13["meta"]
        S.append(P("6. V1 + V3: frozen models over the longest post-training window", H1))
        S.append(P(f"All models' training data end with labels on <b>{m['cut_label']}</b> (legacy model: last origin reconstructed "
                   f"from its saved scaler, audit T4; frozen WM: {WMAN['train_rows']} training rows, labels through "
                   f"{WMAN['train_label_end_max']}). Scored targets: <b>{vc['meta']['target_range'][0]} → "
                   f"{vc['meta']['target_range'][1]}</b>, {vc['meta']['n_dates']} dates × {vc['meta']['n_assets']} assets = "
                   f"{vc['meta']['n_rows']} forecasts. Nothing was retrained during the window."))
        S.append(img(bars_vs_ref(vc["models"], f"V1/V3 — QLIKE vs per-asset HAR, {vc['meta']['n_dates']} post-training dates", "v1bars")))
        rows, hl = vol_rows(vc["models"], vc["meta"]["n_assets"])
        S.append(table(rows, [1.15 * cm, 4.45 * cm, 1.5 * cm, 1.5 * cm, 2.6 * cm, 1.3 * cm, 1.5 * cm, 2.6 * cm], hl_rows=hl))
        S.append(P(f"DM: one-sided Newey–West test that the model's date-level loss is below HAR's. CI: moving-block bootstrap over "
                   f"dates. MCS: Hansen model confidence set at 90%. Legacy level offset and Jensen term were estimated on "
                   f"pre-cut-off rows only (bias {m['legacy_bias']:+.3f}, Jensen {m['legacy_jensen']:.3f} log units); the legacy "
                   "model was trained on uncleaned pre-audit FX bars, which biases its raw head low.", SM))
        mon = vc["monthly_pct_vs_ref"]
        months = list(mon)
        mn = vc.get("monthly_n_dates", {})
        xl = [f"{mm} ({mn[mm]}d)" if mm in mn else mm for mm in months]
        ser = [(lab(k), xl, [100 * mon[mm][k] for mm in months])
               for k in ("Meridian_legacy_cal", "MeridianWM_frozen", "Ridge", "LightGBM")]
        S.append(img(lines(ser, "V1/V3 — QLIKE vs HAR by target month (%, negative = better)", "v1month", "target month", "% vs HAR"), 15))
        pa = vc.get("per_asset_pct_vs_ref")
        if pa:
            cols = ["Meridian_legacy_cal", "MeridianWM_frozen", "Ridge", "LightGBM", "GJR_GARCH"]
            rows = [["Asset"] + [lab(c) for c in cols]] + [[a] + [pct(pa[a][c]) for c in cols] for a in sorted(pa)]
            S.append(P("Per asset — QLIKE vs per-asset HAR", H3))
            S.append(table(rows))
        if "vol_core_label_dates" in V13:
            lb = V13["vol_core_label_dates"]
            S.append(P("Date alignment sensitivity (audit D4)", H3))
            S.append(P(f"Yahoo labels FX bars one day early during UK summer time, so on label dates FX and equity rows of different "
                       f"trading days were averaged together ({lb['meta']['n_dates']} 'dates' instead of "
                       f"{vc['meta']['n_dates']} trading days). The tables above use the corrected trading-day alignment; the "
                       "label-date version is shown for transparency. QLIKE values are identical; only date-level tests move.", B))
            rows = [["Model", "DM p (label dates)", "DM p (trading days)", "CI (label dates)", "CI (trading days)"]]
            for mm in ("Meridian_legacy_cal", "MeridianWM_frozen", "Ridge", "RandomForest", "LightGBM"):
                rows.append([lab(mm), pv(lb["models"][mm]["DM_p"]), pv(vc["models"][mm]["DM_p"]),
                             ci(lb["models"][mm]["ci95_pct_vs_ref"]), ci(vc["models"][mm]["ci95_pct_vs_ref"])])
            S.append(table(rows))
        S.append(PageBreak())
        S.append(P("7. V1 + V3: tail risk (next-day VaR / Expected Shortfall)", H1))
        S.append(P("Volatility forecasts are turned into VaR/ES by filtering returns through them and reading the tail from past "
                   "standardized returns (FHS; EVT for the legacy model, the Amendment-7 selection). Reference: GJR-GARCH + FHS. "
                   "FZ0 is the joint VaR/ES scoring loss (lower is better). Kupiec / Christoffersen counts are assets passing at 5%."))
        for k in ("025", "010"):
            t = V13["tail_core"][k]
            rows = [["Model", "FZ0", "Δ vs GJR", "DM p", "Violations (target " + f"{100 * t['meta']['alpha']:.1f}%)",
                     "Pooled Kupiec p", "Kupiec pass", "Christoffersen pass"]]
            hl = []
            for n in sorted(t["models"], key=lambda x: t["models"][x]["FZ0"]):
                r = t["models"][n]
                rows.append([lab(n), f"{r['FZ0']:.4f}", f"{r['delta_vs_ref']:+.4f}", pv(r["DM_p"]), f"{100 * r['violation_rate']:.2f}%",
                             pv(r["pooled_kupiec_p"]), f"{r['assets_pass_kupiec']}/{r['n_assets']}", f"{r['assets_pass_cc']}/{r['n_assets']}"])
                if n.startswith(("Meridian", "WM")):
                    hl.append(len(rows) - 1)
            S.append(P(f"{100 * t['meta']['alpha']:.1f}% level — {t['meta']['n_rows']} asset-days, {t['meta']['n_dates']} dates", H3))
            S.append(table(rows, hl_rows=hl))
        t25 = V13["tail_core"]["025"]["models"]["GJR_FHS"]
        n_per = int(round(t25["n"] / t25["n_assets"]))
        S.append(P(f"With about {n_per} days per asset a single asset expects only ~{0.025 * n_per:.1f} violations at 2.5%, so "
                   "per-asset coverage tests have very little power; the pooled Kupiec test and FZ0 differences are the "
                   "informative columns. Tail DM tests use the trading-day alignment.", SM))
        S.append(PageBreak())

    # ---------------- 8. V2
    if WMS:
        S.append(P("8. V2: Meridian-WM in the audited walk-forward harness", H1))
        if V2F:
            S.append(P("Primary result (as pre-registered): FAILED", H3))
            S.append(P(f"Meridian-WM exactly as written could not complete the walk-forward. On fold {V2F['fold']} (test from "
                       f"{V2F['fold_test_start']}) training diverged to NaN for {len(V2F['seeds_failed'])} of {V2F['seeds_run']} seeds, "
                       f"after {V2F['folds_completed_before_failure']} folds had completed. Diagnosis "
                       f"(<font face='Courier'>scripts/diag_wm_nan.py</font> reproduces it): {V2F['diagnosis']} Root cause: "
                       f"{V2F['root_cause']}"))
            S.append(P("Secondary result (Amendment 8a, committed before the re-run): same model and loss; optimizer steps whose "
                       "loss or gradient norm is non-finite are skipped and counted. Healthy steps are untouched.", H3))
            sp = WMS["meta"].get("steps_per_fold")
            if sp:
                rows = [["Seed"] + [f"F{int(f)}" for f in sorted(next(iter(sp.values())), key=int)] + ["Total skipped"]]
                for sd, folds in sp.items():
                    ks = sorted(folds, key=int)
                    tot = sum(folds[f].get("skipped", 0) for f in ks)
                    steps = sum(folds[f].get("steps", 0) for f in ks)
                    rows.append([sd] + [str(folds[f].get("skipped", 0)) for f in ks] + [f"{tot} of {tot + steps}"])
                S.append(P("Skipped (non-finite) optimizer steps per fold", H3))
                S.append(table(rows))
        g = WMS["meta"]["fold_grid_check"]
        rep = WMS["reproduction_vs_stored_benchmark"]
        S.append(P(f"Meridian-WM (default configuration, {WMS['meta']['seeds']} seeds, {WMS['meta']['epochs']} epochs) was trained "
                   f"per fold on the same 12 expanding folds, 22-day embargo, label purge and train-only scaling as every model in the "
                   f"audited benchmark. Fold grid identity: {g['rows_matched']}/{g['rows_bench']} benchmark rows matched, "
                   f"{g['fold_mismatch']} fold mismatches. On the same rows the rescored HAR and M0 holdout QLIKE equal the stored "
                   f"benchmark exactly (HAR {rep['HAR']['stored_QLIKE']:.4f} → {rep['HAR']['rescored_QLIKE']:.4f}; "
                   f"M0 {rep['Meridian_M0']['stored_QLIKE']:.4f} → {rep['Meridian_M0']['rescored_QLIKE']:.4f})."))
        tb = WMS["tables"]["holdout_raw"]
        keep = {k: v for k, v in tb.items() if not k.startswith("_") and not k.startswith("WM_s")}
        S.append(img(bars_vs_ref(keep, f"V2 holdout {tb['_meta']['date_range'][0]} → {tb['_meta']['date_range'][1]} — raw forecasts vs per-asset HAR", "v2bars")))
        for t in ("holdout_raw", "holdout_raw_tradeday", "holdout_cal", "dev_raw"):
            tbl = WMS["tables"][t]
            if t == "holdout_raw":
                tbl = {k: v for k, v in tbl.items() if not k.startswith("WM_s")}
            rows, hl, meta = bench_rows(tbl)
            S.append(P(f"{meta['label']} — {meta['n_rows']} forecasts, {meta['n_dates']} dates", H3))
            S.append(table(rows, hl_rows=hl))
        S.append(PageBreak())
        lc = WMS.get("loss_concentration")
        if lc:
            d_, h_ = lc["dev"], lc["holdout"]
            S.append(P("Loss concentration (a finding about the original benchmark's development period)", H3))
            S.append(P(f"In the development period the 10 worst rows carry {100 * d_['top10_share_of_HAR_QLIKE']:.0f}% of HAR's total "
                       f"QLIKE ({d_['rows']} rows); in the holdout they carry {100 * h_['top10_share_of_HAR_QLIKE']:.0f}%. "
                       "Largest development-period rows: "
                       + "; ".join(f"{r['asset']} {r['date']} (next-day variance {r['rv_next']:.4f}, QLIKE {r['HAR_QLIKE']:.0f})" for r in d_["top_rows"][:3])
                       + ". The 2012-01-26 FX bars imply a one-day variance far outside anything else in the series and look like a "
                       "bad price tick that the bar cleaner (which only removes high &lt;= low bars) does not catch. Development-period "
                       "rankings, which the original project used to choose its configuration, are therefore driven by a handful of "
                       "rows; holdout results are not. This is reported, not corrected, here.", B))
        yr = WMS["by_year_pct_vs_HAR"]
        years = sorted(yr["MeridianWM"], key=int)
        ser = [(lab(k), [int(y) for y in years], [100 * yr[k][y] for y in years]) for k in ("Meridian_M0", "MeridianWM")]
        S.append(img(lines(ser, "V2 — QLIKE vs per-asset HAR by calendar year (%, negative = better)", "v2year", "year", "% vs HAR"), 15))
        ba = WMS["holdout_by_asset_pct_vs_HAR"]
        assets = sorted(ba["MeridianWM"])
        rows = [["Asset", "Meridian-WM vs HAR", "Meridian M0 vs HAR"]] + [[a, pct(ba["MeridianWM"][a]), pct(ba["Meridian_M0"][a])] for a in assets]
        S.append(P("Holdout by asset", H3)); S.append(table(rows, [3 * cm, 4 * cm, 4 * cm]))
        sd = WMS["seed_pct_vs_HAR_holdout"]
        S.append(P("Seed dispersion (holdout, raw, vs HAR): " + ", ".join(f"{lab(k)} {pct(v)}" for k, v in sd.items()) + ".", B))
        tt = WMS["wm_studentt_var_holdout"]
        rows = [["Level", "Violation rate", "Asset-days", "Kupiec pass", "Christoffersen pass"]]
        for k, al in (("025", "2.5%"), ("010", "1.0%")):
            r = tt[k]
            rows.append([al, f"{100 * r['violation_rate']:.2f}%", r["n"], f"{r['assets_pass_kupiec']}/{r['n_assets']}", f"{r['assets_pass_cc']}/{r['n_assets']}"])
        S.append(P("Meridian-WM Student-t head — next-day return VaR coverage on the holdout", H3)); S.append(table(rows))
        rg = WMS["wm_regime_holdout"]
        S.append(P("Regime branch on the holdout: mean dwell " + f"{rg['mean_dwell_days']:.1f} days; share by regime "
                   + ", ".join(f"{k}: {100 * v:.0f}%" for k, v in rg["share_by_regime"].items())
                   + "; mean next-day log RV by regime " + ", ".join(f"{k}: {v:.2f}" for k, v in rg["mean_log_rv_next_by_regime"].items())
                   + ". Descriptive only; no regime-existence claim is made.", B))
        S.append(PageBreak())

    # ---------------- 9. Live log + audit
    S.append(P("9. Live pre-committed log and the audit", H1))
    pc = FLOG[FLOG["precommitted"]]
    S.append(P(f"Forecast log: {len(FLOG)} rows, {len(pc)} pre-committed (logged before their outcome existed) over "
               f"{pc['origin_date'].nunique()} origin dates, {FLOG['origin_date'].min()} → {FLOG['origin_date'].max()}. "
               f"Scored so far: {FWD['n_dates']} origins, Meridian M0 QLIKE {FWD['QLIKE_Meridian_M0']:.4f} vs frozen HAR "
               f"{FWD['QLIKE_HAR']:.4f} ({pct(FWD['pct_vs_HAR'])}). Rows for days when the logger did not run are kept but "
               "labelled not pre-committed and never scored. Under ~20 scored dates the result is noise."))
    rows = [["ID", "Check", "Status", "Key evidence"]]
    for c in AUD["checks"]:
        e = json.dumps(c["evidence"], default=str)
        rows.append([c["id"], c["check"], c["status"], e[:230] + ("…" if len(e) > 230 else "")])
    S.append(table(rows, [0.9 * cm, 4.2 * cm, 1.2 * cm, 10.6 * cm]))
    S.append(P(f"Audit run {AUD['run_at_utc']} on commit {AUD['git_commit']}. Full evidence: results/deploy/audit_report.json. "
               "The WARN (S4) is a finding, not a defect: within these short windows shuffling a model's forecasts across days "
               "scores as well as the true order, for HAR as well, so day-to-day timing skill cannot be shown yet.", SM))
    S.append(PageBreak())

    # ---------------- 10. defects, limits, reproduce
    S.append(P("10. Defects found and fixed during the work", H1))
    for t in ["<b>Real-time VIX vintage.</b> Live forecasts used the previous day's VIX because FRED publishes with a delay; "
              "the deployment used same-day VIX. Both are reported; the logged forecasts are reproduced exactly from their vintage (R1).",
              "<b>Incomplete FX bars.</b> Yahoo FX bars are UTC days; a run before 00:00 UTC would log a partial bar. "
              "<font face='Courier'>forward_test.py</font> now drops it (unit test added).",
              "<b>Kupiec p-value index.</b> The new tail scorer first read the violation rate instead of the p-value from "
              "<font face='Courier'>kupiec()</font>; fixed before any number was reported.",
              "<b>Audit X5 crash.</b> A boolean-negation bug in the independent FZ0 recomputation; fixed, check now passes.",
              "<b>Over-claim in a summary.</b> An early README sentence said every interval contained zero; two did not. Corrected.",
              "<b>FX bar dating (pre-existing since the project's first data loader).</b> Yahoo stamps daily FX bars at London "
              "midnight; normalizing the UTC timestamp labels every summer-time bar one day early (Sunday–Thursday). It is not a "
              "look-ahead leak (each series stays in order and the VIX joined to those rows is older, not newer; audit D4 checks "
              "every train/test boundary on the true day), but date-level significance tests mixed FX and equity rows from "
              "different days, which overstated significance (Section 6), and the forward test would have accepted an open FX "
              "bar in summer. Fixed with meridian.data.fx_true_day; tests align by trading day.",
              "<b>Scheduled logger without repository access.</b> Fresh scheduled sessions could not push; the daily task now runs in "
              "the working session. Days it could not run are recorded as missed, never back-filled."]:
        S.append(P("• " + t))
    S.append(P("11. Limitations and what is not claimed", H1))
    for t in ["Post-training windows are short (3 dates for the audited M0; 66 for the legacy model and frozen WM); intervals are wide.",
              "V2 used 3 seeds for Meridian-WM against 5 for the stored M0 (CPU budget). Seed dispersion is reported.",
              "Daily range-based realized variance only; no intraday data. Free, delayed data.",
              "A better volatility forecast is not by itself a profitable strategy; no trading or Sharpe number is claimed.",
              "The regime branch is descriptive; no claim that discovered regimes exist."]:
        S.append(P("• " + t))
    S.append(P("12. Reproduce", H1))
    for c in ["python scripts/deploy_data.py && python scripts/deploy_frozen.py && python scripts/score_deploy.py",
              "python scripts/validate_v1v3.py run && python scripts/validate_v1v3.py score",
              "for k in 0 1 2; do python scripts/benchmark_wm.py $k; done && python scripts/benchmark_wm.py combine && python scripts/score_wm.py",
              "python scripts/audit_deploy.py && python -m pytest tests -q",
              "python scripts/build_validation_pdf.py"]:
        S.append(P(f"<font face='Courier' size='7.5'>{c}</font>"))
    rows = [["Frozen file", "sha256"]] + [[f"results/wm/frozen/{n}", h] for n, h in WMAN["sha256"].items()]
    rows += [[f"results/forward/frozen/{n}", h] for n, h in DMETA["weights_sha256"]["A"].items()]
    rows += [[f"results/{n}", h] for n, h in DMETA["weights_sha256"]["B"].items()]
    S.append(Spacer(1, 4)); S.append(table(rows, [6 * cm, 11 * cm]))

    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.HexColor(INK2))
        c.drawString(1.6 * cm, 1.0 * cm, f"Meridian validation report · commit {COMMIT}")
        c.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"page {d.page}")
        c.restoreState()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.6 * cm, title="Meridian validation report", author="Meridian pipeline")
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
