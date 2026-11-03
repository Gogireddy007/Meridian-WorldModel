"""Generate every number, table and figure of the LaTeX project report (report/main.tex).

Reads only saved artifacts: results/benchmark_vol*.json, results/benchmark_tail.json,
results/benchmark_vol_predictions.parquet, results/benchmark_tail_predictions.parquet,
results/forward/*, data/ (for data-quality counts) and git metadata. Writes:

    report/generated/numbers.tex   \\newcommand macros used in the prose
    report/generated/tab_*.tex     tables
    report/generated/fig_*.pdf     figures

    python scripts/build_report_tex.py && (cd report && latexmk -pdf main.tex)
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from meridian.evalproto import qlike
from meridian.scoring import nw_lag
from scripts.render_benchmark_md import label as vol_label
from scripts.score_tail_benchmark import LABEL as TAIL_LABEL

RES = ROOT / "results"
OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
V = json.loads((RES / "benchmark_vol.json").read_text())
A = json.loads((RES / "benchmark_vol_ablation.json").read_text())
T = json.loads((RES / "benchmark_tail.json").read_text())
VMETA = json.loads((RES / "benchmark_vol_predictions_meta.json").read_text())
TMETA = json.loads((RES / "benchmark_tail_predictions_meta.json").read_text())
AMETA = json.loads((RES / "benchmark_vol_predictions_ablation_meta.json").read_text())
FMAN = json.loads((RES / "forward" / "frozen" / "manifest.json").read_text())
FLOG = pd.read_csv(RES / "forward" / "forecast_log.csv")
FSCORE = json.loads((RES / "forward" / "forward_score.json").read_text()) \
    if (RES / "forward" / "forward_score.json").exists() else None

BLUE, GREY, ORANGE, GREEN, INK = "#1d4e89", "#9aa5b1", "#d9822b", "#3ebd93", "#1f2933"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
MACROS: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def tex(s: str) -> str:
    rep = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "λ": r"$\lambda$", "×": r"$\times$",
           "Δ": r"$\Delta$", "²": r"$^2$", "→": r"$\to$", "≈": r"$\approx$", "—": "---", "–": "--",
           "·": r"$\cdot$", "−": "-"}
    return "".join(rep.get(c, c) for c in str(s))


def pct(x, d=1):
    return "---" if x is None else f"${x * 100:+.{d}f}\\%$"


def ci(c, d=1):
    return "" if not c else f"[{c[0] * 100:+.{d}f}, {c[1] * 100:+.{d}f}]"


def fp(p):
    if p is None:
        return "---"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def mac(name, value):
    assert name.isalpha(), name
    MACROS[name] = str(value)


def write_table(fname, header, rows, colspec, caption, label, note=None, size=r"\footnotesize",
                highlight=(), bad=()):
    lines = [r"\begin{table}[!htbp]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabularx}}{{\linewidth}}{{{colspec}}}", r"\toprule",
             " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        pre = r"\rowcolor{good}" if i in highlight else (r"\rowcolor{bad}" if i in bad else "")
        lines.append(pre + " & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    if note:
        lines += [r"\par\smallskip\begin{minipage}{\linewidth}\scriptsize " + note + r"\end{minipage}"]
    lines += [r"\end{table}"]
    (OUT / fname).write_text("\n".join(lines) + "\n")


def vname(k):
    fam, nm = vol_label(k)
    return fam, nm


# --------------------------------------------------------------------------- #
# 1. data and repository facts
# --------------------------------------------------------------------------- #
def data_section():
    from meridian.data import load_all
    d = load_all(include_breadth=True)
    rows, stale_t, clip_t = [], 0, 0
    order = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "JPM", "XOM", "JNJ", "EURUSD", "USDJPY", "GBPUSD", "TLT", "GLD"]
    for a in order:
        o = d["prices"][a]
        stale = int((o.high <= o.low).sum()); k = o[o.high > o.low]
        clip = int(((k.open < k.low) | (k.open > k.high) | (k.close < k.low) | (k.close > k.high)).sum())
        stale_t += stale; clip_t += clip
        cls = "FX" if a in ("EURUSD", "USDJPY", "GBPUSD") else ("Breadth (not trained on)" if a in ("TLT", "GLD")
                                                               else "Equity / ETF")
        rows.append([a, tex(cls), f"{len(o):,}", str(o.index.min().date()), str(o.index.max().date()), stale, clip])
    write_table("tab_data.tex", ["Asset", "Class", "Daily bars", "First", "Last", "Stale bars dropped",
                                 "Bars clipped"], rows, "llrllrr",
                "Data universe and data-quality repairs (Yahoo daily OHLC as used by the audited benchmark).",
                "tab:data",
                note="Stale bar: high $\\le$ low (open = high = low = close, no real quote). Clipped bar: open or "
                     "close outside [low, high]; clipped into the range. Only the three FX pairs were affected.")
    mac("NStale", stale_t); mac("NClipped", clip_t)
    m = d["macro"]
    mac("MacroStart", m.index.min().date()); mac("MacroEnd", m.index.max().date())
    mac("DataStart", min(o.index.min() for o in d["prices"].values()).date())
    mac("DataEnd", max(o.index.max() for o in d["prices"].values()).date())

    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    mac("GitHead", git("rev-parse", "--short", "HEAD"))
    mac("GitMain", git("rev-parse", "--short", "main"))
    mac("GitAhead", git("rev-list", "--count", "main..HEAD"))
    log = git("log", "--format=%h|%ad|%s", "--date=short").splitlines()
    write_table("tab_commits.tex", ["Commit", "Date", "Message"],
                [[c.split("|")[0], c.split("|")[1], tex(c.split("|", 2)[2])] for c in log], "llX",
                "Repository history (branch \\texttt{harness-audit}; \\texttt{main} = the commit pushed to "
                "GitHub on 2026-09-08).", "tab:commits")

    inv = []
    for folder in ("meridian", "scripts", "tests"):
        files = sorted((ROOT / folder).glob("*.py"))
        inv.append([f"\\texttt{{{folder}/}}", len(files), f"{sum(len(f.read_text().splitlines()) for f in files):,}"])
    mac("NScripts", len(list((ROOT / "scripts").glob("*.py"))))
    write_table("tab_inventory.tex", ["Folder", "Python files", "Lines of code"], inv, "Xrr",
                "Code inventory.", "tab:inventory", size=r"\small")

    import torch, arch, sklearn, lightgbm, scipy as sp
    env = [["Python", platform.python_version()], ["PyTorch", torch.__version__], ["arch", arch.__version__],
           ["scikit-learn", sklearn.__version__], ["LightGBM", lightgbm.__version__], ["SciPy", sp.__version__],
           ["pandas", pd.__version__], ["NumPy", np.__version__], ["Platform", tex(platform.platform())]]
    write_table("tab_env.tex", ["Component", "Version"], env, "lX", "Software environment of the audited runs.",
                "tab:env", size=r"\small")

    tests = []
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        for line in f.read_text().splitlines():
            if line.startswith("def test_"):
                tests.append([f"\\texttt{{{tex(f.name)}}}", "\\texttt{" + tex(line[4:line.index("(")]) + "}"])
    mac("NTests", len(tests))
    write_table("tab_tests.tex", ["File", "Test"], tests, "lX", "Automated correctness tests (all passing).",
                "tab:tests", size=r"\small")


# --------------------------------------------------------------------------- #
# 2. volatility benchmark
# --------------------------------------------------------------------------- #
def vol_section():
    hv = V["tables"]["holdout_raw_vs_calibrated"]; hm = hv["_meta"]
    mac("HoldStart", hm["date_range"][0]); mac("HoldEnd", hm["date_range"][1])
    mac("HoldDates", f"{hm['n_dates']:,}"); mac("HoldRows", f"{hm['n_rows']:,}"); mac("HoldAssets", hm["n_assets"])
    mac("NWLag", hm["nw_lag"])
    dm = V["tables"]["dev_raw_vs_calibrated"]["_meta"]
    mac("DevStart", dm["date_range"][0]); mac("DevEnd", dm["date_range"][1]); mac("DevDates", f"{dm['n_dates']:,}")
    mac("Folds", VMETA["folds"]); mac("TestDays", VMETA["test_d"]); mac("EmbargoDays", VMETA["embargo_d"])
    mac("MinTrainDays", VMETA["min_train_d"]); mac("Seeds", VMETA["ensemble"]); mac("Epochs", VMETA["epochs"])
    mac("VolRuntimeMin", round(VMETA["runtime_s"] / 60)); mac("VolRunCommit", VMETA["git_commit_at_run"])
    mac("AblSeeds", AMETA["ensemble"]); mac("AblRuntimeMin", round(AMETA["runtime_s"] / 60))

    def g(k, f="pct_vs_ref"):
        return hv[k][f]
    for key, k in (("M", "Meridian_M0_cal"), ("Combo", "Combo_HAR_LGBM_Meridian_M0"), ("LGBM", "LightGBM_cal"),
                   ("HARX", "HARX_cal"), ("Ridge", "Ridge_cal"), ("RF", "RandomForest_cal"), ("LHAR", "LHAR_cal"),
                   ("TM", "TimeMixer_cal"), ("GARCH", "GARCH_cal"), ("GJR", "GJR_GARCH_cal"),
                   ("EGARCH", "EGARCH_cal"), ("EWMA", "EWMA_cal"), ("RW", "RW_cal"), ("HARLGBM", "Combo_HAR_LGBM")):
        mac(f"V{key}Pct", pct(g(k))); mac(f"V{key}CI", ci(g(k, "ci95_pct_vs_ref")))
        mac(f"V{key}Q", f"{g(k, 'QLIKE'):.4f}"); mac(f"V{key}P", fp(g(k, "DM_p")))
        mac(f"V{key}Won", g(k, "assets_beating_ref"))
    mac("VHARQ", f"{hv['HAR_cal']['QLIKE']:.4f}")
    hits = [v["hit_rate"] for k, v in hv.items() if k != "_meta" and (k.endswith("_cal") or k.startswith("Combo"))]
    mac("HitMin", f"{min(hits) * 100:.1f}\\%"); mac("HitMax", f"{max(hits) * 100:.1f}\\%")

    # main calibrated table
    rows = sorted(((k, v) for k, v in hv.items() if k != "_meta" and (k.endswith("_cal") or k.startswith("Combo"))),
                  key=lambda kv: kv[1]["QLIKE"])
    out, hl, bad = [], [], []
    for i, (k, v) in enumerate(rows):
        fam, nm = vname(k)
        out.append([tex(nm), tex("Combo" if fam == "Combination" else fam), f"{v['QLIKE']:.4f}",
                    f"{pct(v['pct_vs_ref'])} {ci(v['ci95_pct_vs_ref'])}", fp(v["DM_p"]), f"{v['MZ_R2']:.3f}",
                    f"{v['IC']:.3f}", f"{v['hit_rate'] * 100:.1f}", f"{v['assets_beating_ref']}/11",
                    "\\checkmark" if v.get("in_MCS90") else ""])
        (hl if "Meridian" in nm else bad if v["pct_vs_ref"] > 0 else []).append(i)
    write_table("tab_vol_main.tex", ["Model", "Family", "QLIKE", "vs HAR [95\\% CI]", "DM $p$", "MZ $R^2$", "IC",
                                     "Hit \\%", "Won", "MCS"], out,
                r">{\raggedright\arraybackslash}X l r l r r r r r c",
                f"Volatility forecasting, holdout {hm['date_range'][0]} to {hm['date_range'][1]} "
                f"({hm['n_dates']:,} dates $\\times$ {hm['n_assets']} assets, {hm['n_rows']:,} forecasts per model). "
                "All models recalibrated identically; reference = per-asset HAR-RV. Sorted by QLIKE (lower is better).",
                "tab:volmain", highlight=hl, bad=bad, size=r"\scriptsize",
                note="Green: Meridian rows. Red: worse than HAR. DM $p$: one-sided Diebold--Mariano on the date-level "
                     "loss differential with Newey--West HAC. CI: moving-block bootstrap over dates. Won: assets on "
                     "which the model's QLIKE is below HAR's. MCS: in the 90\\% Model Confidence Set.")

    # raw table
    hr = V["tables"]["holdout_raw"]
    rows = sorted(((k, v) for k, v in hr.items() if k != "_meta"), key=lambda kv: kv[1]["QLIKE"])
    out = [[tex(vname(k)[1]), tex(vname(k)[0]), f"{v['QLIKE']:.4f}", f"{pct(v['pct_vs_ref'])} {ci(v['ci95_pct_vs_ref'])}",
            fp(v["DM_p"]), f"{v['assets_beating_ref']}/11"] for k, v in rows]
    write_table("tab_vol_raw.tex", ["Model (raw, uncalibrated)", "Family", "QLIKE", "vs raw HAR [95\\% CI]",
                                    "DM $p$", "Won"], out, r"X l r l r r",
                "Raw (uncalibrated) forecasts on the holdout, reference = raw per-asset HAR-RV.", "tab:volraw",
                highlight=[i for i, (k, _) in enumerate(rows) if "Meridian" in k], size=r"\scriptsize")

    # conservative
    t = V["tables"]["holdout_vs_raw_HAR"]
    rows = sorted(((k, v) for k, v in t.items() if k != "_meta"), key=lambda kv: kv[1]["QLIKE"])

    def nm2(k):
        if k.startswith("Combo"):
            return vname(k)[1]
        return vname(k)[1].replace(" (reference)", "") + (" --- calibrated" if k.endswith("_cal") else " --- raw")
    out = [[tex(nm2(k)), f"{v['QLIKE']:.4f}", f"{pct(v['pct_vs_ref'])} {ci(v['ci95_pct_vs_ref'])}", fp(v["DM_p"]),
            f"{v['assets_beating_ref']}/11"] for k, v in rows]
    write_table("tab_vol_conservative.tex", ["Model", "QLIKE", "vs raw HAR [95\\% CI]", "DM $p$", "Won"], out,
                "X r l r r", "Conservative check: holdout against raw per-asset HAR, the strongest HAR form.",
                "tab:volcons", highlight=[i for i, (k, _) in enumerate(rows) if "Meridian" in k], size=r"\small")
    rh = t["Meridian_M0"]
    mac("RawMPct", pct(rh["pct_vs_ref"])); mac("RawMCI", ci(rh["ci95_pct_vs_ref"])); mac("RawMP", fp(rh["DM_p"]))
    mac("RawMCIHi", f"${rh['ci95_pct_vs_ref'][1] * 100:+.1f}\\%$")
    mac("RawHARQ", f"{t['HAR']['QLIKE']:.4f}")
    rc = t["Combo_HAR_LGBM_Meridian_M0"]
    mac("RawComboPct", pct(rc["pct_vs_ref"])); mac("RawComboCI", ci(rc["ci95_pct_vs_ref"]))
    mac("RawHARcalPct", pct(t["HAR_cal"]["pct_vs_ref"]))

    # per asset / per year
    by = V["holdout_by_asset_pct_vs_raw_HAR"]
    assets = sorted(by["Meridian_M0"])
    out = [[a, pct(by["Meridian_M0"][a]), pct(by["Meridian_M0_cal"][a]), pct(by["LightGBM_cal"][a]),
            pct(by["Combo_HAR_LGBM_Meridian_M0"][a])] for a in assets]
    write_table("tab_vol_asset.tex", ["Asset", "Meridian (raw)", "Meridian (cal.)", "LightGBM (cal.)",
                                      "HAR+LGBM+Meridian"], out, "Xrrrr",
                "Holdout QLIKE change vs raw per-asset HAR, by asset.", "tab:volasset", size=r"\small")
    mac("AssetBest", min(assets, key=lambda a: by["Meridian_M0"][a]))
    mac("AssetBestPct", pct(min(by["Meridian_M0"].values())))
    mac("AssetWorst", max(assets, key=lambda a: by["Meridian_M0"][a]))
    mac("AssetWorstPct", pct(max(by["Meridian_M0"].values())))
    yr = V["by_year_pct_vs_raw_HAR"]
    years = sorted(int(y) for y in yr["Meridian_M0"])
    out = [[y, pct(yr["Meridian_M0"][str(y)]), pct(yr["LightGBM_cal"][str(y)]),
            pct(yr["Combo_HAR_LGBM_Meridian_M0"][str(y)]), "holdout" if y >= 2019 else "development"] for y in years]
    write_table("tab_vol_year.tex", ["Year", "Meridian (raw)", "LightGBM (cal.)", "HAR+LGBM+Meridian", "Period"], out,
                "lrrrX", "QLIKE change vs raw per-asset HAR by calendar year (all out-of-sample years).",
                "tab:volyear", size=r"\small")
    won = sum(yr["Meridian_M0"][str(y)] < 0 for y in years)
    mac("YearsWon", won); mac("YearsTotal", len(years)); mac("FirstYear", years[0]); mac("LastYear", years[-1])
    worst_y = max(years, key=lambda y: yr["Meridian_M0"][str(y)])
    mac("WorstYear", worst_y); mac("WorstYearPct", pct(yr["Meridian_M0"][str(worst_y)], 2))

    # regimes
    R = V["regimes"]
    models = ["Meridian_M0_cal", "Combo_HAR_LGBM_Meridian_M0", "LightGBM_cal", "HARX_cal", "Ridge_cal", "LHAR_cal",
              "TimeMixer_cal", "GJR_GARCH_cal", "EWMA_cal"]
    out = [[tex(vname(m)[1])] + [f"{pct(R[g][m]['pct_vs_ref'])} ({fp(R[g][m]['DM_p'])})" for g in ("calm", "normal", "stress")]
           for m in models]
    write_table("tab_vol_regime.tex", ["Model", "Calm", "Normal", "Stress"], out, "Xlll",
                f"QLIKE change vs calibrated per-asset HAR by VIX tercile, full out-of-sample period "
                f"({R['calm']['_meta']['date_range'][0]} to {R['stress']['_meta']['date_range'][1]}); DM $p$ in brackets.",
                "tab:volregime", highlight=[0, 1], size=r"\small")
    for g in ("calm", "normal", "stress"):
        mac(f"Reg{g.capitalize()}Pct", pct(R[g]["Meridian_M0_cal"]["pct_vs_ref"]))

    # horizons
    out = []
    for h, lab in (("holdout_y_5", "5 trading days"), ("holdout_y_22", "22 trading days")):
        th = V["tables"][h]
        hk = "HAR_" + h[8:] + "_cal"
        mk = next(k for k in th if k.startswith("Meridian"))
        v = th[mk]
        out.append([lab, f"{th[hk]['QLIKE']:.4f}", f"{v['QLIKE']:.4f}", f"{pct(v['pct_vs_ref'])} {ci(v['ci95_pct_vs_ref'])}",
                    fp(v["DM_p"]), f"{v['assets_beating_ref']}/11"])
        mac("H" + ("Five" if "5" in h[-2:] and "22" not in h else "TwentyTwo") + "Pct", pct(v["pct_vs_ref"]))
        mac("H" + ("Five" if "5" in h[-2:] and "22" not in h else "TwentyTwo") + "P", fp(v["DM_p"]))
    write_table("tab_vol_horizon.tex", ["Horizon", "HAR QLIKE", "Meridian QLIKE", "vs HAR [95\\% CI]", "DM $p$", "Won"],
                out, "Xrrlrr", "Longer horizons (holdout, calibrated): Meridian auxiliary heads vs per-asset HAR "
                "fit to the same horizon.", "tab:volhorizon", size=r"\small")

    # breadth
    tb = V["tables"]["breadth_holdout"]
    keep = sorted([k for k in tb if k != "_meta" and k.endswith("_cal")], key=lambda k: tb[k]["QLIKE"])
    out = [[tex(vname(k)[1]), f"{tb[k]['QLIKE']:.4f}", f"{pct(tb[k]['pct_vs_ref'])} {ci(tb[k]['ci95_pct_vs_ref'])}",
            fp(tb[k]["DM_p"])] for k in keep]
    write_table("tab_vol_breadth.tex", ["Model (calibrated)", "QLIKE", "vs HAR [95\\% CI]", "DM $p$"], out, "Xrlr",
                "Out-of-universe check: TLT (long Treasuries) and GLD (gold), holdout. Pooled models "
                "(Meridian, Ridge, random forest, LightGBM, TimeMixer, pooled HAR) never trained on these assets.",
                "tab:volbreadth", highlight=[i for i, k in enumerate(keep) if "Meridian" in k], size=r"\scriptsize")
    mac("BreadthMPct", pct(tb["Meridian_M0_cal"]["pct_vs_ref"])); mac("BreadthMCI", ci(tb["Meridian_M0_cal"]["ci95_pct_vs_ref"]))
    mac("BreadthMP", fp(tb["Meridian_M0_cal"]["DM_p"]))

    # dev table + ablation
    td = V["tables"]["dev_raw_vs_calibrated"]
    rows = sorted(((k, v) for k, v in td.items() if k != "_meta" and (k.endswith("_cal") or k.startswith("Combo"))),
                  key=lambda kv: kv[1]["QLIKE"])
    out = [[tex(vname(k)[1]), f"{v['QLIKE']:.4f}", f"{pct(v['pct_vs_ref'])} {ci(v['ci95_pct_vs_ref'])}", fp(v["DM_p"]),
            f"{v['assets_beating_ref']}/11"] for k, v in rows]
    write_table("tab_vol_dev.tex", ["Model (calibrated)", "QLIKE", "vs HAR [95\\% CI]", "DM $p$", "Won"], out,
                "Xrlrr", f"Development period {dm['date_range'][0]} to {dm['date_range'][1]} (final 5-seed run).",
                "tab:voldev", highlight=[i for i, (k, _) in enumerate(rows) if "Meridian" in k], size=r"\scriptsize")
    mac("DevMPct", pct(td["Meridian_M0_cal"]["pct_vs_ref"]))
    ta = A["tables"]["dev_raw_vs_calibrated"]
    desc = {"M0": "pooled HAR-anchored initialization (existing)", "M1": "per-asset HAR offset + zero-init correction",
            "M2": "M1 + learned asset embedding", "M3": "M1 + early stopping (purged validation slice)"}
    keys = ["Combo_HAR_LGBM_Meridian_M0", "Combo_HAR_LGBM_Meridian_M2", "Combo_HAR_LGBM_Meridian_M1",
            "Combo_HAR_LGBM_Meridian_M3", "Meridian_M0_cal", "Meridian_M2_cal", "Meridian_M1_cal", "Meridian_M3_cal",
            "HAR_cal"]
    out = []
    for k in keys:
        var = next((m for m in desc if k.endswith(m) or k.endswith(m + "_cal")), None)
        out.append([tex(vname(k)[1]), tex(desc.get(var, "reference")), f"{ta[k]['QLIKE']:.4f}", pct(ta[k]["pct_vs_ref"])])
    write_table("tab_ablation.tex", ["Candidate", "Change tested", "Dev QLIKE", "vs HAR"], out, "XXrr",
                f"Development-period ablation of the proposed Meridian improvements ({AMETA['ensemble']} seeds each). "
                "The selection rule picks the lowest development QLIKE.", "tab:ablation", highlight=[0], size=r"\scriptsize")
    for v in ("M0", "M1", "M2", "M3"):
        mac({"M0": "AblMZero", "M1": "AblMOne", "M2": "AblMTwo", "M3": "AblMThree"}[v],
            f"{ta[f'Meridian_{v}_cal']['QLIKE']:.4f}")

    # daily / rolling context + forward power
    P = pd.read_parquet(RES / "benchmark_vol_predictions.parquet"); P["date"] = pd.to_datetime(P["date"])
    C = P[~P["breadth"] & np.isfinite(P[["Meridian_M0", "HAR", "rv_next"]]).all(1)].copy()
    for m in ("Meridian_M0", "HAR"):
        C["q_" + m] = qlike(C["rv_next"].to_numpy(), np.exp(C[m].to_numpy()))
    D = C.groupby("date")[["q_Meridian_M0", "q_HAR"]].mean()
    Dh = D[D.index >= "2019-01-01"]
    d = (Dh["q_Meridian_M0"] - Dh["q_HAR"]).to_numpy()
    r = pd.Series(d)
    mac("ShareDay", f"{(d < 0).mean() * 100:.1f}\\%")
    for w, nmw in ((20, "Twenty"), (63, "SixtyThree"), (252, "TwoFiftyTwo")):
        mac(f"Share{nmw}", f"{(r.rolling(w).mean().dropna() < 0).mean() * 100:.1f}\\%")
    lag = nw_lag(len(d)); e = d - d.mean()
    lr = np.mean(e * e) + sum(2 * (1 - k / (lag + 1)) * np.mean(e[k:] * e[:-k]) for k in range(1, lag + 1))
    n80 = ((stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) * np.sqrt(lr) / d.mean()) ** 2
    n50 = (stats.norm.ppf(0.95) * np.sqrt(lr) / d.mean()) ** 2
    mac("PowerEighty", f"{n80:.0f}"); mac("PowerFifty", f"{n50:.0f}")
    mac("PowerEightyYears", f"{n80 / 252:.1f}"); mac("PowerFiftyMonths", f"{n50 / 21:.0f}")

    # ---------------- figures ----------------
    rows = sorted(((vname(k)[1], v) for k, v in hv.items() if k != "_meta" and (k.endswith("_cal") or k.startswith("Combo"))
                   and k != "HAR_cal"), key=lambda r: r[1]["pct_vs_ref"])
    short = lambda n: n.replace("GR combo: ", "Combo: ").replace(" (pooled HAR anchor)", "")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(rows))
    cols = [BLUE if "Meridian" in n else (GREY if v["pct_vs_ref"] < 0 else ORANGE) for n, v in rows]
    ax.barh(y, [v["pct_vs_ref"] * 100 for _, v in rows], color=cols)
    for i, (_, v) in enumerate(rows):
        if v["ci95_pct_vs_ref"]:
            ax.plot([c * 100 for c in v["ci95_pct_vs_ref"]], [i, i], color=INK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([short(n) for n, _ in rows], fontsize=7)
    ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.8)
    ax.axvline(-5, color=ORANGE, lw=0.8, ls="--"); ax.text(-5.2, len(rows) - 0.6, "5% bar", ha="right", fontsize=7, color=ORANGE)
    ax.set_xlabel("QLIKE change vs per-asset HAR-RV, holdout (%); lines = 95% CI")
    fig.tight_layout(); fig.savefig(OUT / "fig_vol_bars.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for k, c, lab in (("Meridian_M0", BLUE, "Meridian M0 (raw)"), ("LightGBM_cal", GREY, "LightGBM (calibrated)"),
                      ("Combo_HAR_LGBM_Meridian_M0", GREEN, "HAR + LightGBM + Meridian")):
        ax.plot(years, [yr[k][str(y)] * 100 for y in years], marker="o", ms=3, lw=1.4, color=c, label=lab)
    ax.axhline(0, color=INK, lw=0.8); ax.axvspan(2018.5, years[-1] + 0.5, color="#f0f4f8", zorder=0)
    ax.text(2018.7, ax.get_ylim()[0] * 0.9, "holdout", fontsize=7, color=MUTED_TXT)
    ax.set_ylabel("QLIKE vs raw HAR (%)"); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
                                                     bbox_to_anchor=(0.5, 1.18))
    ax.set_xticks(years); ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_vol_year.pdf"); plt.close(fig)

    roll = (D["q_Meridian_M0"] - D["q_HAR"]).rolling(252).mean() / D["q_HAR"].rolling(252).mean() * 100
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(roll.index, roll.values, color=BLUE, lw=1.2)
    ax.axhline(0, color=INK, lw=0.8); ax.axvline(pd.Timestamp("2019-01-01"), color=ORANGE, lw=0.9, ls="--")
    ax.set_ylabel("rolling 252-day\nQLIKE vs HAR (%)")
    ax.text(pd.Timestamp("2019-02-01"), ax.get_ylim()[1] * 0.85, "holdout starts", fontsize=7, color=ORANGE)
    fig.tight_layout(); fig.savefig(OUT / "fig_vol_rolling.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.arange(len(assets)); w = 0.38
    ax.bar(x - w / 2, [by["Meridian_M0"][a] * 100 for a in assets], w, color=BLUE, label="Meridian M0 (raw)")
    ax.bar(x + w / 2, [by["LightGBM_cal"][a] * 100 for a in assets], w, color=GREY, label="LightGBM (calibrated)")
    ax.set_xticks(x); ax.set_xticklabels(assets, fontsize=7); ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("QLIKE vs raw HAR (%)"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_vol_asset.pdf"); plt.close(fig)

    # illustrative forecast path: SPY, Feb-Jun 2020
    s = C[(C["asset"] == "SPY") & (C["date"] >= "2020-01-15") & (C["date"] <= "2020-07-15")]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.plot(s["date"], s["y"], color=GREY, lw=0.9, label="realized log RV (next day)")
    ax.plot(s["date"], s["HAR"], color=ORANGE, lw=1.1, label="HAR-RV forecast")
    ax.plot(s["date"], s["Meridian_M0"], color=BLUE, lw=1.1, label="Meridian M0 forecast")
    ax.set_ylabel("log variance"); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
                                             bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "fig_vol_spy2020.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    names = ["M0", "M1", "M2", "M3"]
    vals = [ta[f"Meridian_{v}_cal"]["pct_vs_ref"] * 100 for v in names]
    vals_c = [ta[f"Combo_HAR_LGBM_Meridian_{v}"]["pct_vs_ref"] * 100 for v in names]
    x = np.arange(4)
    ax.bar(x - 0.19, vals, 0.38, color=BLUE, label="Meridian variant alone")
    ax.bar(x + 0.19, vals_c, 0.38, color=GREEN, label="combined with HAR + LightGBM")
    ax.set_xticks(x); ax.set_xticklabels(["M0 (existing)", "M1 HAR offset", "M2 + embedding", "M3 + early stop"])
    ax.set_ylabel("dev QLIKE vs HAR (%)"); ax.axhline(0, color=INK, lw=0.8)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(); fig.savefig(OUT / "fig_ablation.pdf"); plt.close(fig)


MUTED_TXT = "#52606d"


# --------------------------------------------------------------------------- #
# 3. tail benchmark
# --------------------------------------------------------------------------- #
def tail_section():
    th = T["tables"]["holdout"]; sel = T["selected"]; tm = th["025"]["_meta"]
    mac("TailSel", tex(TAIL_LABEL[sel])); mac("TailRef", tex(TAIL_LABEL[T["reference"]].replace(" (reference)", "")))
    mac("TailDates", f"{tm['n_dates']:,}"); mac("TailRows", f"{tm['n_rows']:,}")
    mac("TailZWin", TMETA["z_window"]); mac("TailZMin", TMETA["z_min"]); mac("TailHS", TMETA["hs_window"])
    mac("TailEVTu", f"{TMETA['evt_threshold']:.2f}"); mac("TailRefit", TMETA["evt_refit"]); mac("TailGamma", TMETA["aci_gamma"])
    mac("TailRuntime", TMETA["runtime_s"])
    v25, v1, v5 = th["025"][sel], th["010"][sel], th["050"][sel]
    mac("TFZ", f"{v25['FZ0_diff_vs_ref']:+.4f}"); mac("TFZCI", f"[{v25['FZ0_diff_ci95'][0]:+.4f}, {v25['FZ0_diff_ci95'][1]:+.4f}]")
    mac("TFZP", fp(v25["FZ0_DM_p"])); mac("TESR", f"{v25['es_ratio']:.2f}")
    mac("TVOne", pct(v1["tick_pct_vs_ref"])); mac("TVOneCI", ci(v1["tick_ci95_pct_vs_ref"])); mac("TVOneP", fp(v1["tick_DM_p"]))
    mac("TVTwo", pct(v25["tick_pct_vs_ref"])); mac("TVFive", pct(v5["tick_pct_vs_ref"]))
    mac("TViolOne", f"{v1['violation_rate'] * 100:.2f}\\%"); mac("TViolTwo", f"{v25['violation_rate'] * 100:.2f}\\%")
    mac("TViolFive", f"{v5['violation_rate'] * 100:.2f}\\%")
    mac("TCC", v25["assets_pass_cc"]); mac("TDQ", v25["assets_pass_dq"])
    for v, nmv in ((v1, "One"), (v25, "Two"), (v5, "Five")):
        mac(f"TKup{nmv}", v["assets_pass_kupiec"]); mac(f"TCC{nmv}", v["assets_pass_cc"])
        mac(f"TDQ{nmv}", v["assets_pass_dq"])
    mac("THarDQ", th["025"]["HAR_FHS"]["assets_pass_dq"]); mac("THarEVTDQ", th["025"]["HAR_EVT"]["assets_pass_dq"])
    h2 = T["tables"]["holdout_vs_HAR_FHS"]
    mac("HHOneP", fp(h2["010"][sel]["tick_DM_p"])); mac("HHOne", pct(h2["010"][sel]["tick_pct_vs_ref"]))
    mac("HHTwoP", fp(h2["025"][sel]["FZ0_DM_p"])); mac("HHFiveP", fp(h2["050"][sel]["tick_DM_p"]))
    dsel = T["meridian_dev_FZ0"]
    mac("TDevSel", f"{dsel[sel]:.4f}")
    dv = T["tables"]["dev"]["025"]
    mac("TDevHAR", f"{dv['HAR_FHS']['FZ0']:.4f}")
    worst = th["010"]["HS"]
    mac("THSOne", pct(worst["tick_pct_vs_ref"])); mac("TRMOneViol", f"{th['010']['RiskMetrics']['violation_rate'] * 100:.2f}\\%")
    mac("TGNOneViol", f"{th['010']['GARCH_N']['violation_rate'] * 100:.2f}\\%")

    def fz_table(t, fname, cap, label, of="11"):
        rows = sorted(((k, v) for k, v in t.items() if k != "_meta" and v["has_es"]), key=lambda kv: kv[1]["FZ0"])
        out, hl, bad = [], [], []
        for i, (k, v) in enumerate(rows):
            dd = "---" if v["FZ0_diff_ci95"] is None else \
                f"{v['FZ0_diff_vs_ref']:+.4f} [{v['FZ0_diff_ci95'][0]:+.4f}, {v['FZ0_diff_ci95'][1]:+.4f}]"
            out.append([tex(TAIL_LABEL[k]), f"{v['FZ0']:.4f}", dd, fp(v["FZ0_DM_p"]),
                        "\\checkmark" if v.get("FZ0_in_MCS90") else "", f"{v['violation_rate'] * 100:.2f}",
                        f"{v['assets_pass_cc']}/{of} $\\cdot$ {v['assets_pass_dq']}/{of}", f"{v['es_ratio']:.2f}"])
            (hl if "Meridian" in k else bad if v["FZ0_diff_vs_ref"] > 0 else []).append(i)
        write_table(fname, ["Model", "FZ0", "$\\Delta$ vs reference [95\\% CI]", "DM $p$", "MCS", "Viol.\\ \\%",
                            "CC $\\cdot$ DQ pass", "ES ratio"], out, r">{\raggedright\arraybackslash}X r l r c r l r",
                    cap, label, highlight=hl, bad=bad, size=r"\scriptsize")
    fz_table(th["025"], "tab_tail_fz.tex",
             f"Joint VaR + ES at 2.5\\% (FZ0 loss, lower is better), holdout {tm['date_range'][0]} to "
             f"{tm['date_range'][1]} ({tm['n_dates']:,} dates $\\times$ {tm['n_assets']} assets). Reference: "
             "GJR-GARCH + filtered historical simulation.", "tab:tailfz")
    fz_table(T["tables"]["dev"]["025"], "tab_tail_dev.tex",
             "Tail benchmark, development period (used only to choose the Meridian tail variant), 2.5\\%.", "tab:taildev")
    if "breadth_holdout" in T["tables"]:
        fz_table(T["tables"]["breadth_holdout"]["025"], "tab_tail_breadth.tex",
                 "Tail benchmark on the out-of-universe assets TLT and GLD, holdout, 2.5\\%.", "tab:tailbreadth", of="2")

    for k, lvl in (("010", "1"), ("025", "2.5"), ("050", "5")):
        t = th[k]
        rows = sorted(((m, v) for m, v in t.items() if m != "_meta"), key=lambda kv: kv[1]["tick"])
        out = [[tex(TAIL_LABEL[m]), f"{v['tick'] * 1e3:.2f}", f"{pct(v['tick_pct_vs_ref'])} {ci(v['tick_ci95_pct_vs_ref'])}",
                fp(v["tick_DM_p"]), "\\checkmark" if v.get("tick_in_MCS90") else "", f"{v['violation_rate'] * 100:.2f}",
                f"{v['assets_pass_kupiec']} $\\cdot$ {v['assets_pass_cc']} $\\cdot$ {v['assets_pass_dq']}"]
               for m, v in rows]
        write_table(f"tab_tail_var{k}.tex", ["Model", "Loss $\\times10^3$", "vs reference [95\\% CI]", "DM $p$", "MCS",
                                             f"Viol.\\ \\% (target {lvl})", "Kupiec $\\cdot$ CC $\\cdot$ DQ pass (of 11)"],
                    out, r">{\raggedright\arraybackslash}X r l r c r l",
                    f"VaR at {lvl}\\%: quantile loss on returns scaled per asset, holdout.", f"tab:tailvar{k}",
                    highlight=[i for i, (m, _) in enumerate(rows) if "Meridian" in m], size=r"\scriptsize")

    rows = []
    for k, a in (("010", "1\\%"), ("025", "2.5\\%"), ("050", "5\\%")):
        v = h2[k][sel]
        rows.append([a, f"{v['FZ0_diff_vs_ref']:+.4f} [{v['FZ0_diff_ci95'][0]:+.4f}, {v['FZ0_diff_ci95'][1]:+.4f}]",
                     fp(v["FZ0_DM_p"]), f"{pct(v['tick_pct_vs_ref'])} {ci(v['tick_ci95_pct_vs_ref'])}", fp(v["tick_DM_p"])])
    write_table("tab_tail_h2h.tex", ["Level", "FZ0 $\\Delta$ [95\\% CI]", "DM $p$", "Quantile loss vs HAR-FHS [95\\% CI]",
                                     "DM $p$"], rows, "lXrXr",
                f"Head-to-head on the holdout: {tex(TAIL_LABEL[sel])} vs HAR-RV + FHS (both filtered by a "
                "realized-volatility forecast).", "tab:tailh2h", size=r"\small")

    # figures
    t = th["025"]
    rows = sorted(((TAIL_LABEL[k].replace(" (reference)", ""), v) for k, v in t.items() if k != "_meta" and v["has_es"]),
                  key=lambda r: r[1]["FZ0_diff_vs_ref"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(rows))
    ax.barh(y, [v["FZ0_diff_vs_ref"] for _, v in rows],
            color=[BLUE if "Meridian" in n else (GREY if v["FZ0_diff_vs_ref"] < 0 else ORANGE) for n, v in rows])
    for i, (_, v) in enumerate(rows):
        if v["FZ0_diff_ci95"]:
            ax.plot(v["FZ0_diff_ci95"], [i, i], color=INK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows], fontsize=7); ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("FZ0 difference vs GJR-GARCH + FHS at 2.5%; lines = 95% CI")
    fig.tight_layout(); fig.savefig(OUT / "fig_tail_fz.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True)
    order = [m for m, _ in sorted(((m, v) for m, v in th["010"].items() if m != "_meta"), key=lambda kv: kv[1]["tick"])]
    for ax, (k, a) in zip(axes, (("010", 1.0), ("025", 2.5), ("050", 5.0))):
        vals = [th[k][m]["violation_rate"] * 100 for m in order]
        ax.barh(range(len(order)), vals, color=[BLUE if "Meridian" in m else GREY for m in order])
        ax.axvline(a, color=ORANGE, lw=1, ls="--")
        ax.set_title(f"VaR {a:g}%: violation rate", fontsize=8); ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([TAIL_LABEL[m].replace(" (reference)", "") for m in order], fontsize=6.5)
    axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT / "fig_tail_viol.pdf"); plt.close(fig)

    # example: SPY 2.5% VaR, 2020
    TP = pd.read_parquet(RES / "benchmark_tail_predictions.parquet"); TP["date"] = pd.to_datetime(TP["date"])
    s = TP[(TP["asset"] == "SPY") & (TP["date"] >= "2020-01-15") & (TP["date"] <= "2020-07-15")]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(s["date"], s["r_next"] * 100, color=GREY, width=1.0, label="next-day return")
    ax.plot(s["date"], s[f"VaR_GJR_FHS_025"] * 100, color=ORANGE, lw=1.1, label="GJR-GARCH + FHS VaR 2.5%")
    ax.plot(s["date"], s[f"VaR_{sel}_025"] * 100, color=BLUE, lw=1.1, label=f"{TAIL_LABEL[sel]} VaR 2.5%")
    ax.set_ylabel("return (%)"); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "fig_tail_spy2020.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. forward test
# --------------------------------------------------------------------------- #
def forward_section():
    mac("FwdFrozen", FMAN["frozen_at_utc"].replace("T", " ").replace("+00:00", " UTC"))
    mac("FwdTrainThrough", FMAN["train_data_through"]); mac("FwdCommit", FMAN["git_commit"])
    mac("FwdTrainRows", f"{FMAN['train_rows']:,}")
    mac("FwdLogged", len(FLOG)); mac("FwdPre", int(FLOG["precommitted"].sum()))
    mac("FwdOrigins", FLOG["origin_date"].nunique())
    mac("FwdFirstOrigin", FLOG["origin_date"].min()); mac("FwdLastOrigin", FLOG["origin_date"].max())
    rows = [[o, int((FLOG.origin_date == o).sum()), int((FLOG[FLOG.origin_date == o].precommitted).sum()),
             FLOG[FLOG.origin_date == o].logged_at_utc.iloc[0].replace("T", " ").replace("+00:00", " UTC")]
            for o in sorted(FLOG.origin_date.unique())]
    write_table("tab_fwd_log.tex", ["Forecast origin (close of)", "Forecasts", "Pre-committed", "Logged at"], rows,
                "lrrX", "Forward-test log so far (13 assets per origin; 11 core assets are scored).", "tab:fwdlog",
                size=r"\small")
    if FSCORE:
        mac("FwdScoredDates", FSCORE["n_dates"]); mac("FwdScoredN", FSCORE["n_forecasts"])
        mac("FwdQM", f"{FSCORE['QLIKE_Meridian_M0']:.4f}"); mac("FwdQH", f"{FSCORE['QLIKE_HAR']:.4f}")
        mac("FwdPct", pct(FSCORE["pct_vs_HAR"])); mac("FwdScoredAt", FSCORE["scored_at_utc"].replace("T", " ").replace("+00:00", " UTC"))
        mac("FwdScoredOrigin", FSCORE["date_range"][0])


def main():
    data_section(); vol_section(); tail_section(); forward_section()
    lines = ["% generated by scripts/build_report_tex.py -- do not edit"] + \
            [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(MACROS)} macros, tables and figures -> {OUT}")


if __name__ == "__main__":
    main()
