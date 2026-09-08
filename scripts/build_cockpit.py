"""Generate demo/cockpit.html — the Meridian World Model interpretable cockpit, from
results/world_state.json. Self-contained (data inlined), theme-aware, CSP-safe.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
st = json.loads((ROOT / "results" / "world_state.json").read_text())
sysd = st["system"]; assets = st["assets"]

RCLASS = {"Calm": "calm", "Transition": "trans", "Stress": "stress"}
asset_rows = "".join(
    f'<tr><td class="mono">{a}</td><td><span class="pill r-{RCLASS.get(v["regime"],"trans")}">{v["regime"]}</span></td>'
    f'<td class="mono num">{v["vol_forecast_ann_pct"]:.1f}%</td>'
    f'<td class="mono num neg">{v["var95_pct"]:.2f}%</td></tr>'
    for a, v in assets.items())

trans = "".join(f'<span class="node t">{x}</span>' for x in sysd["top_shock_transmitters"])
absb = "".join(f'<span class="node a">{x}</span>' for x in sysd["top_shock_absorbers"])
factors = "".join(f'<div class="fac"><div class="fbar" style="--v:{min(abs(f)*40,100)}%;--s:{"pos" if f>=0 else "neg"}"></div>'
                  f'<span class="mono">F{i+1} {f:+.2f}</span></div>' for i, f in enumerate(sysd["factor_state"]))
conn = sysd["systemic_connectedness_pct"]; surp = sysd["systemic_surprise_sigma"]
surp_cls = "stress" if surp > 1.5 else "trans" if surp > 0.5 else "calm"

HTML = f"""<style>
:root{{--bg:#f6f7fb;--panel:#fff;--ink:#141a26;--muted:#5f6b82;--line:#e3e8f2;--grid:#c9d1e0;
--accent:#0d9488;--calm:#0d9488;--trans:#c77d10;--stress:#e0616d;--pos:#0d9488;--negc:#e0616d;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#080b12;--panel:#111725;--ink:#dbe3f2;--muted:#7d8aa3;
--line:#1c2434;--grid:#222c40;--accent:#2dd4bf;--calm:#2dd4bf;--trans:#f0b429;--stress:#f0616d;--pos:#2dd4bf;--negc:#f0616d;}}}}
:root[data-theme=dark]{{--bg:#080b12;--panel:#111725;--ink:#dbe3f2;--muted:#7d8aa3;--line:#1c2434;--grid:#222c40;--accent:#2dd4bf;--calm:#2dd4bf;--trans:#f0b429;--stress:#f0616d;--pos:#2dd4bf;--negc:#f0616d;}}
:root[data-theme=light]{{--bg:#f6f7fb;--panel:#fff;--ink:#141a26;--muted:#5f6b82;--line:#e3e8f2;--grid:#c9d1e0;--accent:#0d9488;--calm:#0d9488;--trans:#c77d10;--stress:#e0616d;--pos:#0d9488;--negc:#e0616d;}}
*{{box-sizing:border-box}}.mono{{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}}
.wrap{{max-width:1000px;margin:0 auto;padding:26px 20px 40px;font-family:system-ui,sans-serif;color:var(--ink);background:var(--bg);min-height:100%}}
.top{{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);padding-bottom:14px;flex-wrap:wrap}}
.mark{{color:var(--accent);font-size:20px}}.name{{font-weight:700;letter-spacing:.22em;font-size:15px}}
.tag{{color:var(--muted);font-size:11px}}.asof{{margin-left:auto;color:var(--muted);font-size:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:16px}}
.card h3{{margin:0 0 12px;font-size:11px;letter-spacing:.13em;color:var(--muted);text-transform:uppercase}}
.gauge{{height:12px;border-radius:6px;background:linear-gradient(90deg,var(--calm),var(--trans),var(--stress));position:relative;margin:6px 0 4px}}
.gauge::after{{content:"";position:absolute;top:-3px;left:calc({conn}% - 2px);width:4px;height:18px;background:var(--ink);border-radius:2px}}
.big{{font-size:30px;font-weight:700;line-height:1}}.unit{{font-size:14px;color:var(--muted);font-weight:600}}
.sub{{font-size:11.5px;color:var(--muted);margin-top:3px}}
.flow{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:6px}}
.node{{padding:5px 11px;border-radius:8px;font-weight:700;font-size:12px;font-family:ui-monospace,monospace}}
.node.t{{background:rgba(224,97,109,.14);color:var(--stress);border:1px solid var(--stress)}}
.node.a{{background:rgba(13,148,136,.12);color:var(--calm);border:1px solid var(--calm)}}
.arrow{{color:var(--muted);font-size:18px}}
.fac{{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px}}
.fbar{{height:8px;width:var(--v);border-radius:4px;background:var(--accent);min-width:4px}}
.fbar[style*="neg"]{{background:var(--trans)}}
.surp .big{{color:var(--{surp_cls})}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:var(--muted);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;padding:6px 8px;border-bottom:1px solid var(--line)}}
td{{padding:7px 8px;border-bottom:1px solid var(--line)}}.num{{text-align:right}}.neg{{color:var(--negc)}}
.pill{{padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700}}
.r-calm{{color:var(--calm);background:rgba(13,148,136,.12)}}.r-trans{{color:var(--trans);background:rgba(199,125,16,.12)}}.r-stress{{color:var(--stress);background:rgba(224,97,109,.14)}}
.foot{{margin-top:16px;text-align:center;color:var(--muted);font-size:11px}}
@media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
</style>
<div class="wrap">
<div class="top"><span class="mark">◆</span><div><div class="name">MERIDIAN WORLD MODEL</div>
<div class="tag">modular · interpretable · continually-learning</div></div>
<div class="asof">state as of <b class="mono">{sysd["as_of"]}</b></div></div>

<div class="grid">
  <div class="card"><h3>Systemic connectedness (contagion)</h3>
    <div class="big mono">{conn}<span class="unit">%</span></div>
    <div class="gauge"></div>
    <div class="sub">share of cross-asset uncertainty from spillovers · higher = more systemic (Diebold-Yilmaz)</div></div>
  <div class="card surp"><h3>Systemic surprise (DFG energy)</h3>
    <div class="big mono">{surp:+.1f}<span class="unit">σ</span></div>
    <div class="sub">structure-break signal vs 1-yr norm · spikes when the factor model fails (caught the 2019 repo crisis)</div></div>
  <div class="card"><h3>Shock propagation</h3>
    <div class="flow">{trans}<span class="arrow">→</span>{absb}</div>
    <div class="sub">net transmitters (sources) cascade to net absorbers over a 10-day horizon</div></div>
  <div class="card"><h3>Latent market factor-state (DFG)</h3>
    {factors}
    <div class="sub">3 interpretable common factors reconstruct the cross-asset vol panel (R²≈0.70)</div></div>
</div>

<div class="card"><h3>Asset state — regime · volatility forecast · 1-day 95% VaR</h3>
<table><thead><tr><th>Asset</th><th>Regime</th><th class="num">Vol (ann)</th><th class="num">VaR 95%</th></tr></thead>
<tbody>{asset_rows}</tbody></table></div>

<div class="foot mono">MERIDIAN · vol⊕RF ensemble (+9% vs HAR) · switching regime · Student-t VaR · connectedness · DFG factor-state · online-adaptive · research demo, not investment advice</div>
</div>"""

out = ROOT / "demo" / "cockpit.html"
out.write_text(HTML)
print("wrote", out, len(HTML), "bytes")
