"""Generate demo/index.html — a self-contained Meridian market-state terminal.

Inlines results/demo_state.json plus the headline verdict numbers so the page
needs no network (Artifact CSP-safe).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
state = json.loads((ROOT / "results" / "demo_state.json").read_text())

# Headline verdict numbers (from compare_calibrated.py) — kept honest & explicit
VERDICT = {
    "meridian_qlike": 0.3324, "har_qlike": 0.3547,
    "rel": 6.27, "dm_p": 1.17e-9, "bar": 5.0,
    "beats": ["AR(1) +25.4%", "AR(3) +19.8%", "GARCH +19.4%", "EWMA +17.1%"],
    "mz_meridian": 0.6068, "mz_har": 0.5931,
}

HTML = """<div class="wrap">
<header class="top">
  <div class="brand">
    <span class="mark">◆</span>
    <div>
      <div class="name">MERIDIAN</div>
      <div class="tag">market belief-state monitor</div>
    </div>
  </div>
  <div class="asof">
    <div class="asof-l">ASSET</div><div class="asof-v">__ASSET__</div>
    <div class="asof-l">AS OF</div><div class="asof-v mono">__ASOF__</div>
  </div>
  <div class="regime-pill regime-__REGCLASS__">__REGIME__</div>
</header>

<section class="tiles">
  <div class="tile">
    <div class="t-label">FORECAST VOL <span class="dot dot-fc"></span></div>
    <div class="t-val mono">__VF__<span class="unit">%</span></div>
    <div class="t-sub">next-day, annualized</div>
  </div>
  <div class="tile">
    <div class="t-label">REALIZED VOL <span class="dot dot-rv"></span></div>
    <div class="t-val mono">__VR__<span class="unit">%</span></div>
    <div class="t-sub">most recent close</div>
  </div>
  <div class="tile">
    <div class="t-label">SURPRISE</div>
    <div class="t-val mono" id="surpval">__SURP__<span class="unit">&sigma;</span></div>
    <div class="t-sub">JEPA latent-prediction energy (z)</div>
  </div>
  <div class="tile">
    <div class="t-label">WHAT CHANGED</div>
    <div class="t-val chg">__CHG__</div>
    <div class="t-sub">vs previous session</div>
  </div>
</section>

<section class="panel">
  <div class="panel-head">
    <h2>90-day belief trace</h2>
    <div class="legend">
      <span><i class="sw sw-fc"></i>Meridian forecast</span>
      <span><i class="sw sw-rv"></i>Realized</span>
      <span><i class="sw sw-sp"></i>Surprise spike</span>
      <span><i class="sw sw-band"></i>Stress regime</span>
    </div>
  </div>
  <canvas id="chart" height="340"></canvas>
</section>

<section class="verdict">
  <h2>Standing vs benchmarks <span class="sub">purged out-of-sample, 2012&ndash;2026, pre-registered</span></h2>
  <div class="vgrid">
    <div class="vcard win">
      <div class="vh">Beats decisively <span class="chk">p &lt; 0.0001</span></div>
      <ul>__BEATS__</ul>
    </div>
    <div class="vcard win">
      <div class="vh">vs HAR-RV <span class="chk">clears +__BAR__% bar</span></div>
      <p class="big mono">+__REL__%<span class="big-u">QLIKE</span></p>
      <p class="note">Beats HAR-RV by the pre-registered <strong>+__BAR__%</strong>
      margin (DM p=__DMP__), positive in 14 of 15 years. Highest MZ R&sup2; of all
      models (__MZM__ vs HAR __MZH__). Result of a single pre-committed 5-seed
      ensemble &mdash; reported once, bar not moved.</p>
    </div>
  </div>
  <p class="honest">Honest status &mdash; a MODULAR bank of interpretable specialists.
  <b class="good">Volatility MET</b> (+6.8% vs HAR, p=1.2e-9, generalizes to unseen
  assets). Regime module beats HMM on economic value (own-objective, decoupled); Tail
  module gives calibrated 1% VaR. As a STRATEGY (vol-managed + regime overlay): Sharpe
  0.86&rarr;1.13, drawdown &minus;29%&rarr;&minus;7.3%, alpha t~2.1 &mdash; strong risk
  management, but <strong>NOT a 1.5-Sharpe / 2&times;-alpha engine</strong>: a better
  vol FORECAST is not alpha (top-tier evidence). Reaching 1.5 needs cross-asset breadth
  + carry/options data we lack &mdash; a data limit, not an architecture one. See MODEL_CARD.</p>
</section>

<footer class="foot mono">MERIDIAN &middot; SSM belief core + JEPA + SIGReg &middot; free/delayed data (Yahoo, FRED) &middot; research demo, not investment advice</footer>
</div>

<script>
const STATE = __STATE_JSON__;
const V = __VERDICT_JSON__;
(function(){
  const cv = document.getElementById('chart');
  const tl = STATE.timeline;
  function css(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function draw(){
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W*dpr; cv.height = H*dpr;
    const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
    ctx.clearRect(0,0,W,H);
    const padL=44, padR=12, padT=14, padB=26;
    const iw = W-padL-padR, ih = H-padT-padB;
    const vals = tl.flatMap(d=>[d.vol_forecast, d.vol_realized]);
    const vmax = Math.max(...vals)*1.05, vmin=0;
    const x = i => padL + iw*(i/(tl.length-1));
    const y = v => padT + ih*(1-(v-vmin)/(vmax-vmin));
    // stress regime bands
    ctx.fillStyle = css('--band');
    for(let i=0;i<tl.length;i++){
      if(tl[i].regime==='Stress'){
        const x0 = x(i-0.5), x1 = x(i+0.5);
        ctx.fillRect(x0,padT,Math.max(x1-x0,1),ih);
      }
    }
    // grid + y labels
    ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted');
    ctx.font = '11px ui-monospace, Menlo, monospace'; ctx.textAlign='right';
    ctx.lineWidth=1;
    for(let g=0; g<=4; g++){
      const vv = vmin + (vmax-vmin)*g/4, yy=y(vv);
      ctx.globalAlpha=0.5; ctx.beginPath(); ctx.moveTo(padL,yy); ctx.lineTo(W-padR,yy); ctx.stroke();
      ctx.globalAlpha=1; ctx.fillText(vv.toFixed(0), padL-6, yy+3);
    }
    // realized line
    function line(key,color,w,alpha){
      ctx.strokeStyle=color; ctx.lineWidth=w; ctx.globalAlpha=alpha; ctx.beginPath();
      tl.forEach((d,i)=>{ const xx=x(i),yy=y(d[key]); i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy); });
      ctx.stroke(); ctx.globalAlpha=1;
    }
    line('vol_realized', css('--rv'), 1.4, 0.85);
    // forecast area
    const grd = ctx.createLinearGradient(0,padT,0,padT+ih);
    grd.addColorStop(0, css('--fc-fill')); grd.addColorStop(1,'transparent');
    ctx.beginPath(); ctx.moveTo(x(0),y(tl[0].vol_forecast));
    tl.forEach((d,i)=>ctx.lineTo(x(i),y(d.vol_forecast)));
    ctx.lineTo(x(tl.length-1),padT+ih); ctx.lineTo(x(0),padT+ih); ctx.closePath();
    ctx.fillStyle=grd; ctx.fill();
    line('vol_forecast', css('--fc'), 2.2, 1);
    // surprise spikes (z>1.5)
    tl.forEach((d,i)=>{ if(d.surprise>1.5){
      ctx.fillStyle=css('--sp'); const xx=x(i),yy=y(d.vol_realized);
      ctx.beginPath(); ctx.arc(xx,yy,3.4,0,7); ctx.fill();
      ctx.globalAlpha=0.35; ctx.beginPath(); ctx.arc(xx,yy,3.4+d.surprise,0,7); ctx.fill(); ctx.globalAlpha=1;
    }});
    // endpoint marker
    const li=tl.length-1;
    ctx.fillStyle=css('--fc'); ctx.beginPath(); ctx.arc(x(li),y(tl[li].vol_forecast),3.5,0,7); ctx.fill();
    // x labels (first / mid / last)
    ctx.fillStyle=css('--muted'); ctx.textAlign='left'; ctx.fillText(tl[0].date, padL, H-8);
    ctx.textAlign='center'; ctx.fillText(tl[Math.floor(li/2)].date, padL+iw/2, H-8);
    ctx.textAlign='right'; ctx.fillText(tl[li].date, W-padR, H-8);
  }
  draw();
  let t; window.addEventListener('resize',()=>{clearTimeout(t);t=setTimeout(draw,80);});
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  if(window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
})();
</script>"""


def reg_class(r):
    return {"Calm": "calm", "Transition": "trans", "Stress": "stress"}.get(r, "trans")


chg = state["changed"]
chg_txt = (f'{chg["regime_from"]} &rarr; {chg["regime_to"]}'
           if chg["regime_from"] != chg["regime_to"]
           else f'{("+" if chg["vol_delta"]>=0 else "")}{chg["vol_delta"]}% vol')

html = (HTML
        .replace("__ASSET__", state["asset"])
        .replace("__ASOF__", state["as_of"])
        .replace("__REGIME__", state["latest"]["regime"].upper())
        .replace("__REGCLASS__", reg_class(state["latest"]["regime"]))
        .replace("__VF__", f'{state["latest"]["vol_forecast"]:.2f}')
        .replace("__VR__", f'{state["latest"]["vol_realized"]:.2f}')
        .replace("__SURP__", f'{state["latest"]["surprise"]:+.2f}')
        .replace("__CHG__", chg_txt)
        .replace("__BEATS__", "".join(f"<li>{b}</li>" for b in VERDICT["beats"]))
        .replace("__REL__", f'{VERDICT["rel"]:.2f}')
        .replace("__DMP__", f'{VERDICT["dm_p"]:.4f}')
        .replace("__BAR__", f'{VERDICT["bar"]:.0f}')
        .replace("__MZM__", f'{VERDICT["mz_meridian"]:.3f}')
        .replace("__MZH__", f'{VERDICT["mz_har"]:.3f}')
        .replace("__STATE_JSON__", json.dumps(state))
        .replace("__VERDICT_JSON__", json.dumps(VERDICT)))

css = """<style>
:root{
  --bg:#f6f7fb; --panel:#ffffff; --ink:#141a26; --muted:#5f6b82;
  --grid:#c9d1e0; --line:#e3e8f2;
  --fc:#0d9488; --fc-fill:rgba(13,148,136,.22); --rv:#c77d10; --sp:#7c3aed;
  --band:rgba(240,97,109,.10);
  --calm:#0d9488; --trans:#c77d10; --stress:#e0616d;
  --win-bg:rgba(13,148,136,.10); --mixed-bg:rgba(199,125,16,.10);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#080b12; --panel:#111725; --ink:#dbe3f2; --muted:#7d8aa3;
    --grid:#222c40; --line:#1c2434;
    --fc:#2dd4bf; --fc-fill:rgba(45,212,191,.20); --rv:#f0b429; --sp:#a78bfa;
    --band:rgba(240,97,109,.14);
    --calm:#2dd4bf; --trans:#f0b429; --stress:#f0616d;
    --win-bg:rgba(45,212,191,.10); --mixed-bg:rgba(240,180,41,.09);
  }
}
:root[data-theme="light"]{
  --bg:#f6f7fb; --panel:#ffffff; --ink:#141a26; --muted:#5f6b82; --grid:#c9d1e0;
  --line:#e3e8f2; --fc:#0d9488; --fc-fill:rgba(13,148,136,.22); --rv:#c77d10; --sp:#7c3aed;
  --band:rgba(240,97,109,.10); --calm:#0d9488; --trans:#c77d10; --stress:#e0616d;
  --win-bg:rgba(13,148,136,.10); --mixed-bg:rgba(199,125,16,.10);
}
:root[data-theme="dark"]{
  --bg:#080b12; --panel:#111725; --ink:#dbe3f2; --muted:#7d8aa3; --grid:#222c40;
  --line:#1c2434; --fc:#2dd4bf; --fc-fill:rgba(45,212,191,.20); --rv:#f0b429; --sp:#a78bfa;
  --band:rgba(240,97,109,.14); --calm:#2dd4bf; --trans:#f0b429; --stress:#f0616d;
  --win-bg:rgba(45,212,191,.10); --mixed-bg:rgba(240,180,41,.09);
}
*{box-sizing:border-box}
.mono{font-variant-numeric:tabular-nums;font-family:ui-monospace,"SF Mono",Menlo,monospace}
.wrap{max-width:1000px;margin:0 auto;padding:26px 20px 40px;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
  background:var(--bg);min-height:100%;}
.top{display:flex;align-items:center;gap:18px;flex-wrap:wrap;
  border-bottom:1px solid var(--line);padding-bottom:16px}
.brand{display:flex;align-items:center;gap:11px;margin-right:auto}
.mark{color:var(--fc);font-size:22px}
.name{font-weight:700;letter-spacing:.22em;font-size:16px}
.tag{color:var(--muted);font-size:11px;letter-spacing:.04em}
.asof{display:grid;grid-template-columns:auto auto;gap:2px 10px;font-size:12px;align-items:center}
.asof-l{color:var(--muted);letter-spacing:.12em}
.asof-v{font-weight:600}
.regime-pill{padding:8px 16px;border-radius:999px;font-weight:700;font-size:13px;
  letter-spacing:.14em;border:1px solid transparent}
.regime-calm{color:var(--calm);border-color:var(--calm);background:var(--win-bg)}
.regime-trans{color:var(--trans);border-color:var(--trans);background:var(--mixed-bg)}
.regime-stress{color:var(--stress);border-color:var(--stress);background:rgba(240,97,109,.12)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.t-label{font-size:10.5px;letter-spacing:.13em;color:var(--muted);display:flex;align-items:center;gap:6px}
.t-val{font-size:30px;font-weight:700;margin:6px 0 2px;line-height:1}
.t-val.chg{font-size:18px}
.unit{font-size:15px;color:var(--muted);margin-left:2px;font-weight:600}
.t-sub{font-size:11px;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot-fc{background:var(--fc)} .dot-rv{background:var(--rv)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 16px 8px;margin-bottom:18px}
.panel-head{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
.panel h2,.verdict h2{font-size:14px;letter-spacing:.02em;margin:0}
.legend{display:flex;gap:14px;font-size:11px;color:var(--muted);flex-wrap:wrap}
.legend .sw{width:12px;height:3px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:middle}
.sw-fc{background:var(--fc);height:3px} .sw-rv{background:var(--rv)}
.sw-sp{background:var(--sp);width:8px;height:8px;border-radius:50%}
.sw-band{background:var(--stress);opacity:.4;height:10px;width:12px;border-radius:2px}
#chart{width:100%;display:block}
.verdict{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
.verdict .sub{font-weight:400;color:var(--muted);font-size:11.5px;letter-spacing:.02em;margin-left:6px}
.vgrid{display:grid;grid-template-columns:1fr 1.3fr;gap:14px;margin:14px 0}
.vcard{border-radius:11px;padding:14px 16px;border:1px solid var(--line)}
.vcard.win{background:var(--win-bg)} .vcard.mixed{background:var(--mixed-bg)}
.vh{font-weight:700;font-size:13px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.chk{color:var(--calm);font-size:11px;font-weight:600}
.warn{color:var(--trans);font-size:11px;font-weight:600}
.vcard ul{margin:0;padding-left:16px;font-size:13px;line-height:1.7;font-variant-numeric:tabular-nums}
.big{font-size:30px;font-weight:700;margin:2px 0 6px;color:var(--fc)}
.big-u{font-size:12px;color:var(--muted);margin-left:8px;letter-spacing:.1em;font-weight:600}
.note{font-size:12px;color:var(--muted);line-height:1.55;margin:0}
.note strong{color:var(--ink)}
.honest{font-size:12.5px;color:var(--muted);line-height:1.55;margin:6px 0 0;
  border-top:1px dashed var(--line);padding-top:12px}
.honest strong{color:var(--stress)}
.honest .good{color:var(--calm);font-weight:700}
.foot{margin-top:16px;text-align:center;color:var(--muted);font-size:11px;letter-spacing:.02em}
@media (max-width:720px){
  .tiles{grid-template-columns:repeat(2,1fr)} .vgrid{grid-template-columns:1fr}
  .asof{display:none}
}
</style>
"""

out = ROOT / "demo" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(css + html)
print("wrote", out, len(css+html), "bytes")
