#!/usr/bin/env python3
"""Render scores/<name>.confusion.json into a standalone HTML page.

  python3 build_confusion_page.py --name best_pipeline --out /tmp/confusion.html

Embeds the matrix as JSON and draws it client-side; no external assets.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

TEMPLATE = r"""<title>Where the Ensemble Goes Wrong</title>
<style>
:root{
  --ground:#eaeff1; --panel:#fff; --panel-2:#f4f7f8;
  --ink:#11242c; --ink-dim:#5d727b; --ink-faint:#8fa3ab;
  --rule:#d2dcdf; --rule-soft:#e4ebed;
  --accent:#0f6d6c; --hit:#1a7f7b; --miss:#ab4636; --fp:#b07d31; --mix:#5555a0;
  --shadow:0 1px 2px rgba(17,36,44,.06),0 8px 24px -12px rgba(17,36,44,.18);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0b1316; --panel:#121e22; --panel-2:#16252a;
    --ink:#dde8e9; --ink-dim:#8ba1a7; --ink-faint:#63797f;
    --rule:#223238; --rule-soft:#1a292e;
    --accent:#4fb0ab; --hit:#43a6a0; --miss:#d67c69; --fp:#d3a45f; --mix:#9090d8;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
  }
}
:root[data-theme="dark"]{
  --ground:#0b1316; --panel:#121e22; --panel-2:#16252a;
  --ink:#dde8e9; --ink-dim:#8ba1a7; --ink-faint:#63797f;
  --rule:#223238; --rule-soft:#1a292e;
  --accent:#4fb0ab; --hit:#43a6a0; --miss:#d67c69; --fp:#d3a45f; --mix:#9090d8;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  margin:0; padding:40px 24px 96px;
}
.wrap{max-width:1120px;margin:0 auto;display:flex;flex-direction:column;gap:44px}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
.num{font-variant-numeric:tabular-nums}

/* ---- masthead ---- */
header{display:flex;flex-direction:column;gap:14px;border-bottom:2px solid var(--ink);padding-bottom:20px}
.eyebrow{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);
  display:flex;flex-wrap:wrap;gap:6px 14px;
}
h1{
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:clamp(26px,4.2vw,40px);font-weight:600;letter-spacing:-.02em;
  margin:0;text-wrap:balance;line-height:1.12;
}
.sub{color:var(--ink-dim);max-width:66ch;margin:0}
.sub b{color:var(--ink);font-weight:600}

/* ---- section chrome ---- */
section{display:flex;flex-direction:column;gap:16px}
h2{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--ink-dim);font-weight:600;margin:0;
  display:flex;align-items:center;gap:12px;
}
h2::after{content:"";flex:1;height:1px;background:var(--rule)}
.note{color:var(--ink-dim);font-size:14px;max-width:70ch;margin:0}
.note b{color:var(--ink);font-weight:600}

/* ---- error budget ---- */
.budget{display:flex;flex-direction:column;gap:12px}
.bar{display:flex;height:52px;border-radius:3px;overflow:hidden;box-shadow:var(--shadow)}
.seg{
  display:flex;align-items:center;justify-content:center;
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;font-weight:600;
  color:#fff;font-variant-numeric:tabular-nums;overflow:hidden;white-space:nowrap;
}
.keys{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.key{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:13px 15px;
  display:flex;flex-direction:column;gap:3px;border-left-width:4px;border-left-style:solid}
.key .k-n{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:23px;font-weight:600;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.key .k-l{font-size:12.5px;font-weight:600}
.key .k-d{font-size:12px;color:var(--ink-dim);line-height:1.45}

/* ---- matrix ---- */
.mx-tools{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px}
.readout{
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  padding:8px 12px;min-height:35px;display:flex;align-items:center;flex:1;min-width:280px;
  color:var(--ink-dim);
}
.readout b{color:var(--ink)}
.scale{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--ink-dim)}
.scale i{display:block;width:15px;height:15px;border-radius:2px;font-style:normal}
.mx-scroll{overflow:auto;max-height:78vh;background:var(--panel);
  border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow)}
table.mx{border-collapse:separate;border-spacing:0;font-variant-numeric:tabular-nums}
table.mx th,table.mx td{padding:0;font-weight:400}
table.mx thead th{
  position:sticky;top:0;z-index:3;background:var(--panel-2);
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;color:var(--ink-dim);
  height:30px;width:25px;min-width:25px;border-bottom:1px solid var(--rule);
}
table.mx thead th.corner{left:0;z-index:5;width:236px;min-width:236px;text-align:left;
  padding-left:11px;letter-spacing:.1em;text-transform:uppercase;font-size:9.5px}
table.mx tbody th{
  position:sticky;left:0;z-index:2;background:var(--panel);
  text-align:left;padding:0 9px 0 11px;font-size:11.5px;white-space:nowrap;
  width:236px;min-width:236px;height:25px;border-right:1px solid var(--rule);
  overflow:hidden;text-overflow:ellipsis;
}
table.mx tbody th .id{color:var(--ink-faint);font-family:ui-monospace,Menlo,Consolas,monospace;
  font-size:10.5px;display:inline-block;width:19px}
table.mx td{width:25px;min-width:25px;height:25px;text-align:center;font-size:10.5px;
  font-family:ui-monospace,Menlo,Consolas,monospace;
  border-right:1px solid var(--rule-soft);border-bottom:1px solid var(--rule-soft);
  color:var(--ink);cursor:default}
table.mx tr.bgrow th,table.mx tr.bgrow td{border-top:2px solid var(--rule)}
table.mx td.bgcol,table.mx thead th.bgcol{border-left:2px solid var(--rule)}
table.mx td:hover{outline:2px solid var(--accent);outline-offset:-2px}
.thin{color:var(--miss);font-weight:700}

/* ---- data tables ---- */
.tbl-scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);
  border-radius:3px;box-shadow:var(--shadow)}
table.data{border-collapse:collapse;width:100%;font-size:13.5px}
table.data th{
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-dim);font-weight:600;text-align:left;
  padding:11px 14px;border-bottom:1px solid var(--rule);white-space:nowrap;background:var(--panel-2);
}
table.data td{padding:8px 14px;border-bottom:1px solid var(--rule-soft);white-space:nowrap}
table.data tr:last-child td{border-bottom:none}
table.data td.n{text-align:right;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Menlo,Consolas,monospace}
.pill{display:inline-block;padding:1px 7px;border-radius:2px;font-size:11px;font-weight:600;
  font-family:ui-monospace,Menlo,Consolas,monospace}
.pill.miss{background:color-mix(in srgb,var(--miss) 17%,transparent);color:var(--miss)}
.pill.fp{background:color-mix(in srgb,var(--fp) 20%,transparent);color:var(--fp)}
.pill.mix{background:color-mix(in srgb,var(--mix) 17%,transparent);color:var(--mix)}
.meter{position:relative;height:15px;width:104px;background:var(--rule-soft);border-radius:2px;
  overflow:hidden;display:inline-block;vertical-align:middle}
.meter i{position:absolute;left:0;top:0;bottom:0;display:block;font-style:normal}
footer{border-top:1px solid var(--rule);padding-top:18px;color:var(--ink-faint);font-size:12px;
  line-height:1.7}
</style>

<div class="wrap">
<header>
  <div class="eyebrow"><span>A + B + E + F</span><span>&#183;</span><span>WBF &rarr; rescorer</span>
    <span>&#183;</span><span>val 1,661 images</span><span>&#183;</span>
    <span>conf &ge; __CONF__</span><span>&#183;</span><span>IoU &ge; __IOU__</span></div>
  <h1>Where the Ensemble Goes Wrong</h1>
  <p class="sub">Of <b>__ERRTOT__ errors</b> on the validation split, only <b>__MIX__ (__MIXPCT__%)</b>
  are the model naming an object it found. The rest are objects it never found and boxes it
  invented. This is a <b>recall problem, not a taxonomy problem</b> &mdash; which means better
  classification, including the crop rescorer, has almost nothing left to fix.</p>
</header>

<section>
  <h2>Error budget</h2>
  <div class="budget">
    <div class="bar" id="bar"></div>
    <div class="keys" id="keys"></div>
  </div>
  <p class="note">Read the bar as the whole job: <b>__HIT__</b> of <b>__GTTOT__</b> ground-truth
  boxes are correctly found and named. Every intervention we have measured &mdash; TTA,
  Copy-Paste, the rescorer's &alpha; sweep &mdash; moves the small purple slice.</p>
</section>

<section>
  <h2>Confusion matrix &#183; 34 classes + background</h2>
  <div class="mx-tools">
    <div class="readout" id="readout">Hover a cell to read it.</div>
    <div class="scale">
      <i style="background:var(--hit)"></i><span>correct</span>
      <i style="background:var(--mix)"></i><span>wrong name</span>
      <i style="background:var(--miss)"></i><span>missed</span>
      <i style="background:var(--fp)"></i><span>false positive</span>
    </div>
  </div>
  <div class="mx-scroll"><table class="mx" id="mx"></table></div>
  <p class="note">Rows are what the pipeline <b>said</b>; columns are what was <b>actually
  there</b>. Colour intensity is logarithmic &mdash; the diagonal would otherwise drown
  everything else. Class ids in <span class="thin">red</span> have &le;10 validation boxes,
  so their entire row and column are anecdote, not measurement.</p>
</section>

<section>
  <h2>Genuine class confusions</h2>
  <p class="note">Background excluded &mdash; these are boxes localized correctly and labelled
  wrong. The list is short and it is dominated by one pair.</p>
  <div class="tbl-scroll"><table class="data" id="pairs"></table></div>
</section>

<section>
  <h2>Per-class precision and recall</h2>
  <p class="note">Sorted by validation support. Everything above the rule has enough boxes to
  believe; everything below it does not.</p>
  <div class="tbl-scroll"><table class="data" id="perclass"></table></div>
</section>

<footer>
  Generated by <span class="mono">12_ensemble/confusion.py</span> from
  <span class="mono">__DUMP__</span>, matched greedily by descending score against
  <span class="mono">preds/gt_val_namr33.json</span> at IoU &ge; __IOU__, class-agnostic, so
  off-diagonal cells mean the object was found and misnamed. Counts here are taken at a single
  confidence threshold and are a diagnostic view; they are not the ranked-list AP that decides
  anything.
</footer>
</div>

<script>
const D = __DATA__;
const NC = 34, BG = 34, M = D.matrix, NAMES = D.class_names;
const gtTot = c => M.reduce((s,r)=>s+r[c],0);
const thin = new Set(Object.keys(D.per_class).filter(c=>D.per_class[c].gt<=10).map(Number));

/* ---------- error budget ---------- */
let hit=0, mix=0, miss=0, fp=0;
for(let i=0;i<NC;i++){ hit+=M[i][i]; miss+=M[BG][i]; fp+=M[i][BG];
  for(let j=0;j<NC;j++) if(i!==j) mix+=M[i][j]; }
const errTot = miss+fp+mix, total = hit+errTot;
const segs=[
  ["--hit", hit,  "Found and named", "the pipeline got these right"],
  ["--miss",miss, "Missed entirely", "a real object, no detection above threshold"],
  ["--fp",  fp,   "False positive",  "a confident box over nothing annotated"],
  ["--mix", mix,  "Found, misnamed", "correct box, wrong class &mdash; the only part a classifier can fix"],
];
document.getElementById("bar").innerHTML = segs.map(([v,n,l])=>
  `<div class="seg" style="background:var(${v});width:${100*n/total}%">${100*n/total>7?n:""}</div>`).join("");
document.getElementById("keys").innerHTML = segs.map(([v,n,l,d])=>
  `<div class="key" style="border-left-color:var(${v})">
     <span class="k-n" style="color:var(${v})">${n.toLocaleString()}</span>
     <span class="k-l">${l}</span><span class="k-d">${d}</span></div>`).join("");

/* ---------- matrix ---------- */
let maxOff=0;
for(let i=0;i<=NC;i++)for(let j=0;j<=NC;j++) if(i!==j&&M[i][j]>maxOff) maxOff=M[i][j];
let maxDiag=0; for(let i=0;i<NC;i++) if(M[i][i]>maxDiag) maxDiag=M[i][i];
const ramp=(n,max)=> n<=0?0: 0.10+0.90*Math.log(1+n)/Math.log(1+max);
function cellStyle(i,j){
  if(!M[i][j]) return "";
  const v = i===j ? "--hit" : (i===BG ? "--miss" : (j===BG ? "--fp" : "--mix"));
  const a = ramp(M[i][j], i===j?maxDiag:maxOff);
  const dark = a>0.55 ? ";color:#fff" : "";
  return `background:color-mix(in srgb,var(${v}) ${(a*100).toFixed(0)}%,transparent)${dark}`;
}
const label = k => k===BG ? "&#8212; background &#8212;" : NAMES[k];
let h = "<thead><tr><th class='corner'>said &#8595; &nbsp;/&nbsp; actual &#8594;</th>";
for(let j=0;j<=NC;j++) h += `<th class="${j===BG?'bgcol':''}${thin.has(j)?' thin':''}" title="${label(j)}">${j===BG?'bg':j}</th>`;
h += "</tr></thead><tbody>";
for(let i=0;i<=NC;i++){
  h += `<tr class="${i===BG?'bgrow':''}"><th title="${label(i)}">`+
       (i===BG ? "<em style='color:var(--miss)'>nothing detected</em>"
               : `<span class="id ${thin.has(i)?'thin':''}">${i}</span>${NAMES[i]}`)+"</th>";
  for(let j=0;j<=NC;j++){
    const n=M[i][j];
    h += `<td class="${j===BG?'bgcol':''}" style="${cellStyle(i,j)}"
           data-i="${i}" data-j="${j}">${n||""}</td>`;
  }
  h += "</tr>";
}
document.getElementById("mx").innerHTML = h+"</tbody>";

const ro=document.getElementById("readout");
document.getElementById("mx").addEventListener("mouseover", e=>{
  const td=e.target.closest("td"); if(!td) return;
  const i=+td.dataset.i, j=+td.dataset.j, n=M[i][j];
  if(!n){ ro.innerHTML="<span>&#8212;</span>"; return; }
  const tot=gtTot(j);
  if(i===BG) ro.innerHTML=`<b>${n}</b> &nbsp;<b>${NAMES[j]}</b> missed entirely &mdash; ${(100*n/tot).toFixed(1)}% of that class`;
  else if(j===BG) ro.innerHTML=`<b>${n}</b> false positives called <b>${NAMES[i]}</b>`;
  else if(i===j) ro.innerHTML=`<b>${n}</b> &nbsp;<b>${NAMES[i]}</b> correct &mdash; ${(100*n/tot).toFixed(1)}% of that class`;
  else ro.innerHTML=`<b>${n}</b> &nbsp;<b>${NAMES[j]}</b> called <b>${NAMES[i]}</b> &mdash; ${(100*n/tot).toFixed(1)}% of that class`;
});

/* ---------- confusion pairs ---------- */
const cc = D.pairs.filter(p=>p.pred!==BG&&p.gt!==BG);
document.getElementById("pairs").innerHTML =
 "<thead><tr><th>Boxes</th><th>Actually was</th><th>Called it</th><th>Share of that class</th></tr></thead><tbody>"+
 cc.slice(0,16).map(p=>`<tr>
   <td class="n"><span class="pill mix">${p.n}</span></td>
   <td>${p.gt_name}</td><td>${p.pred_name}</td>
   <td class="n">${(100*p.frac_of_gt_class).toFixed(1)}%</td></tr>`).join("")+
 "</tbody>";

/* ---------- per class ---------- */
const rows = Object.keys(D.per_class).map(Number).sort((a,b)=>D.per_class[b].gt-D.per_class[a].gt);
const meter=(v,c)=>`<span class="meter"><i style="width:${(100*v).toFixed(1)}%;background:var(${c})"></i></span>
  <span class="n mono" style="font-size:11.5px"> ${(100*v).toFixed(0)}%</span>`;
document.getElementById("perclass").innerHTML =
 "<thead><tr><th>Id</th><th>Class</th><th>Val boxes</th><th>Recall</th><th>Precision</th>"+
 "<th>Missed</th><th>False pos</th></tr></thead><tbody>"+
 rows.map(c=>{const p=D.per_class[c]; const t=thin.has(c);
   return `<tr${t?' style="opacity:.72"':''}>
   <td class="n ${t?'thin':''}">${c}</td>
   <td>${p.name}${t?' <span class="pill miss">thin</span>':''}</td>
   <td class="n">${p.gt}</td>
   <td>${meter(p.recall,"--hit")}</td><td>${meter(p.precision,"--accent")}</td>
   <td class="n">${p.missed}</td><td class="n">${p.false_pos}</td></tr>`}).join("")+
 "</tbody>";
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="best_pipeline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads((HERE / "scores" / f"{args.name}.confusion.json").read_text())
    m = d["matrix"]
    nc, bg = 34, 34
    hit = sum(m[i][i] for i in range(nc))
    miss = sum(m[bg][i] for i in range(nc))
    fp = sum(m[i][bg] for i in range(nc))
    mix = sum(m[i][j] for i in range(nc) for j in range(nc) if i != j)
    err = miss + fp + mix

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(d, separators=(",", ":")))
            .replace("__CONF__", str(d["conf"]))
            .replace("__IOU__", str(d["iou"]))
            .replace("__DUMP__", d["dump"])
            .replace("__ERRTOT__", f"{err:,}")
            .replace("__MIXPCT__", f"{100 * mix / err:.1f}")
            .replace("__MIX__", f"{mix:,}")
            .replace("__HIT__", f"{hit:,}")
            .replace("__GTTOT__", f"{hit + mix + miss:,}"))
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({len(html) / 1024:.0f} KB)")
    print(f"  correct {hit} | missed {miss} | false pos {fp} | misnamed {mix}")


if __name__ == "__main__":
    main()
