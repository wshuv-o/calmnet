"""Live dashboard for the CALM-Net experiments.

    python tools/dashboard.py            # http://localhost:8765

Reads results/*.json off disk on every poll, so it tracks running jobs without
touching them. The primary panel is the MODULE ABLATION: which pieces of the
architecture actually earn their place, scored on accuracy penalised by movement
leakage rather than on accuracy alone.
"""
from __future__ import annotations
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TASKDIR = Path(os.environ["CLAUDE_TASKDIR"]) if os.environ.get("CLAUDE_TASKDIR") else None
PORT = 8765
T0 = time.time()

ARCH_MODS = ["cancel", "spec_gate", "basis", "attn", "multiband"]
LOSS_MODS = ["adv", "decorr", "hsic", "art"]
ALL_MODS = ARCH_MODS + LOSS_MODS
N_ABLATION = 2 + 2 * len(ALL_MODS) + 1        # bare, full, add-one, leave-one-out, stacked


def _load(name):
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def ablation():
    d = _load("ablation.json") or {}
    done = [k for k, v in d.items() if isinstance(v, dict) and "acc" in v]
    base = (d.get("A_bare") or {}).get("score")
    full = (d.get("B_full") or {}).get("score")
    rows = []
    for m in ALL_MODS:
        A, B = d.get(f"A_+{m}") or {}, d.get(f"B_-{m}") or {}
        a, b = A.get("score"), B.get("score")
        ag = (a - base) if (a is not None and base is not None) else None
        lg = (full - b) if (b is not None and full is not None) else None
        rows.append({"mod": m, "kind": "arch" if m in ARCH_MODS else "loss",
                     "add": a, "add_gain": ag, "loo": b, "loo_gain": lg,
                     "add_acc": A.get("acc"), "add_r2": A.get("r2"),
                     "keep": (ag is not None and ag > 0.005) or (lg is not None and lg > 0.005)})
    return {"done": len(done), "total": N_ABLATION, "rows": rows,
            "bare": d.get("A_bare"), "full": d.get("B_full"),
            "stacked": d.get("C_stacked")}


def sweep():
    d = _load("sweep100.json") or {}
    s1 = {k: v for k, v in d.get("stage1", {}).items() if "mean" in v}
    if not s1:
        return None
    base = (s1.get("000_baseline") or {}).get("mean", {}).get("bal_acc")
    rows = [{"acc": v["mean"]["bal_acc"], "r2": v["mean"]["r2"]} for v in s1.values()]
    n = len(rows)
    ma = sum(r["acc"] for r in rows) / n
    mr = sum(r["r2"] for r in rows) / n
    va = sum((r["acc"] - ma) ** 2 for r in rows)
    vr = sum((r["r2"] - mr) ** 2 for r in rows)
    cov = sum((r["acc"] - ma) * (r["r2"] - mr) for r in rows)
    rep = d.get("stage2", {})
    sds = [v["acc_sd"] for v in rep.values() if "acc_sd" in v]
    return {"n": n, "baseline": base, "corr": cov / (va * vr) ** 0.5 if va and vr else None,
            "real_gains": sum(1 for r in rows if base and r["acc"] > base and r["r2"] <= 0),
            "admissible": sum(1 for r in rows if r["r2"] <= 0),
            "replicated": len(rep),
            "seed_sd": (sum(sds) / len(sds)) if sds else None}


def backbones():
    d = _load("backbone_selection.json") or {}
    return sorted(({"name": k, "acc": v["acc"], "r2": v["r2"]}
                   for k, v in d.items() if "acc" in v), key=lambda r: -r["acc"])


def logs(n=16):
    out = []
    if not (TASKDIR and TASKDIR.is_dir()):
        return out
    c = []
    for f in TASKDIR.glob("*.output"):
        try:
            t = f.read_text(errors="replace")
        except Exception:
            continue
        if "Dashboard:" in t or len(t) < 40:
            continue
        if not any(k in t for k in ("STAGE", "acc ", "R2", "score", "[rep]")):
            continue
        c.append((f, t))
    c.sort(key=lambda ft: -ft[0].stat().st_mtime)
    for f, t in c[:2]:
        out.append({"name": f.stem, "age": int(time.time() - f.stat().st_mtime),
                    "lines": [l for l in t.splitlines() if l.strip()][-n:]})
    return out


def collect():
    return {"ablation": ablation(), "sweep": sweep(), "backbones": backbones(),
            "logs": logs(), "elapsed": time.time() - T0}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>CALM-Net</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--card:#fff;
 --ok:#1a7f5a;--bad:#b4341f;--accent:#2f6f9f}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#16161a;--fg:#e9e9e6;--mut:#9a9a94;--line:#2c2c32;--card:#1e1e24;
 --ok:#4fc08d;--bad:#ef7a63;--accent:#6fa8d0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:19px;margin:0 0 2px}
.sub{color:var(--mut);font-size:12.5px}
.bar{height:8px;background:var(--line);border-radius:99px;overflow:hidden;margin:10px 0}
.bar>i{display:block;height:100%;background:var(--accent);transition:width .4s}
h2{font-size:12.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);
 margin:26px 0 8px;font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.tile .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
.tile .v{font-size:24px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:right;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
 color:var(--mut);padding:8px 10px;border-bottom:1px solid var(--line);font-weight:600}
th:first-child,td:first-child{text-align:left}
td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;font-size:13px}
tr:last-child td{border-bottom:0}
.keep{color:var(--ok);font-weight:600}.drop{color:var(--mut)}
.pos{color:var(--ok);font-weight:600}.neg{color:var(--bad)}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.tag{font-size:10px;padding:1px 6px;border-radius:99px;border:1px solid var(--line);
 color:var(--mut);margin-left:6px}
.empty{padding:14px;color:var(--mut);font-size:13px}
.log{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;white-space:pre-wrap;
 padding:10px 12px;margin:0;max-height:250px;overflow-y:auto}
.logh{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);
 padding:8px 12px 0;font-weight:600;display:flex;justify-content:space-between}
.live{color:var(--ok)}
</style></head><body><div class="wrap">
<h1>CALM-Net &mdash; module ablation</h1>
<div class="sub" id="sub">connecting&hellip;</div>
<div class="bar"><i id="pb" style="width:0%"></i></div>
<div class="tiles" id="tiles"></div>
<h2>Which modules earn their place</h2>
<div class="card" id="abl"></div>
<div class="sub" style="margin-top:8px">score = balanced accuracy &minus; max(0, intent&rarr;motion R&sup2;).
Raw accuracy correlates +0.67 with leakage across 131 architectures, so it cannot be used to select modules.</div>
<h2>Live training log</h2>
<div class="card" id="logs"></div>
<h2>Prior evidence</h2>
<div class="card" id="sw"></div>
<h2>Published backbones</h2>
<div class="card" id="bb"></div>
</div>
<script>
const f=(x,n=3)=>x==null||isNaN(x)?"—":x.toFixed(n);
const sg=x=>x==null||isNaN(x)?"—":(x>=0?"+":"")+x.toFixed(3);
const tile=(k,v,c="")=>`<div class="tile"><div class="k">${k}</div><div class="v" style="${c}">${v}</div></div>`;
function tbl(rows,cols){if(!rows||!rows.length)return '<div class="empty">waiting…</div>';
 return "<table><thead><tr>"+cols.map(c=>`<th>${c[0]}</th>`).join("")+"</tr></thead><tbody>"+
  rows.map(r=>"<tr>"+cols.map(c=>`<td>${c[1](r)}</td>`).join("")+"</tr>").join("")+"</tbody></table>"}
async function tick(){
 let d;try{d=await(await fetch("/data.json",{cache:"no-store"})).json()}catch(e){return}
 const a=d.ablation;
 document.getElementById("pb").style.width=(100*a.done/a.total)+"%";
 document.getElementById("sub").textContent=
  `${a.done} of ${a.total} ablation runs · elapsed ${(d.elapsed/60).toFixed(1)} min`;
 const kept=a.rows.filter(r=>r.keep).map(r=>r.mod);
 document.getElementById("tiles").innerHTML=
  tile("Bare",f(a.bare&&a.bare.score))+
  tile("Full",f(a.full&&a.full.score))+
  tile("Stacked",f(a.stacked&&a.stacked.score),a.stacked?"color:var(--ok)":"")+
  tile("Modules kept",kept.length+" / "+a.rows.length)+
  tile("Stacked acc",f(a.stacked&&a.stacked.acc))+
  tile("Stacked R²",sg(a.stacked&&a.stacked.r2),
   a.stacked?(a.stacked.r2<=0?"color:var(--ok)":"color:var(--bad)"):"");
 document.getElementById("abl").innerHTML=tbl(a.rows,[
  ["module",r=>`<span class="mono">${r.mod}</span><span class="tag">${r.kind}</span>`],
  ["add-one",r=>f(r.add)],
  ["gain",r=>`<span class="${r.add_gain>0?'pos':'neg'}">${sg(r.add_gain)}</span>`],
  ["acc",r=>f(r.add_acc)],
  ["R²",r=>`<span class="${r.add_r2<=0?'pos':'neg'}">${sg(r.add_r2)}</span>`],
  ["leave-out",r=>f(r.loo)],
  ["gain",r=>`<span class="${r.loo_gain>0?'pos':'neg'}">${sg(r.loo_gain)}</span>`],
  ["verdict",r=>r.keep?'<span class="keep">KEEP</span>':'<span class="drop">drop</span>']]);
 const L=d.logs||[];
 document.getElementById("logs").innerHTML=L.length?L.map(g=>
  `<div class="logh"><span>${g.name}</span><span class="${g.age<120?'live':''}">${g.age<120?'● live':g.age+'s idle'}</span></div>`+
  `<pre class="log">${g.lines.map(x=>x.replace(/[<&]/g,c=>c=='<'?'&lt;':'&amp;')).join("\n")}</pre>`).join("")
  :'<div class="empty">no active job</div>';
 const s=d.sweep;
 document.getElementById("sw").innerHTML=s?
  `<div class="empty">${s.n} architectures · baseline ${f(s.baseline)} ·
   <b>real gains ${s.real_gains}</b> · admissible ${s.admissible} ·
   corr(accuracy, leakage) <b>${sg(s.corr)}</b><br>
   seed replication: ${s.replicated} variants · mean sd ${f(s.seed_sd)}</div>`
  :'<div class="empty">—</div>';
 document.getElementById("bb").innerHTML=tbl((d.backbones||[]).slice(0,6),[
  ["backbone",r=>`<span class="mono">${r.name}</span>`],["acc",r=>f(r.acc)],
  ["R²",r=>`<span class="${r.r2<=0?'pos':'neg'}">${sg(r.r2)}</span>`]]);
}
tick();setInterval(tick,3000);
</script></body></html>"""


class Server(HTTPServer):
    # Windows honours SO_REUSEADDR by letting SEVERAL sockets bind the same
    # port; connections then get routed to whichever bound first, including a
    # zombie from a previous run. Refusing reuse makes a stale server an
    # immediate, visible "address in use" instead of a silent hang.
    allow_reuse_address = False


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/data.json"):
            body, ct = json.dumps(collect()).encode(), "application/json"
        else:
            body, ct = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"Dashboard: http://localhost:{port}", flush=True)
    Server(("127.0.0.1", port), H).serve_forever()
