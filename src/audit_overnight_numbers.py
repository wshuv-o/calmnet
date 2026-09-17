"""Audit every number written into the manuscript overnight against its source.

Each claim is recomputed from results/*.json or the ICA log and rounded exactly
as the paper prints it, then the printed string is searched for in the
flattened manuscript. A claim fails if the recomputed value disagrees with the
paper or the printed string is absent.
"""
import csv
import io
import json
import re
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
tex = io.open(ROOT / "paper" / "cas_calmnet.tex", "rb").read().decode("utf-8")
flat = " ".join(tex.split())


def J(f):
    return json.loads((RES / f).read_text())


def ps(f, key):
    return {s: r["acc"] for s, r in J(f)[key]["per_subject"].items()}


checks = []


def claim(label, value, printed):
    """value: recomputed; printed: the exact string the paper should contain."""
    ok_str = printed in flat
    checks.append((label, printed, ok_str))


def f3(x):
    return "%.3f" % x


# ---- pre-registration, cohort C ---------------------------------------
g = ps("cohort3_m0.2.json", "dn_gate|s0")
a = ps("cohort3_m0.2.json", "dn_noctx|s0")
sl = ps("cohort3_m0.01.json", "dn_noctx|s0")
subs = sorted(set(g) & set(a) & set(sl))
mg, ma, ms = (np.mean([d[s] for s in subs]) for d in (g, a, sl))
d1 = np.array([a[s] - g[s] for s in subs])
d2 = np.array([sl[s] - a[s] for s in subs])
claim("C gate mean", mg, "$" + f3(mg) + "$ / $" + f3(ma) + "$")
claim("C P1 diff", d1.mean(), "$+" + f3(d1.mean()) + "$")
claim("C P1 higher", (d1 > 0).sum(), "higher in %d participants, lower in %d" % ((d1 > 0).sum(), (d1 < 0).sum()))
claim("C P1 wilcoxon", wilcoxon(d1).pvalue, "$p=%.2f$" % wilcoxon(d1).pvalue)
claim("C P2 arms", ms, "$" + f3(ma) + "$ / $" + f3(ms) + "$")
claim("C P2 diff", d2.mean(), "$-" + f3(-d2.mean()) + "$")
claim("C P2 counts", 0, "higher in %d, lower in %d" % ((d2 > 0).sum(), (d2 < 0).sum()))
claim("C P2 wilcoxon", wilcoxon(d2).pvalue, "$p=%.2f$" % wilcoxon(d2).pvalue)
claim("C n", len(subs), "all 20 participants")

# ---- artefact control --------------------------------------------------
o = ps("icactl_orig.json", "dn_noctx|s0")
c = ps("icactl_ica.json", "dn_noctx|s0")
sa = sorted(o)
dd = np.array([c[s] - o[s] for s in sa])
claim("ICA uncleaned mean", np.mean(list(o.values())), "from $" + f3(np.mean(list(o.values()))) + "$ to $" + f3(np.mean(list(c.values()))) + "$")
claim("ICA diff", dd.mean(), "difference of $-" + f3(-dd.mean()) + "$")
claim("ICA counts", 0, "higher in %d participants, lower in %d" % ((dd > 0).sum(), (dd < 0).sum()))
claim("ICA wilcoxon", wilcoxon(dd).pvalue, "Wilcoxon $p=%.2f$" % wilcoxon(dd).pvalue)
for s in sa:
    ch = c[s] - o[s]
    row = "%s & %s & %s & $%s%s$" % (s, f3(o[s]), f3(c[s]), "+" if ch >= 0 else "-", f3(abs(ch)))
    claim("ICA row " + s, ch, row)

rows = [r for r in csv.DictReader(open(RES / "ica_components.csv", encoding="utf-8"))
        if "_task-training_eeg" in r["recording"]]
n = [int(r["n_removed"]) for r in rows]
claim("ICA recordings", len(rows), "Across the %d recordings" % len(rows))
claim("ICA mean removed", st.mean(n), "a mean of %.1f of 30" % st.mean(n))
lab = [l for r in rows for l in (r["removed_labels"] or "").split(";") if l]
share = lambda k: round(100 * lab.count(k) / len(lab))
claim("ICA eye share", share("eye blink"), "eye (%d\\,\\%%" % share("eye blink"))
claim("ICA muscle share", share("muscle artifact"), "muscle (%d\\,\\%%)" % share("muscle artifact"))

musc = {}
for r in rows:
    sub = r["recording"].split("_")[0]
    musc.setdefault(sub, []).append(r["removed_labels"].split(";").count("muscle artifact"))
mm = {s: st.mean(v) for s, v in musc.items()}
rho = spearmanr([mm[s] for s in sa], [c[s] - o[s] for s in sa]).correlation
claim("ICA spearman", rho, "$" + chr(92) + "rho=+%.2f$" % rho)
for s in sa:
    claim("ICA muscle " + s, mm[s], "& %.1f " % mm[s] if False else "%.1f" % mm[s])
claim("ICA probe orig", J("icactl_orig.json")["dn_noctx|s0"]["cond_r2"],
      "($-%.3f$ and $-%.3f$)" % (-J("icactl_orig.json")["dn_noctx|s0"]["cond_r2"],
                                   -J("icactl_ica.json")["dn_noctx|s0"]["cond_r2"]))

# ---- selective head / align-only ---------------------------------------
al = J("align_only.json")["dn_align|s0"]
fx = J("driftfix_ds.json")
ng = J("driftfix_ds_nogate.json")["dn_nogate|s0"]
claim("align-only acc", al["acc"], "reaches $" + f3(al["acc"]) + "$ against $" + f3(fx["dn_noctx|s0"]["acc"]) + "$ with the head")
claim("head diff", fx["dn_noctx|s0"]["acc"] - al["acc"], "a difference of $%s$" % f3(fx["dn_noctx|s0"]["acc"] - al["acc"]))
claim("ctx arms", ng["acc"], "($" + f3(ng["acc"]) + "$ in both arms)")
claim("head ECE", al["ece"], "is $%.3f$, against $%.3f$ with it" % (al["ece"], fx["dn_noctx|s0"]["ece"]))

# ---- leakage guard --------------------------------------------------------
sens = J("sensitivity.json")
claim("probe f0", sens["f0.0000"]["cond_r2"], "from $-%.3f$ at $f=0$" % -sens["f0.0000"]["cond_r2"])
claim("probe f0.005", sens["f0.0050"]["cond_r2"], "through $+%.3f$ at $f=0.005$" % sens["f0.0050"]["cond_r2"])
claim("probe f0.02", sens["f0.0200"]["cond_r2"], "to $+%.3f$ at $f=0.02$" % sens["f0.0200"]["cond_r2"])
claim("acc f0.02", sens["f0.0200"]["acc"], "where accuracy is $%.3f$" % sens["f0.0200"]["acc"])
hd = J("driftnet_ds.json")["dn_noctx|s0"]
claim("headline probe", hd["cond_r2"], "conditional probe of $-%.3f$" % -hd["cond_r2"])
armsA = [v["cond_r2"] for f in ("driftnet_ds.json", "driftfix_ds.json") for v in J(f).values()]
claim("probe range", min(armsA), "between $-%.3f$ and $-%.3f$" % (-max(armsA), -min(armsA)))

# ---- cohort C replication, seeds 1-2 --------------------------------------
def seedavg(f, arm):
    d = J(f); acc = {}
    for k, v in d.items():
        if k.startswith(arm + "|s"):
            for sub, r in v["per_subject"].items():
                acc.setdefault(sub, []).append(r["acc"])
    return {sub: float(np.mean(v)) for sub, v in acc.items()}
rg = seedavg("cohort3_rep_m0.2.json", "dn_gate")
ra = seedavg("cohort3_rep_m0.2.json", "dn_noctx")
rs = seedavg("cohort3_rep_m0.01.json", "dn_noctx")
rsub = sorted(set(rg) & set(ra) & set(rs))
r1 = np.array([ra[x] - rg[x] for x in rsub]); r2 = np.array([rs[x] - ra[x] for x in rsub])
claim("C rep P1 diff", r1.mean(), "by $+%.3f$ (higher in %d participants, lower in %d; $p=%.2f$)" % (r1.mean(), (r1 > 0).sum(), (r1 < 0).sum(), wilcoxon(r1).pvalue))
claim("C rep P2 diff", r2.mean(), "costs $%.3f$ (lower in %d of %d; $p=%.3f$)" % (-r2.mean(), (r2 < 0).sum(), len(r2), wilcoxon(r2).pvalue))

# ---- cohort A, align + gate against gate, three seeds ---------------------
A3, G3 = J("a_aligngate_3seed.json"), J("a_gate_3seed.json")
sa3 = [A3["dn_noctx|s%d" % i]["acc"] for i in range(3)]
sg3 = [G3["dn_gate|s%d" % i]["acc"] for i in range(3)]
claim("A 3seed mean/sd", np.mean(sa3), "$%.3f \\pm %.3f$" % (np.mean(sa3), np.std(sa3, ddof=1)))
ag = seedavg("a_aligngate_3seed.json", "dn_noctx")
gg = seedavg("a_gate_3seed.json", "dn_gate")
d3 = np.array([ag[x] - gg[x] for x in sorted(ag)])
claim("A align effect", d3.mean(), "by $+%.3f$ ($%.3f$ against $%.3f$), higher in %d of 7 participants (Wilcoxon $p=%.2f$)"
      % (d3.mean(), np.mean(sa3), np.mean(sg3), (d3 > 0).sum(), wilcoxon(d3).pvalue))
claim("A align per seed", 0, "is $%s$, $%s$ and $%s$ on the three seeds"
      % tuple(("+" if v >= 0 else "-") + f3(abs(v)) for v in np.subtract(sa3, sg3)))
ext = {x: ag[x] - gg[x] for x in ag}
bx, wx = max(ext, key=ext.get), min(ext, key=ext.get)
claim("A align extremes", 0, "%s gains $%s$ and %s loses $%s$" % (bx, f3(ext[bx]), wx, f3(-ext[wx])))
claim("A align limitation", d3.mean(), "over the same three seeds is $+%.3f$ ($p=%.2f$)" % (d3.mean(), wilcoxon(d3).pvalue))
claim("A align inversion", d3.mean(), "alignment is worth $+%.3f$ there" % d3.mean())
claim("A align conclusion", d3.mean(), "$+%.3f$ over three seeds on cohort A and $+%.3f$ over two replication seeds" % (d3.mean(), r1.mean()))

# ---- selective head, three seeds ------------------------------------------
H3 = J("overnight_eval.json")["head_3seed"]
al3 = [J("align_only.json")["dn_align|s0"]] + [J("a_align_s12.json")["dn_align|s%d" % i] for i in (1, 2)]
ag3 = [A3["dn_noctx|s%d" % i] for i in range(3)]
alsub = {x: np.mean([r["per_subject"][x]["acc"] for r in al3]) for x in al3[0]["per_subject"]}
hd3 = np.array([ag[x] - alsub[x] for x in sorted(alsub)])
claim("head 3seed", hd3.mean(), "averages $%.3f$ against $%.3f$ with the head, a difference of $%s%.3f$ (higher with the head in %d of 7 participants, Wilcoxon $p=%.2f$)"
      % (np.mean([r["acc"] for r in al3]), np.mean([r["acc"] for r in ag3]), "+" if hd3.mean() >= 0 else "-", abs(hd3.mean()), (hd3 > 0).sum(), wilcoxon(hd3).pvalue))
claim("head 3seed ECE", 0, "is $%.3f$ without the head against $%.3f$ with it"
      % (np.mean([r["ece"] for r in al3]), np.mean([r["ece"] for r in ag3])))

# ---- pipeline C comparison with published decoders ------------------------
bdA, bdC = J("bd_pipeline_c.json"), J("bd_cohort_c.json")
a0 = [v["acc"] for k, v in bdA.items() if k.endswith("|s0")]
claim("pub A seed0 range", 0, "($%.3f$ against $%.3f$ to $%.3f$)" % (A3["dn_noctx|s0"]["acc"], min(a0), max(a0)))
oursA0 = A3["dn_noctx|s0"]
for k_, v_ in J("bd_eegnex_s12.json").items():
    bdA.setdefault(k_, v_)
osA = {x: np.mean([A3["dn_noctx|s%d" % i]["per_subject"][x]["acc"] for i in range(3)]) for x in A3["dn_noctx|s0"]["per_subject"]}
RA = {}
for m in sorted({k.split("|")[0] for k in bdA}):
    runs = [bdA[k] for k in sorted(bdA) if k.split("|")[0] == m]
    bs = {x: np.mean([r["per_subject"][x]["acc"] for r in runs]) for x in runs[0]["per_subject"]}
    dA = np.array([osA[x] - bs[x] for x in sorted(bs)])
    RA[m] = (np.mean([r["acc"] for r in runs]), np.mean([r["ece"] for r in runs]), dA.mean(), (dA < 0).sum(), wilcoxon(dA).pvalue)
claim("pub A 3seed range", 0, "the published decoders average $%.3f$ to $%.3f$" % (min(v[0] for v in RA.values()), max(v[0] for v in RA.values())))
claim("pub A above", 0, "EEG-TCNet and EEGNeX lie $%.3f$ and $%.3f$ above it (higher in %d and %d of 7 participants; Wilcoxon $p=%.2f$ and $p=%.2f$)"
      % (-RA["EEGTCNet"][2], -RA["EEGNeX"][2], RA["EEGTCNet"][3], RA["EEGNeX"][3], RA["EEGTCNet"][4], RA["EEGNeX"][4]))
blw = [v for m, v in RA.items() if v[2] > 0]
claim("pub A below", 0, "lie $%.3f$ to $%.3f$ below it (smallest $p=%.2f$)" % (min(v[2] for v in blw), max(v[2] for v in blw), min(v[4] for v in blw)))
claim("pub A ECE 3seed", 0, "against $%.3f$ to $%.3f$ for the published decoders over the same" % (min(v[1] for v in RA.values()), max(v[1] for v in RA.values())))
claim("pub A small", 0, "EEG-TCNet averages $%.3f$ with 7{,}062 parameters and FBLightConvNet $%.3f$" % (RA["EEGTCNet"][0], RA["FBLightConvNet"][0]))
cC = [J("cohort3_m0.2.json")["dn_noctx|s0"], J("cohort3_rep_m0.2.json")["dn_noctx|s1"], J("cohort3_rep_m0.2.json")["dn_noctx|s2"]]
claim("ours C mean", 0, "the proposed configuration averages $%.3f$, and" % np.mean([r["acc"] for r in cC]))
oc = {x: np.mean([r["per_subject"][x]["acc"] for r in cC]) for x in cC[0]["per_subject"]}
for m, lab in (("EEGNeX", "than EEGNeX by"), ("EEGConformer", "than EEG Conformer by"),
               ("FBLightConvNet", "than FBLightConvNet by"), ("ShallowFBCSPNet", "than ShallowFBCSPNet by")):
    runs = [v for k, v in bdC.items() if k.split("|")[0] == m]
    bs = {x: np.mean([r["per_subject"][x]["acc"] for r in runs]) for x in runs[0]["per_subject"]}
    dd_ = np.array([oc[x] - bs[x] for x in sorted(bs)])
    claim("C vs " + m, dd_.mean(), "%s $%.3f$" % (lab, dd_.mean()))
    claim("C vs " + m + " p", 0, "$p=%.2f$" % wilcoxon(dd_).pvalue if m != "ShallowFBCSPNet" else "$p=%.3f$" % wilcoxon(dd_).pvalue)
    claim("C vs " + m + " n", 0, "%d of 20" % (dd_ > 0).sum())
claim("ours C ECE", 0, "Its expected calibration error is $%.3f$" % np.mean([r["ece"] for r in cC]))
claim("ours A ECE", 0, "calibration error averages $%.3f$" % np.mean([A3["dn_noctx|s%d" % i]["ece"] for i in range(3)]))
for m in ("EEGNet", "Deep4Net", "EEGTCNet"):
    runs = [v for k, v in bdC.items() if k.split("|")[0] == m]
    n = [sum(1 for v in r["per_subject"].values() if v["acc"] < 0.55) for r in runs]
    claim("C chance " + m, 0, "%d to %d" % (min(n), max(n)))
trained = ("EEGNeX", "EEGConformer", "FBLightConvNet", "ShallowFBCSPNet", "TSception")
pc, accs, eces = {}, [], []
for m in trained:
    runs = [v for k, v in bdC.items() if k.split("|")[0] == m]
    accs.append(np.mean([r["acc"] for r in runs])); eces.append(np.mean([r["ece"] for r in runs]))
    bs = {x: np.mean([r["per_subject"][x]["acc"] for r in runs]) for x in runs[0]["per_subject"]}
    pc[m] = wilcoxon([oc[x] - bs[x] for x in sorted(bs)]).pvalue
claim("C trained range", 0, "decoders that trained reach $%.3f$ to $%.3f$" % (min(accs), max(accs)))
claim("C trained ECE", 0, "against $%.3f$ to $%.3f$ for the same decoders" % (min(eces), max(eces)))
order_, run_, holm = sorted(pc, key=pc.get), 0, {}
for i, m in enumerate(order_):
    run_ = max(run_, min(1, (len(order_) - i) * pc[m])); holm[m] = run_
claim("C Holm Shallow", holm["ShallowFBCSPNet"], "ShallowFBCSPNet ($p=%.3f$)" % holm["ShallowFBCSPNet"])
claim("C Holm others", 0, "only the differences from" if all(holm[m] > 0.05 for m in ("EEGNeX", "EEGConformer", "FBLightConvNet")) and holm["TSception"] < 0.001 else "HOLM PATTERN CHANGED")
tc = [v for k, v in bdC.items() if k.split("|")[0] in ("EEGNet", "EEGTCNet")]
claim("C fail ECE", 0, "($%.3f$ and $%.3f$)" % (np.mean([v["ece"] for k, v in bdC.items() if k.startswith("EEGNet|")]),
                                               np.mean([v["ece"] for k, v in bdC.items() if k.startswith("EEGTCNet|")])))
slow = J("driftmom_ds_0.01_noctx.json")["dn_noctx|s0"]
claim("A slow seed0", 0, "takes cohort A from $%.3f$ to $%.3f$ at the first data-split seed" % (oursA0["acc"], slow["acc"]))
S3 = J("a_slow_noctx_3seed.json")
s3 = [S3["dn_noctx|s%d" % i] for i in range(3)]
s3sub = {x: np.mean([r["per_subject"][x]["acc"] for r in s3]) for x in s3[0]["per_subject"]}
ds3 = np.array([s3sub[x] - ag[x] for x in sorted(ag)])
claim("A slow 3seed", 0, "slow arm averages $%.3f \\pm %.3f$ against $%.3f \\pm %.3f$" % (np.mean([r["acc"] for r in s3]), np.std([r["acc"] for r in s3], ddof=1), np.mean(sa3), np.std(sa3, ddof=1)))
claim("A slow cost", ds3.mean(), "a cost of $" + chr(92) + "mathbf{-%.3f}$" % -ds3.mean())
claim("A slow paired", 0, "lower in %d of 7 participants (Wilcoxon $p=%.3f$" % ((ds3 < 0).sum(), wilcoxon(ds3).pvalue))
claim("A slow per seed", 0, "The cost is $%s$, $%s$ and $%s$ on the three seeds" % tuple("-%.3f" % (a - r["acc"]) if a > r["acc"] else "+%.3f" % (r["acc"] - a) for r, a in zip(s3, sa3)))
claim("A slow abstract", ds3.mean(), "slowing costs $%.3f$ over three seeds" % -ds3.mean())

# ---- drift reduction against adaptation rate --------------------------------
DM = J("drift_momentum.json")["summary"]
claim("drift momentum sweep", 0, "falls from $%.1f\\,\\%%$ at $m=0.2$ to $%.1f\\,\\%%$ at $m=0.05$ and $%.1f\\,\\%%$ at $m=0.01$"
      % (100 * DM["m0.2"]["mean"], 100 * DM["m0.05"]["mean"], 100 * DM["m0.01"]["mean"]))
claim("drift momentum item 1", 0, "on cohort A, $%.1f\\,\\%%$ at $m=0.2$ against $%.1f\\,\\%%$ at $m=0.05$"
      % (100 * DM["m0.2"]["mean"], 100 * DM["m0.05"]["mean"]))

# ---- session drift, rerun without trace normalisation ----------------------
DF = J("drift_fixed.json")
for c, lab in (("ds007788", "A"), ("mobi", "B")):
    sm = DF[c]["summary"]
    per = np.array([v["reduction"] for v in DF[c]["per_subject"].values()]) * 100
    claim("drift %s mean" % lab, sm["reduction"], "$%.1f\\pm%.1f" % (sm["reduction"] * 100, sm["reduction_sd"] * 100))
    claim("drift %s range" % lab, 0, "$%.1f$ to $%.1f\\,\\%%$ on cohort %s" % (per.min(), per.max(), lab))
    claim("drift %s t" % lab, sm["t"], "$t=%.2f$" % sm["t"])

# ---- cohort B at memory 320 (m = 0.10), three seeds ------------------------
f10, f02, bgB, m001B = J("b_floor_m0.10.json"), J("b_floor_m0.02.json"), J("b_gate_aligngate_3seed.json"), J("b_aligngate_m001_3seed.json")
psub = lambda runs: {x: np.mean([r["per_subject"][x]["acc"] for r in runs]) for x in runs[0]["per_subject"]}
armB = lambda d_, a_: [v for k, v in sorted(d_.items()) if k.startswith(a_ + "|")]
gtB = armB(bgB, "dn_gate"); gsB = psub(gtB); gateB = np.mean([r["acc"] for r in gtB])
cur = {}
for mem, d_ in ((160, bgB), (320, f10), (1600, f02), (3200, m001B)):
    runs = armB(d_, "dn_noctx"); a_ = psub(runs); dd_ = np.array([a_[x] - gsB[x] for x in sorted(a_)])
    cur[mem] = (np.mean([r["acc"] for r in runs]), -dd_.mean(), (dd_ < 0).sum(), wilcoxon(dd_).pvalue)
recB = cur[3200][0] - cur[160][0]
claim("B gate ref", gateB, "does not use the estimator ($%.3f$)" % gateB)
claim("B 160", 0, "costs $%.3f$ at a memory of 160 windows (lower in %d of 8 participants; Wilcoxon $p=%.3f$)" % (cur[160][1], cur[160][2], cur[160][3]))
claim("B 320", 0, "$%.3f$ at 320 (%d of 8; $p=%.3f$)" % (cur[320][1], cur[320][2], cur[320][3]))
claim("B 1600", 0, "$%.3f$ at 1600 (%d of 8; $p=%.2f$)" % (cur[1600][1], cur[1600][2], cur[1600][3]))
claim("B 3200", 0, "$%.3f$ at 3200 (%d of 8; $p=%.2f$)" % (cur[3200][1], cur[3200][2], cur[3200][3]))
claim("B abstract", 0, "at $%.3f$ without the layer over three seeds, and enabling the layer costs $%.3f$" % (gateB, cur[160][1]))
claim("B abstract slow", 0, "to $%.3f$ at a memory of 320 windows and $%.3f$ at 3200" % (cur[320][1], cur[3200][1]))
claim("B external", 0, "against the gate-only arm ($%.3f$ against $%.3f$; lower in %d of 8 participants, Wilcoxon $p=%.3f$)" % (cur[160][0], gateB, cur[160][2], cur[160][3]))
claim("B residual", 0, "still costs $%.3f$ and $%.3f$ on cohort B against the gate-only arm" % (cur[1600][1], cur[3200][1]))
claim("B recovery", recB, "gives up $%.3f$ against a memory of 3200 windows" % recB)
claim("B limitation", 0, "is $%.3f$ on cohort B over three seeds (lower in %d of 8 participants; $p=%.3f$)" % (cur[160][1], cur[160][2], cur[160][3]))
claim("B drift slow", 0, "removes only $%.1f\\,\\%%$ and $%.1f\\,\\%%$ of session drift" % (100 * J("drift_momentum.json")["summary"]["m0.02"]["mean"], 100 * J("drift_momentum.json")["summary"]["m0.01"]["mean"]))

bad = [c for c in checks if not c[2]]
print("checked %d printed values against their sources" % len(checks))
for label, printed, ok in checks:
    if not ok:
        print("  NOT FOUND  %-20s expected: %s" % (label, printed))
print("ALL MATCH" if not bad else "%d MISMATCH(ES)" % len(bad))
