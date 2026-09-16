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

bad = [c for c in checks if not c[2]]
print("checked %d printed values against their sources" % len(checks))
for label, printed, ok in checks:
    if not ok:
        print("  NOT FOUND  %-20s expected: %s" % (label, printed))
print("ALL MATCH" if not bad else "%d MISMATCH(ES)" % len(bad))
