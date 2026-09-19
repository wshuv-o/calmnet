r"""Rebuild the ablation floats around the final architecture, and restore fig:rate.

Replacing the ablation and rate sections took their floats with them. Rather
than restoring the old versions, the ablation is rebuilt as an ablation of the
proposed model: the stem, the stem plus the branch, and the three choices of
reference the branch can use. Every row is a property of the final architecture.

fig:rate is restored unchanged, because it already plots the memory
manipulation on the branch.

Remaining references to removed tables are repaired in the Limitations section,
which still cites drift and old-arm measurements.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)
NL = BS + BS

ROWS = [
    ("stem alone", "a_ablation_s12.json", "dn_stem", "---", False),
    ("stem $+$ branch, running reference", "a_tangent_3seed.json",
     "dn_stem_tan", "adapts on the test stream", True),
    ("stem $+$ branch, reference frozen after fitting",
     "a_tangent_frozen_3seed.json", "dn_stem_tan", "no test-time update",
     False),
    ("stem $+$ branch, no reference", "a_tangent_ref_none.json",
     "dn_stem_tan", "no whitening", False),
]

LIM_SUBS = [
    ("The cohort in which more drift was removed ($62.5" + BS + ","
     + BS + "%$ against $45.9" + BS + "," + BS + "%$, Table~" + BS
     + "ref{tab:drift}) is the", "The cohort in which the branch fails is the"),
    ("the drift results (Table~" + BS + "ref{tab:drift}), the temporal and "
     "normalisation controls, t", "the temporal and normalisation controls, t"),
    ("(Section~" + BS + "ref{sec:ablation}, Table~" + BS
     + "ref{tab:ablation3}),", "(Section~" + BS + "ref{sec:ablation}),"),
    ("(Table~" + BS + "ref{tab:rate}), because the diagnosis accounted only "
     "for t", "because the diagnosis accounted only for t"),
    ("(Table~" + BS + "ref{tab:ratecurve}).", "."),
    ("(Table~" + BS + "ref{tab:rate}), which is how the two-sided form arose.",
     ", which is how the two-sided form arose."),
]


def load(f):
    p = os.path.join(ROOT, "results", f)
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))


def cells(f, arm):
    d = load(f)
    if d is None:
        return None
    ks = sorted(k for k in d if k.startswith(arm + "|s"))
    if len(ks) < 3:
        return None
    subs = sorted(d[ks[0]]["per_subject"])
    return np.array([[d[k]["per_subject"][s]["acc"] for s in subs]
                     for k in ks]), subs


def main():
    ctl, subs = cells("a_ablation_s12.json", "dn_stem")
    base = ctl.mean(axis=0)

    body = ""
    report = []
    for lab, f, arm, note, star in ROWS:
        got = cells(f, arm)
        if got is None:
            print("   skipped (needs 3 seeds): %s" % lab)
            continue
        M, _ = got
        acc = M.mean()
        sd = float(np.std(M.mean(axis=1), ddof=1))
        if arm == "dn_stem":
            dl, p, w = None, None, None
        else:
            d = M.mean(axis=0) - base
            dl, p, w = d.mean(), stats.wilcoxon(M.mean(axis=0), base).pvalue, \
                int((d > 0).sum())
        nm = (BS + "textbf{" + lab + "}") if star else lab
        av = ("$" + BS + "mathbf{%.3f}$" % acc) if star else "$%.3f$" % acc
        body += ("%s & %s & $%.3f$ & %s & %s %s\n"
                 % (nm, av, sd,
                    "---" if dl is None else "$%+.3f$" % dl,
                    "---" if p is None else "$%.3f$ (%d/%d)"
                    % (p, w, len(subs)), NL))
        report.append((lab, acc, sd, dl, p))

    tab = (
        BS + "begin{table}[t]\n" + BS + "centering\n"
        + BS + "caption{Ablation of the proposed model on cohort~A, three "
        "data-split seeds. Every row is a choice the architecture actually "
        "makes: whether the second-order branch is present, and where its "
        "tangent reference comes from. SD is across seeds, so it measures "
        "reproducibility; $" + BS + "Delta$ and the test are paired against "
        "the stem across participants with seeds averaged within participant. "
        "The branch raises accuracy and reduces the seed spread by about "
        "nine-fold. Freezing the reference after fitting or removing it "
        "entirely both cost accuracy on this cohort, whose class blocks are "
        "short enough for an adapting reference to be safe.}\n"
        + BS + "label{tab:ablation_rep}\n"
        + BS + "begin{tabular}{lrrrr}\n" + BS + "toprule\n"
        "Configuration & Acc & SD$_{" + BS + "mathrm{seed}}$ & $" + BS
        + "Delta$ vs stem & $p$ (higher) " + NL + "\n" + BS + "midrule\n"
        + body + BS + "bottomrule\n" + BS + "end{tabular}\n"
        + BS + "end{table}\n\n")

    figabl = (
        BS + "begin{figure}[t]\n" + BS + "centering\n"
        + BS + "includegraphics[width=" + BS + "columnwidth]"
        "{../results/fig_ablation.pdf}\n"
        + BS + "caption{The branch against the stem on cohort~A, three seeds. "
        "(a) Balanced accuracy, with the proposed model solid and the stem "
        "grey; white bars are one standard deviation across seeds. (b) The "
        "same as a paired change against the stem, with the annotation giving "
        "how many participants improved and the shaded band the $0.02$ "
        "threshold below which differences are not interpreted.}\n"
        + BS + "label{fig:ablation}\n" + BS + "end{figure}\n\n")

    figrate = (
        BS + "begin{figure}[t]\n" + BS + "centering\n"
        + BS + "includegraphics[width=" + BS + "columnwidth]"
        "{../results/fig_rate.pdf}\n"
        + BS + "caption{The rate condition measured by manipulation. Each "
        "point moves only the estimator memory $" + BS + "mu = N/m$, with the "
        "architecture, the data and every other setting fixed; dashed "
        "verticals mark each cohort's class-block length and annotations give "
        "how many participants improved. The branch gains at memories above "
        "the block length and reverses below it.}\n"
        + BS + "label{fig:rate}\n" + BS + "end{figure}\n\n")

    s = io.open(TEX, encoding="utf-8").read()
    i = s.index(BS + "subsection{The branch against the stem}")
    j = s.index("\n\n", s.index("threshold below which", i)) + 2 \
        if "threshold below which" in s[i:i + 2000] else s.index("\n\n", i) + 2
    k = s.index(BS + "subsection", i + 10)
    s = s[:k] + figabl + tab + s[k:]

    r = s.index(BS + "subsection{The reference has an admissible rate}")
    r2 = s.index(BS + "subsection", r + 10)
    s = s[:r2] + figrate + s[r2:]

    for old, new in LIM_SUBS:
        if old in s:
            s = s.replace(old, new)
        else:
            print("   limitations: no match for %s..." % old[:48])

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    print("\n  ablation rebuilt on the final architecture:")
    for lab, acc, sd, dl, p in report:
        print("   %-46s %.4f  sd %.4f  %s"
              % (lab.replace("$+$", "+"), acc, sd,
                 "--" if dl is None else "%+.4f p=%.3f" % (dl, p)))

    labs = set(re.findall(r"label\{([^}]+)\}", s + io.open(
        os.path.join(ROOT, "paper", "tables_auto.tex"),
        encoding="utf-8").read()))
    from collections import Counter
    miss = Counter(x for x in re.findall(r"ref\{([^}]+)\}", s)
                   if x not in labs)
    print("\n  dangling: %d %s" % (sum(miss.values()), dict(miss) or ""))


if __name__ == "__main__":
    main()
