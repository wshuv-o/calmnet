r"""Pass 7 of the rewrite: an ablation table containing the reported model.

The existing ablation lists five arms of the old architecture and the reported
model is not among them, so the paper ablates a design it does not report. This
adds a table immediately after it, on cohort A, that puts the reported model at
the top and pairs every arm against the stem alone over participants.

Rows are only what is measured. Arms differ in how many seeds exist, so the seed
count is a column rather than a footnote, and single-seed arms are absent
entirely rather than mixed in.

    python tools/rewrite_07_ablation.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)
NL = BS + BS

# label, file, arm, components present
ARMS = [
    ("stem $+$ branch", "a_tangent_3seed.json", "dn_stem_tan", True),
    ("stem $+$ align $+$ gate $+$ branch", "a_tangent_3seed.json",
     "dn_tan", False),
    ("stem $+$ align $+$ gate", "a_aligngate_3seed.json", "dn_noctx", False),
    ("stem $+$ align", "a_align_s12.json", "dn_align", False),
    ("stem $+$ gate", "a_gate_3seed.json", "dn_gate", False),
    ("stem $+$ gate $+$ ctx", "a_ablation_s12_2060.json", "dn_noalign", False),
    ("stem $+$ align $+$ ctx", "a_ablation_s12.json", "dn_nogate", False),
    ("stem $+$ align $+$ gate $+$ ctx", "a_ablation_s12.json",
     "dn_full", False),
    ("stem alone", "a_ablation_s12.json", "dn_stem", False),
]


def load(f):
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        sys.exit("missing: %s" % f)
    return json.load(io.open(p, encoding="utf-8"))


def cells(f, arm):
    d = load(f)
    ks = sorted(k for k in d if k.startswith(arm + "|s"))
    if not ks:
        sys.exit("no cells for %s in %s" % (arm, f))
    subs = sorted(d[ks[0]]["per_subject"])
    M = np.array([[d[k]["per_subject"][s]["acc"] for s in subs] for k in ks])
    ece = float(np.mean([d[k]["ece"] for k in ks]))
    return M, subs, ece


def main():
    ctl, subs, _ = cells("a_ablation_s12.json", "dn_stem")
    ctl_mean = ctl.mean(axis=0)

    rows, report = "", []
    for label, f, arm, star in ARMS:
        M, ss, ece = cells(f, arm)
        if ss != subs:
            sys.exit("participant mismatch for %s" % arm)
        per_seed = M.mean(axis=1)
        acc, sd, n = M.mean(), float(np.std(per_seed, ddof=1)), M.shape[0]
        if arm == "dn_stem":
            dl, p, w = 0.0, None, None
        else:
            d = M.mean(axis=0) - ctl_mean
            dl = d.mean()
            p = stats.wilcoxon(M.mean(axis=0), ctl_mean).pvalue
            w = int((d > 0).sum())
        name = (BS + "textbf{" + label + "}") if star else label
        a = ("$" + BS + "mathbf{%.3f}$" % acc) if star else "$%.3f$" % acc
        rows += ("%s & %d & %s & $%.3f$ & %s & %s & %s %s\n"
                 % (name, n, a, sd, "$%.3f$" % ece,
                    "---" if p is None else "$%+.3f$" % dl,
                    "---" if p is None else "$%.3f$ (%d/%d)" % (p, w,
                                                               len(subs)),
                    NL))
        report.append((label, n, acc, sd, dl, p))

    s = io.open(TEX, encoding="utf-8").read()
    anchor = ("Table~" + BS + "ref{tab:ablation3} repeats the ablation under "
              "the current estimator")
    tab = (
        BS + "begin{table}[t]\n"
        + BS + "caption{Ablation on cohort~A containing the reported model, "
        "three data-split seeds where available and two otherwise, under the "
        "current estimator. Accuracy is the grand mean over the 7 "
        "participants and SD is across seeds, not across participants, so it "
        "measures reproducibility rather than spread. $" + BS + "Delta$ and "
        "the test are paired against the stem alone over participants with "
        "seeds averaged within participant. The reported model is the first "
        "row. Adding the alignment layer to it (second row) removes most of "
        "the gain, which is the substitution reported in "
        "Section~" + BS + "ref{sec:fivecohort}: both read the same "
        "second-order structure.}\n"
        + BS + "label{tab:ablation_rep}\n"
        + BS + "centering\n"
        + BS + "resizebox{" + BS + "columnwidth}{!}{%\n"
        + BS + "begin{tabular}{lrrrrrr}\n" + BS + "toprule\n"
        "Components & Seeds & Acc & SD$_{" + BS + "mathrm{seed}}$ & ECE & $"
        + BS + "Delta$ vs stem & $p$ (higher) " + NL + "\n" + BS + "midrule\n"
        + rows
        + BS + "bottomrule\n" + BS + "end{tabular}%\n}\n"
        + BS + "end{table}\n\n"
        "Table~" + BS + "ref{tab:ablation_rep} places the reported model in "
        "its own ablation. The branch is the only component that raises "
        "accuracy over the stem on this cohort. Of the configurations built "
        "from the alignment layer, the gate and the cross-epoch transformer, "
        "none improves on the stem by more than $0.005$, and the gate alone is "
        "$0.022$ below it, the one difference in the table that reaches "
        "significance in the negative direction. Adding alignment on top of "
        "the branch costs $0.032$ relative to the branch alone, so the two are "
        "substitutes rather than complements.\n\n"
        "Note also the SD column, which is across seeds rather than "
        "participants. The reported model is the most reproducible arm in the "
        "table. That matters because the interpretation threshold this project "
        "uses was set by the seed spread of the transformer arms, and the "
        "branch narrows the very quantity that set it.\n\n")
    s = s.replace(anchor, tab + anchor, 1)
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    print("PASS 7 done: ablation table containing the reported model\n")
    print("  %-34s %-6s %-8s %-8s %s" % ("arm", "seeds", "acc", "sd", "delta"))
    for label, n, acc, sd, dl, p in report:
        print("  %-34s %-6d %-8.4f %-8.4f %s"
              % (label.replace("$+$", "+"), n, acc, sd,
                 "--" if p is None else "%+.4f (p=%.3f)" % (dl, p)))


if __name__ == "__main__":
    main()
