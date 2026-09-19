r"""Add the table a device reviewer will look for first.

Balanced accuracy is not what a wearer experiences. Every cell in this project
already carries the quantities that matter for control -- spurious walk commands
per minute of standing, missed onsets, and calibration error -- measured on all
five cohorts for both configurations, and none of it is in the manuscript for
the reported model.

It is added as it stands, including the places where the branch is worse. On
cohort A it triples the spurious-activation rate, which is a real cost for a
device that moves someone's legs and is not hidden here.

    python tools/add_deploy_table.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)
NL = BS + BS

COH = [
    ("A, exoskeleton", "a_tangent_3seed.json", "a_ablation_s12.json"),
    ("B, treadmill", "b_tangent_3seed.json", "b_tangent_3seed.json"),
    ("C, motor execution", "c_stem_tangent_3seed.json",
     "c_stem_tangent_3seed.json"),
    ("D, BCI IV-2a", "d_bnci_3seed.json", "d_bnci_3seed.json"),
    ("E, exoskeleton", "e_decoded_3seed.json", "e_decoded_3seed.json"),
]
FIELDS = ["acc", "ece", "false_onsets_per_min", "missed_onsets"]


def mean_of(f, arm, field):
    p = os.path.join(ROOT, "results", f)
    if not os.path.exists(p):
        sys.exit("missing " + f)
    d = json.load(io.open(p, encoding="utf-8"))
    v = [d[k][field] for k in d
         if k.startswith(arm + "|s") and d[k].get(field) is not None]
    return float(np.mean(v)) if v else None


def main():
    rows, report = "", []
    for name, bf, cf in COH:
        s_ = [mean_of(cf, "dn_stem", x) for x in FIELDS]
        b_ = [mean_of(bf, "dn_stem_tan", x) for x in FIELDS]
        rows += ("%s & stem & $%.3f$ & $%.3f$ & $%.2f$ & $%.3f$ %s\n"
                 % (name, s_[0], s_[1], s_[2], s_[3], NL))
        rows += ("& " + BS + "textbf{$+$ branch} & $" + BS
                 + "mathbf{%.3f}$ & $%.3f$ & $%.2f$ & $%.3f$ %s\n"
                 % (b_[0], b_[1], b_[2], b_[3], NL))
        rows += BS + "addlinespace[2pt]\n"
        report.append((name, s_, b_))

    ece_better = sum(1 for _, s_, b_ in report if b_[1] < s_[1])
    fa_better = sum(1 for _, s_, b_ in report if b_[2] < s_[2])
    miss_better = sum(1 for _, s_, b_ in report if b_[3] < s_[3])

    tab = (
        BS + "begin{table}[t]\n" + BS + "centering\n"
        + BS + "caption{What a wearer would experience, all five cohorts, "
        "three seeds, mean over participants. ECE is expected calibration "
        "error; FA/min is spurious Walk commands per minute of standing, which "
        "is the quantity a per-window error rate hides, since one long false "
        "run and many scattered false windows score identically; Missed is the "
        "fraction of true onsets never detected. The branch lowers calibration "
        "error on %d of the five cohorts and misses fewer onsets on %d, and it "
        "raises the spurious-activation rate on cohort~A from $0.70$ to "
        "$2.09$ per minute. For a device that moves a wearer's legs that is a "
        "real cost, and it is the one place where higher accuracy does not "
        "mean a better controller.}\n"
        + BS + "label{tab:deploy}\n"
        + BS + "begin{tabular}{llrrrr}\n" + BS + "toprule\n"
        "Cohort & Model & Acc & ECE & FA/min & Missed " + NL + "\n"
        + BS + "midrule\n" + rows + BS + "bottomrule\n"
        + BS + "end{tabular}\n" + BS + "end{table}\n\n"
        "Table~" + BS + "ref{tab:deploy} reports the quantities a controller "
        "would act on. Calibration improves with the branch on %d of the five "
        "cohorts and the missed-onset rate falls on %d, so the accuracy gain is "
        "not bought by a worse-behaved posterior. The exception is the "
        "spurious-activation rate on cohort~A, which rises from $0.70$ to "
        "$2.09$ per minute of standing. A decoder that is more often right "
        "per window can still command more false starts, because the two are "
        "different functions of the same posterior, and on a lower-limb "
        "exoskeleton the second is the one that throws a wearer off balance. "
        "We report it rather than the accuracy alone.\n\n")
    # % binds tighter than +, so applying it inline formats only the final
    # literal. This project has hit that twice; format the whole string once.
    tab = tab % (ece_better, miss_better, ece_better, miss_better)

    s = io.open(TEX, encoding="utf-8").read()
    anchor = BS + "subsection{Published decoders on the two new cohorts}"
    if anchor not in s:
        sys.exit("anchor not found")
    sec = (BS + "subsection{What the branch costs a controller}" + BS
           + "label{sec:deploy}\n\n" + tab)
    s = s.replace(anchor, sec + anchor, 1)
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    print("tab:deploy added\n")
    print("  %-20s %-8s %-8s %-9s %s" % ("cohort", "acc", "ece", "FA/min",
                                         "missed"))
    for name, s_, b_ in report:
        print("  %-20s %.3f->%.3f  %.3f->%.3f  %.2f->%.2f  %.3f->%.3f"
              % (name, s_[0], b_[0], s_[1], b_[1], s_[2], b_[2], s_[3], b_[3]))
    print("\n  branch better: ECE %d/5, FA/min %d/5, missed %d/5"
          % (ece_better, fa_better, miss_better))


if __name__ == "__main__":
    main()
