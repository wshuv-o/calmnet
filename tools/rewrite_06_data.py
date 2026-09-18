r"""Pass 6 of the rewrite: the Data section and the block-structure table.

The section described three cohorts and the manuscript now reports five. Cohorts
D and E are added, and the block table gains their rows so that every cohort the
results table names is defined before it appears.

Block statistics are computed from the label sequence by tools/tau_blk_rule.py
over the fitting recordings only, which is what a deployer would have, and
counted within a recording rather than across one.

    python tools/rewrite_06_data.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)
NL = BS + BS

# blocks, median tau_blk, longest -- measured, see the commit for provenance
BLOCKS = [
    ("A (exoskeleton)", 789, 18, 486),
    ("B (treadmill)", 66, 214, 2211),
    ("C (motor execution)", 3600, 5, 5),
    ("D (BCI IV-2a)", 734, 1, 6),
    ("E (exoskeleton, DECODED)", 192, 13, 33),
]


def main():
    s = io.open(TEX, encoding="utf-8").read()

    # ---- two new cohort paragraphs, before the "protocols differ" sentence
    anchor = "The protocols differ sharply in temporal structure"
    new = (
        BS + "textbf{Cohort D (BCI Competition IV-2a, BNCI2014-001).} Nine "
        "participants recorded in two sessions on different days, 22-channel "
        "EEG at 250\\,Hz resampled to 100\\,Hz, cued motor imagery of the left "
        "against the right hand. The model is fitted on the first session and "
        "evaluated on the second, with 4\\,s windows covering the imagery "
        "interval, 288 trials per participant and balanced classes. Trial "
        "order is randomised, so a single-class block is one trial: this is "
        "the cohort where a covariance update spans the largest number of "
        "blocks. It contributes about 100 fitting windows per participant, the "
        "fewest of the five, which Section~" + BS + "ref{sec:newbaselines} "
        "shows breaks most of the published decoders under shared "
        "hyperparameters.\n\n"
        + BS + "textbf{Cohort E (lower-limb exoskeleton, DECODED).} Seven "
        "participants wearing a powered lower-limb exoskeleton, 27-channel EEG "
        "at 200\\,Hz resampled to 100\\,Hz, from the DECODED subproject of "
        "EUROBENCH. Each run presents 15\\,s standing relaxed with the device "
        "stopped, 24\\,s walking under kinesthetic motor imagery, 22\\,s "
        "walking under a regressive-counting distractor and 14\\,s standing "
        "relaxed. We decode standing relaxed against walking under motor "
        "imagery, the same contrast as cohort A on a second device, and drop "
        "the counting condition rather than folding it into either class, "
        "since it is a distractor rather than a movement-intent state. Windows "
        "are 4\\,s at 0.5\\,s stride and never cross a condition boundary. "
        "Only one participant recorded on two dates, so unlike cohorts~A, C "
        "and~D the split is temporal across runs, the first eight fitting and "
        "the rest testing. This cohort therefore matches best on task and "
        "tests least drift, and Section~" + BS + "ref{sec:limits} says so.\n\n")
    s = s.replace(anchor, new + anchor, 1)

    # ---- the block table, all five cohorts
    i = s.index(BS + "label{tab:blocks}")
    j = s.index(BS + "end{table}", i)
    body = "".join("%s & %d & %d & %d %s\n" % (n, b, t, lg, NL)
                   for n, b, t, lg in BLOCKS)
    tab = (
        BS + "label{tab:blocks}\n"
        + BS + "centering\n"
        + BS + "begin{tabular}{lrrr}\n" + BS + "toprule\n"
        "Cohort & Blocks & $" + BS + "tau_{" + BS + "mathrm{blk}}$ & Longest "
        + NL + "\n" + BS + "midrule\n"
        + body
        + BS + "bottomrule\n" + BS + "end{tabular}\n")
    s = s[:i] + tab + s[j:]

    # ---- the caption needs to say what changed
    old_cap_tail = ("of the protocol and not of the participant: every "
                    "participant of every cohort returns the same median.}")
    new_cap_tail = ("of the protocol and not of the participant: every "
                    "participant within a cohort returns the same median. "
                    "Computed over the fitting recordings only, which is what "
                    "is available when a rate must be chosen. The comparison "
                    "that matters in Section~" + BS + "ref{sec:fivecohort} is "
                    "against $N=32$, the number of windows averaged into one "
                    "covariance update: cohort~B is the only one where a block "
                    "outlasts an update.}")
    s = s.replace(old_cap_tail, new_cap_tail, 1)

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("PASS 6 done: Data section and block table")
    for n, b, t, lg in BLOCKS:
        print("  %-26s blocks %5d  tau_blk %3d  longest %4d  %s"
              % (n, b, t, lg, "AT RISK" if t > 32 else "safe"))


if __name__ == "__main__":
    main()
