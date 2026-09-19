r"""Finish the mu correction: remaining phrases, the abstract, the limitation.

The 2060's cohort A rows measure the boundary by manipulation rather than by
comparing cohorts: mu = 160 and 40 both keep the gain, mu = 8 reverses it in
every participant. That does three things to the text.

It replaces the last sentences that named N or "one covariance update" as the
threshold. It earns a sentence in the abstract, because switching the effect off
and on by moving one number is stronger evidence than five cohorts agreeing. And
it closes the limitation saying the boundary was untested at the boundary: mu =
8, 40 and 160 straddle a block of 18 from both sides on one cohort with nothing
else changing.

Lines 1157 and the tab:batchsweep caption are left alone; they already say mu.
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
FIG = os.path.join(ROOT, "src", "fig_cohorts.py")
BS = chr(92)

SUBS = [
    ("cohort~B is the cohort whose class blocks outlast one covariance "
     "update, so that estimate converges on the streaming class",
     "cohort~B is the cohort whose class blocks outlast the estimator memory, "
     "so that estimate converges on the streaming class"),
    ("that block length, with the number of windows averaged into one "
     "covariance update drawn as a dashed line",
     "that block length, with the estimator memory drawn as a dashed line"),
    ("the block length is read from the labels and the update size is the "
     "batch size, so the separation uses no accuracy.}",
     "the block length is read from the labels and the memory is fixed by the "
     "optimiser settings, so the separation uses no accuracy. "
     "Table~" + BS + "ref{tab:batchsweep} moves the memory directly and "
     "crosses this boundary in both directions on one cohort.}"),
    ("that is exactly the cohort where a class block outlasts one covariance "
     "update.",
     "that is exactly the cohort where a class block outlasts the estimator "
     "memory."),
    ("so it survives only while one covariance update spans more than one "
     "class block.",
     "so it survives only while its memory outlasts a class block."),
    # ---- the limitation this closes
    ("Every cohort where the branch helps has a block of 1 to 18 windows and "
     "the one where it fails has 214,",
     "Across cohorts the boundary is straddled but not located: every cohort "
     "where the branch helps has a block of 1 to 18 windows and the one where "
     "it fails has 214,"),
    # ---- abstract
    ("Enlarging the update to span a block recovers 74",
     "Moving the memory across the block length on a single cohort, with "
     "nothing else changed, switches the effect off and on: it is $+0.045$ at "
     "a memory of 160 windows, $+0.048$ at 40, and $-0.076$ at 8, worse in "
     "every participant. Enlarging the memory on the failing cohort recovers "
     "74"),
]

FIGDOC = [
    ("the same numbers against that block length, with the covariance update "
     "size\ndrawn as a vertical line: the branch helps on every cohort whose "
     "blocks are\nshorter than one update and fails on the only one whose "
     "blocks are longer.",
     "the same numbers against that block length, with the estimator memory\n"
     "mu = N/m drawn as a vertical line: the branch helps on every cohort "
     "whose\nblocks are shorter than the memory and fails on the only one "
     "whose blocks\nare longer."),
]


def apply(path, subs, label):
    s = io.open(path, encoding="utf-8").read()
    ok = bad = 0
    for old, new in subs:
        if old in s:
            s = s.replace(old, new, 1)
            ok += 1
        else:
            bad += 1
            print("   [%s] NOT FOUND: %s..." % (label, old[:58]))
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("  %s: %d applied, %d missed" % (label, ok, bad))
    return s


def main():
    s = apply(TEX, SUBS, "manuscript")
    apply(FIG, FIGDOC, "figure docstring")
    print("\n  residual N-framing:")
    left = [(p, s.count(p)) for p in
            ("one covariance update", "update size", "$N=32$")
            if s.count(p)]
    for p, n in left:
        print("     %-24s %d" % (p, n))
    if not left:
        print("     none")


if __name__ == "__main__":
    main()
