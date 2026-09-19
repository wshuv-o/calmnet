r"""Replace the N framing with mu = N/m throughout.

I wrote the boundary as the class-block length against N, the windows in one
covariance update, because at the default N=32 that separated all five cohorts.
It separated them only because no cohort has a block between 32 and 160. The
2060's cohort A runs disambiguate: at N=8, m=0.2 the memory is 40, still above
the block of 18, and the gain survives; at N=8, m=1.0 the memory is 8 and the
effect reverses. The governing quantity is mu = N/m, which is how Condition 1
states it.

Every sentence that named N or "one covariance update" as the threshold is
rewritten. The figure boundary moves from 32 to 160.

    python tools/fix_mu_framing.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
FIG = os.path.join(ROOT, "src", "fig_cohorts.py")
BS = chr(92)

TEX_SUBS = [
    ("Its label-free reference fails where a class block outlasts one "
     "covariance update.",
     "Its label-free reference fails where a class block outlasts the "
     "estimator memory."),
    ("Block length against update size, read from the protocol, separates "
     "all five.",
     "Block length against estimator memory, read from the protocol, "
     "separates all five."),
    ("so it is safe only while one covariance update spans more than one "
     "class block; on the cohort where a block outlasts an update the branch "
     "costs",
     "so it is safe only while its memory outlasts a class block; on the "
     "cohort where a block outlasts that memory the branch costs"),
    ("A single-class block there lasts 214 windows while one covariance "
     "update averages 32, so the update sits inside a block and the reference "
     "converges on the streaming class.",
     "A single-class block there lasts 214 windows while the estimator memory "
     "is 160, so a block outlasts the memory and the reference converges on "
     "the streaming class."),
    ("On the four cohorts where the branch helps a block is 1 to 18 windows "
     "and each update is class-mixed by construction.",
     "On the four cohorts where the branch helps a block is 1 to 18 windows "
     "and the memory spans many blocks."),
    ("Comparing block length against update size, both read from the protocol "
     "before training and neither fitted, separates all five cohorts.",
     "Comparing the block length against the memory $" + BS + "mu = N/m$, the "
     "block length read from the protocol and the memory fixed by the "
     "optimiser settings, separates all five cohorts."),
    ("A covariance update averages $N=32$ windows, so a cohort is at risk "
     "when a block outlasts an update.",
     "The estimator memory is $" + BS + "mu = N/m = 160$ windows at the "
     "settings used throughout, so a cohort is at risk when a block outlasts "
     "that memory."),
    ("The comparison between $" + BS + "tau_{" + BS + "mathrm{blk}}$ and the "
     "update size separates all five cohorts with nothing fitted",
     "The comparison between $" + BS + "tau_{" + BS + "mathrm{blk}}$ and the "
     "estimator memory separates all five cohorts with nothing fitted"),
    ("nothing is measured near $N=32$ where the behaviour is claimed to "
     "change.",
     "no cohort has a block between 32 and 160 windows, which is why the "
     "block length appeared for a time to be competing with $N$ rather than "
     "with $" + BS + "mu$ (Table~" + BS + "ref{tab:batchsweep})."),
    ("Nothing is measured between 18 and 214, so the location of the change "
     "in behaviour is inferred from five points that straddle it, not "
     "measured.",
     "Nothing is measured between 18 and 214, so across cohorts the location "
     "of the change is inferred from five points that straddle it. Within "
     "cohorts it is measured directly: varying $" + BS + "mu$ at fixed block "
     "length crosses the boundary in both directions "
     "(Table~" + BS + "ref{tab:batchsweep})."),
]

FIG_SUBS = [
    ("NBATCH = 32", "MU = 160          # estimator memory N/m at the settings"
     " used throughout"),
    ("ax2.axvspan(0.6, NBATCH,", "ax2.axvspan(0.6, MU,"),
    ("ax2.axvline(NBATCH,", "ax2.axvline(MU,"),
    ("S.note(ax2, NBATCH * 0.82, 0.085,\n"
     '           "one covariance\\nupdate, $N=32$", ha="right", fontsize=6.6,',
     "S.note(ax2, MU * 0.82, 0.085,\n"
     '           "estimator memory\\n$\\\\mu = N/m = 160$", ha="right", '
     'fontsize=6.6,'),
    ("with the number of windows averaged into one covariance "
     "update drawn as a dashed line",
     "with the estimator memory drawn as a dashed line"),
]


def apply(path, subs, label):
    s = io.open(path, encoding="utf-8").read()
    hit = miss = 0
    for old, new in subs:
        if old in s:
            s = s.replace(old, new)
            hit += 1
        else:
            miss += 1
            print("   [%s] no match: %s..." % (label, old[:56]))
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("  %s: %d applied, %d unmatched" % (label, hit, miss))


def main():
    apply(TEX, TEX_SUBS, "manuscript")
    apply(FIG, FIG_SUBS, "figure")


if __name__ == "__main__":
    main()
