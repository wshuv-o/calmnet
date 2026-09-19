r"""Finish the mu framing fix: the five sentences whose wrapping differed.

The first pass matched on text I had composed rather than on what the file
contains after LaTeX line wrapping. These anchors are taken verbatim from the
file, and each substitution is verified present before and absent after.
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)

SUBS = [
    ("where the branch helps a block is 1 to 18 windows and every update is "
     "class-mixed by construction. Both quantities are read from the "
     "recording protocol, neither is fitted,",
     "where the branch helps a block is 1 to 18 windows and the memory spans "
     "many blocks. Both quantities are fixed before training, the block "
     "length by the protocol and the memory by the optimiser settings,"),
    ("One covariance update averages $N=32$ windows, so a cohort is at risk "
     "when a block outlasts an update.",
     "The estimator memory is $" + BS + "mu = N/m = 160$ windows at the "
     "settings used throughout, so a cohort is at risk when a block outlasts "
     "that memory."),
    ("and nothing is measured near $N=32$ where behaviour is claimed to "
     "change. Table~" + BS + "ref{tab:batchsweep} is reserved for t",
     "and no cohort has a block between 32 and 160 windows, which is why the "
     "block length appeared for a time to compete with $N$ rather than with "
     "$" + BS + "mu$. Table~" + BS + "ref{tab:batchsweep} settles it by t"),
]


def main():
    s = io.open(TEX, encoding="utf-8").read()
    for old, new in SUBS:
        if old not in s:
            print("  NOT FOUND: %s..." % old[:60])
            continue
        s = s.replace(old, new, 1)
        print("  replaced: %s..." % old[:60])
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    left = []
    for bad in ["$N=32$", "one covariance update", "One covariance update",
                "update size", "outlasts an update"]:
        n = s.count(bad)
        if n:
            left.append((bad, n))
    print("\n  remaining N-framing phrases:")
    for b, n in left:
        print("     %-26s %d" % (b, n))
    if not left:
        print("     none")


if __name__ == "__main__":
    main()
