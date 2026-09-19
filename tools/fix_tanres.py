r"""Clear the last references to the alignment layer, and one inconsistency.

1. "The same rate that makes the alignment layer safe makes the reference safe"
   and the closing paragraph comparing the branch against the aligned arm are
   the last two places the dropped layer appears outside the related-work
   discussion of Euclidean and Riemannian alignment in the literature. With the
   layer described nowhere in the paper, both read as references to something
   the reader has never met.

2. tab:tangent's caption calls a memory of 3200 windows "the rate
   Condition~1 prescribes". Condition~1 prescribes a range, not a rate, and the
   stronger form that did prescribe one (a margin of two, giving 428 windows)
   is reported as falsified: it reaches 0.650 against 0.744 for the stem. The
   caption is reworded to say what was actually run.

    python tools/fix_tanres.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")


def edit(s, old, new, label):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    n = len(pat.findall(s))
    if n != 1:
        sys.exit("FAILED %s: %d matches" % (label, n))
    print("  ok  " + label)
    return pat.sub(lambda _: new, s, count=1)


def main():
    s = io.open(TEX, encoding="utf-8").read()

    s = edit(s,
             r"indistinguishable from the stem ($p=0.31$). The same rate that "
             r"makes the alignment layer safe makes the reference safe, on a "
             r"mechanism introduced independently of it.",
             r"indistinguishable from the stem ($p=0.31$).",
             "rate-transfer sentence")

    s = edit(s,
             r"One boundary on the claim. The branch helps the stem, and it "
             r"does not help the configuration that already contains the "
             r"alignment layer: on cohort~C the same branch added to the "
             r"aligned arm changes accuracy by $+0.002$ ($p=0.52$, higher in "
             r"14 of 20). Both read the same second-order structure, so they "
             r"are substitutes rather than complements, and the configuration "
             r"we report is the stem with the branch and no alignment layer.",
             "",
             "substitutes paragraph")

    s = edit(s,
             r"Cohort~B is shown at the default adaptation rate and at the "
             r"rate Condition~\ref{prop:band} prescribes for its 214-window "
             r"class blocks.",
             r"Cohort~B is shown at the default adaptation rate, whose "
             r"160-window memory is shorter than its 214-window class blocks, "
             r"and at a memory of 3200 windows, which is far longer than "
             r"them.",
             "tab:tangent caption")

    s = re.sub(r"\n{3,}", "\n\n", s)
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    left = len(re.findall(r"(?i)alignment layer|aligned arm", s))
    print("\n  'alignment layer' / 'aligned arm' remaining: %d" % left)


if __name__ == "__main__":
    main()
