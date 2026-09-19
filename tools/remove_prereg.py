r"""Remove the cohort-C pre-registration section, the last of the dropped layer.

Both registered predictions manipulated the adaptive alignment layer. P1
compared alignment on against alignment off and P2 changed that layer's
momentum, so neither says anything about the reported model while the section
presented them as the paper's advance test.

What is NOT lost. The falsified stronger prescription, which set the adaptation
rate from the block length, was run on the branch and is reported in
Section~\ref{sec:fivecohort}: at the prescribed memory of 428 windows cohort~B
reaches 0.650 against 0.744 for the stem alone. The abstract's claim to have
pre-registered and falsified it therefore keeps its evidence; only the
references are repointed.

What IS lost, and is removed rather than reworded: the claim that the two-sided
condition was tested prospectively on cohort~C and held. That test was the
alignment comparison.

Every \ref into the removed span is rewritten here. tab:prereg is referenced
only from inside it. Guard: no label inside a removed span may be referenced
from outside it.

    python tools/remove_prereg.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def edit(s, old, new, label):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    n = len(pat.findall(s))
    if n != 1:
        sys.exit("FAILED %s: %d matches" % (label, n))
    print("  ok  " + label)
    return pat.sub(lambda _: new, s, count=1)


def main():
    s = io.open(TEX, encoding="utf-8").read()

    # ---------------------------------------------- repoint, then remove
    s = edit(s,
             r"\item Pre-register a stronger version of the same condition, "
             r"one that sets the adaptation rate from the block length, and "
             r"report it as falsified (Section~\ref{sec:prereg}).",
             r"\item Report as falsified a stronger version of the same "
             r"condition, one that sets the adaptation rate from the block "
             r"length: at the memory it prescribes the failing cohort reaches "
             r"$0.650$ against $0.744$ for the stem alone "
             r"(Section~\ref{sec:fivecohort}).",
             "contribution item")

    s = edit(s,
             r"Its analysis plan was committed to version control before any "
             r"model was trained on it (Section~\ref{sec:prereg}).",
             r"Its analysis plan was committed to version control before any "
             r"model was trained on it.",
             "data section")

    s = edit(s,
             r"as the third cohort shows (Section~\ref{sec:prereg})",
             r"as the block-length comparison shows "
             r"(Section~\ref{sec:fivecohort})",
             "discussion")

    s = edit(s,
             r"the comparison with published decoders on cohort~C "
             r"(Section~\ref{sec:master}) and the cohort~C predictions "
             r"(Section~\ref{sec:prereg}).",
             r"and the comparison with published decoders on cohort~C "
             r"(Section~\ref{sec:master}).",
             "limitations paired-test list")

    s = edit(s,
             r"One prediction from it has since been tested out-of-sample, "
             r"and that test falsified the one-sided form , which is how the "
             r"two-sided form arose. The corrected form has since been tested "
             r"prospectively on cohort~C (Section~\ref{sec:prereg}), where "
             r"both registered predictions held. That test covers the lower "
             r"bound and the contrast with cohort~B; the upper bound has not "
             r"been tested in advance.",
             r"One prediction from it has since been tested out-of-sample, "
             r"and that test falsified the one-sided form, which is how the "
             r"two-sided form arose. A stronger prescription, setting the "
             r"adaptation rate from the block length, was also falsified "
             r"(Section~\ref{sec:fivecohort}). The two-sided form has been "
             r"applied prospectively only to cohorts~D and~E, and its upper "
             r"bound has not been tested in advance.",
             "limitations, hypothesis-generating")

    s = edit(s,
             r"Only its applications to cohorts~D and~E were prospective, and "
             r"of those only cohort~C's original rate prediction was "
             r"registered in the repository before the run.",
             r"Only its applications to cohorts~D and~E were prospective.",
             "limitations, found after the fact")

    # ------------------------------------------------------ remove the span
    a = s.index(BS + "subsection{A prediction registered before a third "
                     "cohort}")
    b = s.index(BS + "subsection{", a + 10)
    span = s[a:b]
    inside = set(re.findall(r"\\label\{([^}]*)\}", span))
    outside = s[:a] + s[b:]
    dangling = sorted(l for l in inside
                      if re.search(r"\\(?:ref|autoref|eqref)\{"
                                   + re.escape(l) + r"\}", outside))
    if dangling:
        sys.exit("REFUSING: %s referenced from outside the span" % dangling)
    s = outside
    print("  ok  removed %d chars, labels %s" % (len(span), sorted(inside)))

    left = re.findall(r"\\ref\{(sec:prereg|tab:prereg)\}", s)
    if left:
        sys.exit("dangling refs remain: %s" % left)

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("\n  no reference to the removed span remains")


if __name__ == "__main__":
    main()
