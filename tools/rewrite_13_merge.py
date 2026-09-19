r"""Pass 13: merge three superseded Results sections into one.

"Comparison with published decoders" is superseded by Table 3, which now covers
the reported model against the same decoders on five cohorts instead of the
align-plus-gate configuration on three. "External validation and the inverted
ordering" and "Slow adaptation has a cost" both argue about arms that are not in
the model. Together they are about 2,000 words.

They are replaced by one section that keeps what the newer material does not
already carry: the ordering inversion, which is the same phenomenon the branch
shows on cohort B, and the cost of slowing adaptation where class blocks are
short, which is the evidence that the condition has two sides.

Every float inside the removed span is checked: tab:external and fig:inversion
are carried into the new section verbatim, and the script aborts if any other
float would be lost. The Conclusion rewrite earlier tonight deleted the
bibliography by not doing this.

    python tools/rewrite_13_merge.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)

CUT = ["Comparison with published decoders",
       "External validation and the inverted ordering",
       "Slow adaptation has a cost"]
FLOAT = re.compile(BS + BS + r"begin\{(table\*?|figure\*?)\}.*?"
                   + BS + BS + r"end\{\1\}", re.S)
SEC = re.compile(BS + BS + r"(sub)?section\*?\{([^}]*)\}")

NEW = (
    BS + "subsection{What the alignment layer did, and why it is not "
    "retained}\n"
    + BS + "label{sec:alignfindings}\n\n"
    "The alignment layer is not in the reported model, and the measurements "
    "that removed it are worth one section because they are the same "
    "phenomenon the branch shows.\n\n"
    + BS + "paragraph{The ordering inverts between cohorts} On cohort~A the "
    "arms that use alignment lead and the alignment-free arms trail; on "
    "cohort~B the order reverses completely (Table~" + BS + "ref{tab:external}, "
    "Fig.~" + BS + "ref{fig:inversion}). With alignment enabled at the default "
    "rate the configuration averages $0.580$ on cohort~B, below every "
    "published decoder in Table~" + BS + "ref{tab:compare}, while the "
    "alignment-free arm reaches $0.792$. This is not a transfer failure of a "
    "particular architecture. It is the same reversal the branch shows on the "
    "same cohort, for the same reason: both mechanisms divide by a running "
    "covariance, and cohort~B is the cohort whose class blocks outlast one "
    "covariance update, so that estimate converges on the streaming class "
    "(Section~" + BS + "ref{sec:fivecohort}).\n\n"
    + BS + "paragraph{Slowing the estimator has a cost where blocks are "
    "short} Lengthening the estimator memory from 160 to 3200 windows recovers "
    "cohort~B, and costs accuracy on both cohorts whose blocks are shorter "
    "than one update: $-0.050$ on cohort~A and $-0.016$ on cohort~C, the "
    "latter lower in 15 of 20 participants ($p=0.003$) and negative on all "
    "three seeds ($-0.085$, $-0.056$, $-0.008$). A rate cannot therefore be "
    "chosen once for all protocols, which is what makes the condition "
    "two-sided in principle even though only its lower side is measured here "
    "(Section~" + BS + "ref{sec:limits}).\n\n"
    + BS + "paragraph{Why it is not retained} Added to the stem, the layer "
    "changes accuracy by $+0.004$ on cohort~A, inside the interpretation "
    "threshold (Table~" + BS + "ref{tab:ablation_rep}). Added on top of the "
    "branch it costs $0.032$, and on cohort~C the same addition is worth "
    "$+0.002$ ($p=0.52$). Both mechanisms read the same second-order "
    "structure, so they are substitutes, and the branch is the one that "
    "raises accuracy.\n\n")


def main():
    s = io.open(TEX, encoding="utf-8").read()
    marks = [(m.start(), m.group(2)) for m in SEC.finditer(s)]
    marks.append((len(s), "END"))

    spans = []
    for i in range(len(marks) - 1):
        a, name = marks[i]
        if name in CUT:
            spans.append((a, marks[i + 1][0], name))
    if len(spans) != len(CUT):
        sys.exit("expected %d sections, found %d" % (len(CUT), len(spans)))

    # floats inside the removed span must be accounted for
    keep_labels = {"tab:external", "fig:inversion"}
    carried, lost = [], []
    for a, b, name in spans:
        for m in FLOAT.finditer(s[a:b]):
            lm = re.search(BS + BS + r"label\{([^}]+)\}", m.group(0))
            lab = lm.group(1) if lm else "(no label)"
            (carried if lab in keep_labels else lost).append((lab, m.group(0)))
    if lost:
        print("floats that would be lost:")
        for lab, _ in lost:
            print("   %s" % lab)
        sys.exit("refusing to drop a float; add it to keep_labels or "
                 "confirm it is redundant")

    removed = sum(b - a for a, b, _ in spans)
    body = NEW + "\n".join(t for _, t in carried) + "\n\n"

    out, prev = [], 0
    for a, b, _ in sorted(spans):
        out.append(s[prev:a])
        prev = b
    out.append(s[prev:])
    s2 = "".join(out)

    # insert the merged section where the first removed one began
    anchor = spans[0][0]
    shift = 0
    for a, b, _ in sorted(spans):
        if a < anchor:
            shift += b - a
    s2 = s2[:anchor - shift] + body + s2[anchor - shift:]

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s2)
    print("removed %d chars across %d sections" % (removed, len(spans)))
    print("carried floats: %s" % ", ".join(l for l, _ in carried))
    print("net change: %+d chars" % (len(s2) - len(s)))


if __name__ == "__main__":
    main()
