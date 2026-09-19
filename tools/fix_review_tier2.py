r"""Fix the two claims a reviewer can disprove from our own tables.

1. "the branch helps wherever mu exceeds the block length and fails wherever
   it does not". Table 12 says otherwise for cohort B: crossing the boundary
   takes the effect from -0.175 to -0.046 and then -0.026. It never becomes
   positive. The condition orders and bounds the damage; it is not sufficient
   for a gain, and the Limitations section already said so while the Results
   section did not.

2. "log power per channel ... the off-diagonal entries, how channels covary,
   are discarded". A learned spatial filter w has output power w' C w, which
   contains the cross terms w_i w_j C_ij, so a spatial-filter bank is not
   blind to off-diagonal covariance. What the stem lacks is an explicit
   representation of the full covariance; what the branch adds is that
   representation in the tangent space. The weaker statement is the one the
   evidence supports, and it is still the paper's point.

Also corrects the abstract, which described the safety condition in terms of
one covariance update spanning a block. The paper's own contribution, and the
N=8 row of Table 12, is that the effective memory mu = N/m governs rather than
the update size N.

    python tools/fix_review_tier2.py
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

    # -------------------------------------- 1. what the sweep actually shows
    s = edit(s,
             "The branch helps wherever $" + BS + "mu$ exceeds the block "
             "length and fails wherever it does not, on both cohorts and "
             "across a 400-fold range of $" + BS + "mu$.",
             "Across a 400-fold range of $" + BS + "mu$ on two cohorts, the "
             "sign and the size of the effect track the comparison. On "
             "cohort~A, where $" + BS + "mu$ exceeds the block length, the "
             "branch gains, and driving $" + BS + "mu$ below it reverses the "
             "effect to $-0.076$. On cohort~B, raising $" + BS + "mu$ above "
             "the block length takes the cost from $0.175$ to $0.046$ and "
             "then to $0.026$, so crossing the boundary bounds the damage "
             "without turning it into a gain. The comparison therefore "
             "predicts the direction and the severity of the failure; it is "
             "not sufficient for a positive effect, and Section~"
             + BS + "ref{sec:limits} says what that leaves open.",
             "rate claim, Results")

    # ----------------------------------- 2. what a log-power stem can see
    s = edit(s,
             "That summary is the diagonal of the window's spatial covariance. "
             "The off-diagonal entries, how channels covary, are discarded, "
             "and covariance structure is the basis of the Riemannian methods "
             "that remain competitive on movement-intent tasks",
             "That summary is the diagonal of the covariance of the filtered "
             "signal. A learned spatial filter $" + BS + "mathbf{w}$ has "
             "output power $" + BS + "mathbf{w}^" + BS + "top " + BS
             + "mathbf{C}" + BS + "mathbf{w}$, so the stem is not blind to "
             "off-diagonal structure; it represents that structure implicitly, "
             "through whichever projections the filters happen to learn, and "
             "never holds the full covariance. Covariance structure is the "
             "basis of the Riemannian methods that remain competitive on "
             "movement-intent tasks",
             "what the stem can see, Introduction")

    # --------------------------------------------- 3. memory, not update size
    s = edit(s,
             "The reference is a running mean over incoming windows, safe only "
             "while one covariance update spans more than one class block.",
             "The reference is a running mean over incoming windows, safe only "
             "while its effective memory $" + BS + "mu=N/m$ outlasts a class "
             "block.",
             "abstract, memory not update size")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)


if __name__ == "__main__":
    main()
