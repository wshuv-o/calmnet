r"""Pass 10 of the rewrite: Limitations and Conclusion.

The Limitations section was written for the old architecture. It discusses the
alignment effect's seed behaviour, counts three cohorts, and omits every
limitation the reported model introduces. The stale entries are replaced and six
new ones added, each stating what would have to be done to remove it.

The Conclusion is rewritten to be forward-looking rather than a summary, per the
project's writing rules, and to name the reported model rather than the one it
replaced.

    python tools/rewrite_10_limits.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def main():
    s = io.open(TEX, encoding="utf-8").read()

    # ---------- replace the stale "Seeds" entry
    i = s.index(BS + "textbf{Seeds.}")
    j = s.index(BS + "textbf{No paired significance tests", i)
    seeds = (
        BS + "textbf{Seeds and effect size.} Every cohort in "
        "Table~" + BS + "ref{tab:master} is three data-split seeds. The effect "
        "of the branch is nevertheless modest in absolute terms, $+0.021$ to "
        "$+0.064$ where it helps, and on cohort~A the paired bootstrap "
        "interval spans zero ($[-0.020, +0.109]$, $p=0.22$ at seven "
        "participants). The cohorts that carry the claim are C, with twenty "
        "participants and all twenty improving, and D and E, where the "
        "intervals exclude zero. A reader who wants a single number should "
        "take cohort~C's $+0.064$ and treat cohort~A as consistent with it "
        "rather than as independent support.\n\n")
    s = s[:i] + seeds + s[j:]

    # ---------- cohort size
    s = s.replace(
        BS + "textbf{Cohort size.} Seven, eight and twenty participants.",
        BS + "textbf{Cohort size.} Seven, eight, twenty, nine and seven "
        "participants, 51 in total.", 1)

    # ---------- new limitations, before the Conclusion
    anchor = "%" + "=" * 77 + "\n" + BS + "section{Conclusion}"
    new = (
        BS + "textbf{Parameter cost.} The branch is 238{,}028 parameters and "
        "the fusion it requires a further 32{,}896, against 19{,}505 in the "
        "stem path. The reported model is therefore about fifteen times the "
        "size of the stem it improves on, for $0.02$ to $0.06$ of balanced "
        "accuracy. Two of the published decoders it beats are an order of "
        "magnitude smaller again. Whether the tangent vector can be projected "
        "more cheaply, for instance by reducing its dimension before the "
        "linear layer, is not tested here and is the obvious first place to "
        "look.\n\n"
        + BS + "textbf{The block-length boundary was found after the fact.} "
        "The comparison between $" + BS + "tau_{" + BS + "mathrm{blk}}$ and "
        "the update size separates all five cohorts with nothing fitted, but "
        "it was identified after cohorts~A, B and~C had been run. Only its "
        "applications to cohorts~D and~E were prospective, and of those only "
        "cohort~C's original rate prediction was registered in the repository "
        "before the run. The boundary is supported, not established, and the "
        "test that would establish it is a cohort chosen in advance because "
        "its protocol sits near the boundary.\n\n"
        + BS + "textbf{The boundary is untested at the boundary.} Every cohort "
        "where the branch helps has a block of 1 to 18 windows and the one "
        "where it fails has 214, against an update of 32. Nothing is measured "
        "between 18 and 214, so the location of the change in behaviour is "
        "inferred from five points that straddle it, not measured.\n\n"
        + BS + "textbf{No calibrated threshold.} Two manipulations move the "
        "effect in the predicted direction and neither restores parity. "
        "Setting the adaptation rate from the block length with a fixed margin "
        "of two was pre-registered and falsified. Enlarging the update to 256 "
        "windows against a 214-window block recovers 74\\,\\% of the loss and "
        "still costs $0.046$. The condition therefore gives an ordering and a "
        "direction, not a rate a practitioner could set, and the upper side of "
        "the two-sided condition remains unmeasured on both cohorts where it "
        "was attempted.\n\n"
        + BS + "textbf{Cohort E tests task, not drift.} Only one participant "
        "recorded on two dates, so its split is temporal across runs within a "
        "session. It is the closest match in the paper to the target "
        "application and the weakest test of longitudinal shift, and those two "
        "facts are in tension.\n\n"
        + BS + "textbf{Reproducibility across machines.} The cohort~D baseline "
        "sweep was run independently on two machines. Sixteen of 24 cells "
        "differ, by up to $0.022$ on a single cell, while the three-seed arm "
        "means agree to $0.003$ and the strongest published decoder is "
        "identical on both. Single-cell numbers anywhere in this paper "
        "therefore carry cross-machine variation wider than several of the "
        "gaps between neighbouring rows, and only seed-averaged quantities "
        "should be compared.\n\n")
    s = s.replace(anchor, new + anchor, 1)

    # ---------- conclusion
    i = s.index(BS + "section{Conclusion}")
    j = len(s)
    tail = s[i:]
    end = tail.find(BS + "end{document}")
    concl = (
        BS + "section{Conclusion}\n\n"
        "A log-power stem reads the diagonal of a window's spatial covariance "
        "and discards the rest of it. Supplying that structure through a "
        "tangent-space branch raises balanced accuracy on four of five cohorts "
        "and 51 participants, and beats eight published decoders trained in "
        "the same pipeline. It fails on the fifth, and the protocol says in "
        "advance which case a cohort is: the branch's reference is an average "
        "over whatever is currently streaming, so it survives only while one "
        "covariance update spans more than one class block.\n\n"
        "That condition is the part worth building on, and it is also the part "
        "least finished. It separates five cohorts with nothing fitted, it "
        "predicted the sign on two cohorts prospectively, and enlarging the "
        "update recovers most of the loss where it is violated. It still "
        "yields no rate a practitioner could set, its upper side has never "
        "been measured, and no cohort has been run near the boundary itself. "
        "The experiment that would settle it is small and specific: choose a "
        "protocol whose class blocks fall between 18 and 214 windows, register "
        "the prediction, and run it once.\n\n"
        "The wider point is that a label-free adaptive statistic inherits the "
        "structure of the protocol it adapts on. Test-time adaptation methods "
        "are usually evaluated on streams that are shuffled or assumed "
        "i.i.d.; EEG protocols are blocked by design, and the block length is "
        "known before a single recording is made. Reporting it alongside "
        "accuracy would let a reader see, without running anything, whether an "
        "adaptive component is operating inside its domain of validity.\n\n")
    s = s[:i] + concl + tail[end:]

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("PASS 10 done: Limitations and Conclusion")
    print("  - stale seeds entry replaced; effect size stated honestly")
    print("  - cohort count 3 -> 5, 51 participants")
    print("  + parameter cost, post-hoc boundary, untested at the boundary,")
    print("    no calibrated threshold, cohort E tests task not drift,")
    print("    cross-machine reproducibility")
    print("  - conclusion rewritten forward-looking, names the reported model")


if __name__ == "__main__":
    main()
