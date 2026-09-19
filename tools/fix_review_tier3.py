r"""Make the abstract, highlights and results as careful as the limitations.

Every item here is a claim the manuscript's own limitations section already
qualifies, stated more strongly elsewhere. Nothing is deleted; the wording is
brought into line with the evidence.

  discards          a spatial-filter bank already sees off-diagonal structure
                    through w'Cw, so "discards" is wrong wherever it appears
  held out          cohort C was held out from architecture and pipeline
                    development, but its result contributed to revising the
                    rate condition, so it was not held out from every design
                    decision in the final manuscript
  pre-registered    the record is a commit in version control before analysis,
                    not a public registration with a timestamped record
  leads all eight   true within this harness, with no per-model tuning, and
                    several baselines fail to train; say that
  nine times        three seeds support "lower seed-to-seed variability", not
                    a reproducibility ratio
  detection latency declared as an outcome measure but never reported; the
                    inference-cost table measures computation, not detection

    python tools/fix_review_tier3.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def edit(s, old, new, label, count=1):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    n = len(pat.findall(s))
    if n != count:
        sys.exit("FAILED %s: %d matches, expected %d" % (label, n, count))
    print("  ok  %s (%d)" % (label, n))
    return pat.sub(lambda _: new, s, count=count)


def main():
    s = io.open(TEX, encoding="utf-8").read()

    # --------------------------------------------------- "discards" wording
    s = edit(s, "A tangent-space branch supplies the channel covariance a "
                "log-power stem discards.",
             "A tangent-space branch represents the channel covariance a "
             "log-power stem leaves implicit.", "highlight, discards")
    s = edit(s, "which is what a log-power stem discards",
             "which a log-power stem represents only implicitly",
             "fig:activations text, discards")

    # ------------------------------------------------------------ held out
    s = edit(s, "held out from every design decision",
             "held out from architecture and pipeline development",
             "held out, three places", count=3)

    # -------------------------------------------------------- pre-registered
    # the limitations instance carries "with a fixed margin of two", so it is
    # replaced first and the remaining instance is the abstract's
    s = edit(s, "with a fixed margin of two was pre-registered and falsified.",
             "with a fixed margin of two was pre-specified in version control "
             "and falsified.", "limitations, pre-registered")
    s = edit(s, "was pre-registered and falsified.",
             "was pre-specified, committed to version control before the run, "
             "and falsified.", "abstract, pre-registered")

    # ----------------------------------------------------- leads all eight
    s = edit(s, "On those four it leads all eight published decoders trained "
                "in the same pipeline.",
             "On those four it has the highest mean balanced accuracy among "
             "eight published decoders trained in the same pipeline, with no "
             "per-model tuning.", "abstract, leads all eight")

    # -------------------------------------------------------- nine times
    s = edit(s, "so adding the branch makes the model roughly nine times more "
                "reproducible on this cohort.",
             "so across the three seeds evaluated here the branch shows "
             "substantially lower seed-to-seed variability on this cohort.",
             "nine times more reproducible")

    # --------------------------------------------------- detection latency
    s = edit(s, "spurious activations per minute of standing; and detection "
                "latency, reported alongside the benefit it buys.",
             "and spurious activations per minute of standing. Detection "
             "latency is not reported: the inference cost of "
             "Section~" + BS + "ref{sec:cost} measures computation, which is "
             "not the same quantity, and the model's 4" + BS + ",s windows set "
             "a floor on detection that this evaluation does not resolve.",
             "detection latency declared but not reported")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)


if __name__ == "__main__":
    main()
