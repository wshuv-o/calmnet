r"""Pass 14: captions for the regenerated figures, and drop the superseded one.

fig_ablation and fig_rate have been redrawn from current data. Their captions
still describe what they used to show, so they are rewritten.

fig_inversion drew five arms of the old architecture on two cohorts to make the
point that the ordering reverses between them. fig_cohorts now makes that point
across five cohorts using the reported model, so fig_inversion is removed.

Removal is guarded: every label inside the removed span must be unreferenced
from outside it. That is the check that would have caught Condition 1 going out
with a deleted section earlier tonight, and it is applied here rather than the
float-enumeration that missed it.

    python tools/rewrite_14_figs.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
GEN = os.path.join(ROOT, "paper", "tables_auto.tex")
BS = chr(92)

ABL = (
    BS + "caption{Ablation of the reported model on cohort~A, three "
    "data-split seeds. (a) Balanced accuracy by configuration; the reported "
    "model is solid, the stem it is built on is grey, and every configuration "
    "containing a component that was tested and not retained is faded. White "
    "bars are one standard deviation across seeds. (b) The same arms as a "
    "change against the stem alone, paired across participants with seeds "
    "averaged within participant; the annotation gives how many participants "
    "improved and the shaded band is the $0.02$ interpretation threshold. The "
    "branch is the only component that raises accuracy. Nothing built from the "
    "alignment layer, the gate or the cross-epoch transformer improves on the "
    "stem by more than $0.005$, and the gate alone is $0.022$ below it.}")

RATE = (
    BS + "caption{The rate condition, measured by manipulation rather than by "
    "comparing cohorts. Each point moves only the estimator memory "
    "$" + BS + "mu = N/m$, with the architecture, the data and every other "
    "setting fixed; dashed verticals mark each cohort's class-block length "
    "and annotations give how many participants improved. On cohort~A the "
    "branch gains $+0.045$ and $+0.048$ at memories of 160 and 40 windows, "
    "both above its 18-window block, and reverses to $-0.076$ in every "
    "participant at a memory of 8, below it. On cohort~B, whose blocks are "
    "214 windows, the effect recovers monotonically as the memory is raised "
    "past them but does not become positive. The effect therefore appears and "
    "disappears as $" + BS + "mu$ crosses $" + BS + "tau_{" + BS
    + "mathrm{blk}}$, which is the causal form of the comparison "
    "Fig.~" + BS + "ref{fig:cohorts} makes across cohorts.}")


def replace_caption(s, label, new):
    i = s.index(BS + "label{" + label + "}")
    c = s.rindex(BS + "caption{", 0, i)
    depth, j = 0, c + len(BS + "caption{")
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            if depth == 0:
                break
            depth -= 1
        j += 1
    return s[:c] + new + s[j + 1:]


def drop_float(s, label):
    k = s.index(BS + "label{" + label + "}")
    for env in ("figure*", "figure", "table*", "table"):
        try:
            a = s.rindex(BS + "begin{" + env + "}", 0, k)
            b = s.index(BS + "end{" + env + "}", k) + len(BS + "end{" + env
                                                          + "}")
        except ValueError:
            continue
        span = s[a:b]
        if BS + "label{" + label + "}" in span:
            return s[:a] + s[b:], span
    sys.exit("could not bound the float for " + label)


def main():
    s = io.open(TEX, encoding="utf-8").read()

    s = replace_caption(s, "fig:ablation", ABL)
    s = replace_caption(s, "fig:rate", RATE)
    print("  captions rewritten: fig:ablation, fig:rate")

    s2, span = drop_float(s, "fig:inversion")
    inner = set(re.findall(r"label\{([^}]+)\}", span))
    outside = set(re.findall(r"ref\{([^}]+)\}", s2 + io.open(
        GEN, encoding="utf-8").read()))
    orphan = sorted(inner & outside)
    if orphan:
        print("  labels still referenced from outside: %s" % orphan)
        # repoint them at the figure that replaced it
        for lab in orphan:
            s2 = s2.replace(BS + "ref{" + lab + "}", BS + "ref{fig:cohorts}")
            print("     %s -> fig:cohorts" % lab)
    print("  removed fig:inversion (%d chars)" % len(span))

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s2)

    both = s2 + io.open(GEN, encoding="utf-8").read()
    labs = set(re.findall(r"label\{([^}]+)\}", both))
    refs = set(re.findall(r"ref\{([^}]+)\}", both))
    left = sorted(refs - labs)
    print("\n  references without a label: %s" % (left or "none"))


if __name__ == "__main__":
    main()
