r"""Second prep step: stop generating the tables that archive old architectures.

tab:ablation3, tab:rate, tab:ratecurve and tab:drift compare configurations
built from the alignment layer, the gate and the cross-epoch transformer. Per
the audit they go. They are dropped from the generator's table list rather than
from its output, so regenerating does not bring them back, and the functions
that build them are left in the file untouched: nothing is deleted from the
repository, only from the manuscript.

The marker sentence that pointed at "components tested and not retained" is also
removed, since that section is going and no table should reference it.

Two remaining citing sentences in the manuscript are repaired here as well.

    python tools/cleanup_prep2.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "make_tables.py")
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)

TEX_SUBS = [
    ("decoding by $+0.005$ and $-0.016$ (Section~" + BS + "ref{sec:rejected}).",
     "decoding by $+0.005$ and $-0.016$."),
    ("same pipeline as Table~" + BS + "ref{tab:external}",
     "same pipeline as Table~" + BS + "ref{tab:compare}"),
]


def main():
    s = io.open(SRC, encoding="utf-8").read()

    old_list = ("             t_compare(), t_ablation3(), t_rate(), "
                "t_ratecurve(), t_repr(), t_norm(),\n"
                "             t_drift(),\n"
                "             t_noise()]")
    new_list = ("             # t_ablation3, t_rate, t_ratecurve and t_drift "
                "compare configurations\n"
                "             # built from components that are not in the "
                "reported model. They are\n"
                "             # no longer emitted; the functions stay in this "
                "file so nothing is\n"
                "             # lost from the repository.\n"
                "             t_compare(), t_repr(), t_norm(), t_noise()]")
    if old_list in s:
        s = s.replace(old_list, new_list, 1)
        print("  generator now emits 4 tables instead of 8")
    else:
        print("  WARNING: table list not matched")

    # the marker pointed at a section that is being removed
    i = s.find("DROPPED_ARM_TABLES = {")
    if i >= 0:
        j = s.index("def table(", i)
        s = s[:i] + ("# No table in the manuscript now reports an arm built "
                     "from a component\n# outside the proposed model, so the "
                     "marker that named them is gone.\n\n\n") + s[j:]
        s = s.replace("    if label in DROPPED_ARM_TABLES:\n"
                      "        caption = caption + DROPPED_NOTE\n", "")
        print("  removed the dropped-arm marker from the generator")
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)

    t = io.open(TEX, encoding="utf-8").read()
    for old, new in TEX_SUBS:
        if old in t:
            t = t.replace(old, new)
            print("  repaired: %s..." % old[:54])
        else:
            print("  no match: %s..." % old[:54])
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(t)


if __name__ == "__main__":
    main()
