r"""How much of the manuscript is still about components the model does not use.

Counts mentions of the alignment layer, the cross-epoch transformer and the
selective gate, per section, so the decision about what to cut is made against
numbers rather than an impression.

A mention is legitimate in three places: the ablation, the dropped-components
method subsection, and the rate condition, which was derived on the alignment
layer and is what explains where the branch fails. Everywhere else is residue
from the previous architecture.
"""
from __future__ import annotations

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92)
TERMS = ["align", "alignment", "gate", "ctx", "cross-epoch", "selective",
         "abstention", "SelectiveNet", "whiten"]
SEC = re.compile(BS + BS + r"(sub)?section\*?\{([^}]*)\}")


def main():
    for f in ("paper/cas_calmnet.tex", "paper/tables_auto.tex"):
        s = io.open(os.path.join(ROOT, f), encoding="utf-8").read()
        marks = [(m.start(), m.group(2)) for m in SEC.finditer(s)]
        if not marks:
            marks = [(0, "(generated tables)")]
        marks.append((len(s), "END"))
        print("=== %s ===" % f)
        rows = []
        for i in range(len(marks) - 1):
            a, name = marks[i]
            b = marks[i + 1][0]
            body = s[a:b]
            n = sum(len(re.findall(r"\b" + t + r"\b", body, re.I))
                    for t in TERMS)
            words = len(body.split())
            if n:
                rows.append((n, words, name))
        for n, w, name in sorted(rows, reverse=True):
            print("  %4d mentions  %6d words  %s"
                  % (n, w, name[:62]))
        print("  %d total\n" % sum(r[0] for r in rows))


if __name__ == "__main__":
    main()
