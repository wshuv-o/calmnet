r"""Audit every table and figure for language describing dropped components.

The reported model is the stem plus the tangent branch. Any float whose caption
or body still names the alignment layer, the gate, the cross-epoch transformer
or the selective head as part of the model is stale, unless it is deliberately
reporting those components as dropped.

Scans both the manuscript and the generated tables file, since the generated one
was missed in the first sweep and contained the headline comparison.

    python tools/audit_floats.py
"""
from __future__ import annotations

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92)

STALE = ["align + gate", "align + ctx", "ctx + gate", "align $+$ gate",
         "proposed configuration", "proposed decoder", "24{,}181",
         "618{,}997", "gate only", "selective head", "abstention",
         "Ours, align"]
# phrases that mark a float as deliberately about the dropped components
EXEMPT = ["not in the reported model", "tested and", "not retained",
          "None of these arms", "dropped", "containing the reported model",
          "None of the arms in this table"]

PAT = re.compile(BS + BS + r"begin\{(table\*?|figure\*?)\}(.*?)"
                 + BS + BS + r"end\{\1\}", re.S)
LAB = re.compile(BS + BS + r"label\{([^}]+)\}")


def main():
    total = stale = 0
    for f in ("paper/cas_calmnet.tex", "paper/tables_auto.tex"):
        p = os.path.join(ROOT, f)
        s = io.open(p, encoding="utf-8").read()
        print("=== %s ===" % f)
        for m in PAT.finditer(s):
            body = m.group(2)
            lm = LAB.search(body)
            lab = lm.group(1) if lm else "(no label)"
            hits = sorted(set(k for k in STALE if k in body))
            ok_by_design = any(e in body for e in EXEMPT)
            total += 1
            if hits and not ok_by_design:
                stale += 1
                print("  %-22s STALE   %s" % (lab, ", ".join(hits)[:60]))
            elif hits:
                print("  %-22s ok*     mentions dropped components on purpose"
                      % lab)
            else:
                print("  %-22s ok" % lab)
    print("\n  %d floats, %d need attention" % (total, stale))


if __name__ == "__main__":
    main()
