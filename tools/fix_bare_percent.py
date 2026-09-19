r"""Escape the bare percent signs that have been truncating sentences.

A percent sign begins a comment in LaTeX, so "74\,% of the loss" renders as
"74" and silently discards the rest of the line. Three of these were introduced
during the rewrite and one of them is in the abstract, which has been shipping
with its final sentence cut off since pass 1. The build reports nothing.

A trailing "%" at end of line after {, } or a tabular is deliberate line
continuation and is left alone; only a "%" with text after it on the same line
is a defect.

This is the third time this bug has appeared in the project, so the check is
kept as a tool rather than done by hand.

    python tools/fix_bare_percent.py [--check]
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92)
FILES = ["paper/cas_calmnet.tex", "paper/tables_auto.tex"]


def defects(line):
    """Positions of a percent that is not escaped, not a whole-line comment,
    and has non-space text after it."""
    out = []
    if line.lstrip().startswith("%"):
        return out
    for j, ch in enumerate(line):
        if ch != "%" or j == 0 or line[j - 1] == BS:
            continue
        if line[j + 1:].strip():          # text follows -> it is being eaten
            out.append(j)
    return out


def main():
    check = "--check" in sys.argv
    total = 0
    for f in FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        lines = io.open(p, encoding="utf-8").read().split("\n")
        changed = False
        for i, line in enumerate(lines):
            d = defects(line)
            if not d:
                continue
            total += len(d)
            print("  %s:%d  ...%s" % (os.path.basename(f), i + 1,
                                      line[max(0, d[0] - 46):d[0] + 34]))
            if not check:
                new = ""
                prev = 0
                for j in d:
                    new += line[prev:j] + BS + "%"
                    prev = j + 1
                lines[i] = new + line[prev:]
                changed = True
        if changed:
            io.open(p, "w", encoding="utf-8",
                    newline="\n").write("\n".join(lines))
    print("\n  %d truncating percent sign(s) %s"
          % (total, "found" if check else "escaped"))
    return 1 if (check and total) else 0


if __name__ == "__main__":
    sys.exit(main())
