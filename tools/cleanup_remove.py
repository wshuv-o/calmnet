r"""Remove the earlier architectures from the manuscript, safely.

Three bounded deletions failed tonight, each the same way: a bound that looked
structural and was not. This one bounds differently.

Sections are bounded by a single ordered pass over every \section and
\subsection, so a subsection delete stops at the next heading of ANY level and
cannot run into the following section.

Floats are bounded by matching \begin{X}...\end{X} with a backreference over the
whole file first, then selecting the span that contains the label. Scanning
backwards for a \begin of a guessed type is what swallowed three pages earlier.

Before writing anything it reports every label inside the removed spans that is
referenced from outside them, because that is the check that would have caught
Condition 1 leaving with a deleted section.

    python tools/cleanup_remove.py --dry-run
    python tools/cleanup_remove.py
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

DROP_SECTIONS = [
    "The drift mechanism",
    "What the alignment layer did, and why it is not retained",
    "Correcting the covariance estimator",
    "Components tested and not retained",
    "Calibration and selective prediction",
]
DROP_FLOATS = ["fig:drift", "tab:ablation", "fig:inversion", "tab:external",
               "fig:topo"]

HEAD = re.compile(BS + BS + r"(?:sub)?section\*?\{")
FLOAT = re.compile(BS + BS + r"begin\{(table\*?|figure\*?)\}.*?"
                   + BS + BS + r"end\{\1\}", re.S)
LABEL = re.compile(BS + BS + r"label\{([^}]+)\}")


def heading_spans(s):
    """(start, end, title) for every heading, bounded by the next heading of
    any level."""
    starts = [m.start() for m in HEAD.finditer(s)]
    out = []
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else len(s)
        t = re.match(BS + BS + r"(?:sub)?section\*?\{([^}]*)\}", s[a:])
        out.append((a, b, t.group(1) if t else ""))
    return out


def float_spans(s):
    return [(m.start(), m.end(), m.group(0)) for m in FLOAT.finditer(s)]


def main():
    dry = "--dry-run" in sys.argv
    s = io.open(TEX, encoding="utf-8").read()
    gen = io.open(GEN, encoding="utf-8").read()

    spans = []
    for a, b, title in heading_spans(s):
        if title in DROP_SECTIONS:
            spans.append((a, b, "section: " + title))
    for a, b, body in float_spans(s):
        lm = LABEL.search(body)
        if lm and lm.group(1) in DROP_FLOATS:
            spans.append((a, b, "float: " + lm.group(1)))

    # a float inside a dropped section is already covered
    spans.sort()
    merged = []
    for a, b, why in spans:
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b),
                          merged[-1][2] + " + " + why)
        else:
            merged.append((a, b, why))

    removed = "".join(s[a:b] for a, b, _ in merged)
    kept = ""
    prev = 0
    for a, b, _ in merged:
        kept += s[prev:a]
        prev = b
    kept += s[prev:]

    inner = set(LABEL.findall(removed))
    outside_refs = set(re.findall(r"ref\{([^}]+)\}", kept + gen))
    orphan = sorted(inner & outside_refs)

    print("spans to remove: %d, %d chars (%.1f%% of the file)"
          % (len(merged), len(removed), 100 * len(removed) / len(s)))
    for a, b, why in merged:
        print("   %-64s %6d chars" % (why[:64], b - a))
    print("\nlabels defined inside them: %s" % (sorted(inner) or "none"))
    print("of those, still referenced from outside: %s" % (orphan or "none"))

    if orphan:
        print("\nreferences that would break, with their citing text:")
        for lab in orphan:
            for m in re.finditer(r".{0,70}ref\{" + re.escape(lab) + r"\}", kept):
                print("   [%s] ...%s" % (lab, m.group(0)[-70:]))
                break

    if dry:
        print("\n(dry run, nothing written)")
        return
    if orphan:
        sys.exit("\nrefusing to write while %d reference(s) would break; "
                 "repoint or remove them first" % len(orphan))
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(kept)
    print("\nwritten: %d -> %d chars" % (len(s), len(kept)))


if __name__ == "__main__":
    main()
