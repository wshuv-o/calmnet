r"""Pass 11 of the rewrite: consistency audit and the fixes it found.

An automated sweep for language describing the old architecture found the
comparison section still calling the align-plus-gate arm "the proposed
configuration" six times, and quoting 24,181 as the model's parameter count.
That section predates the branch and is superseded by the master table, but it
carries measurements that exist nowhere else, so it is relabelled rather than
deleted. The previous pass showed what deleting by structural marker costs.

Also fixes one remaining conflation of the old full-model total.

    python tools/rewrite_11_audit.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)

SUBS = [
    # the comparison section reports a configuration that was dropped
    ("the proposed configuration", "the align $+$ gate configuration"),
    ("the proposed configuration's", "the align $+$ gate configuration's"),
    ("pipeline as the proposed decoder", "pipeline as the decoder"),
    # a parameter count belonging to a model that is not reported
    ("against 24{,}181.",
     "against 24{,}181 for that configuration. The reported model is "
     "290{,}943 and is compared with the same decoders across all five "
     "cohorts in Table~" + BS + "ref{tab:master}."),
]

HEADER = (
    "This section reports the align $+$ gate configuration, which is one of "
    "the components of Section~" + BS + "ref{sec:dropped} and is not the "
    "reported model. It is retained because it carries per-participant "
    "comparisons and calibration numbers that exist nowhere else in the "
    "paper. The comparison for the reported model, against the same eight "
    "decoders on all five cohorts, is Table~" + BS + "ref{tab:master}.\n\n")


def main():
    s = io.open(TEX, encoding="utf-8").read()
    before = {k: s.count(k) for k, _ in SUBS}

    for old, new in SUBS:
        s = s.replace(old, new)

    # header sentence at the top of the comparison section
    anchor = BS + "subsection{Comparison with published decoders}"
    if anchor in s and "is not the reported model. It is retained" not in s:
        i = s.index(anchor) + len(anchor)
        # skip a label if one follows
        m = re.match(r"\s*\\label\{[^}]*\}", s[i:])
        if m:
            i += m.end()
        s = s[:i] + "\n\n" + HEADER + s[i:].lstrip("\n")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    print("PASS 11 fixes applied")
    for k, n in before.items():
        print("   %-34s %d occurrence(s)" % (k[:34], n))

    # ---- re-audit
    print("\nre-audit:")
    checks = [("proposed configuration", "names a dropped arm"),
              ("the proposed decoder", "ambiguous"),
              ("24{,}181", "old parameter total"),
              ("618{,}997", "old full-model total")]
    for k, why in checks:
        print("   %-26s %d" % (k, s.count(k)))
    labs = re.findall(r"\\label\{([^}]+)\}", s)
    refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
    dup = sorted(l for l in set(labs) if labs.count(l) > 1)
    miss = sorted(refs - set(labs))
    print("   duplicate labels          %s" % (dup or "none"))
    print("   refs without a label      %s" % (miss or "none"))
    bare = 0
    for line in s.split("\n"):
        st = line.lstrip()
        if st.startswith("%"):
            continue
        for m in re.finditer("%", line):
            if m.start() > 0 and line[m.start() - 1] != BS:
                bare += 1
                break
    print("   lines with a bare percent %d" % bare)


if __name__ == "__main__":
    main()
