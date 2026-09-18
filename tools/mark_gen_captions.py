r"""Mark the four generated captions whose arms are components we do not report.

The previous attempt located the caption by index arithmetic from the label
argument and inserted the sentence into the label itself, breaking the column
specification. This anchors on a verbatim fragment at the START of each caption
and inserts immediately after it, which is unambiguous.

The sentence is written as a plain Python string with no backslash constructs,
so nothing can be eaten between here and the file.
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "make_tables.py")

MARK = ('"None of the arms in this table is the reported model, which is the '
        'stem plus the tangent branch; these are the components of "\n'
        '        + BS + "ref{sec:dropped} reported in Table~" + BS '
        '+ "ref{tab:compare}. "\n        ')

# anchor: the opening fragment of each caption, verbatim from the source
ANCHORS = {
    "tab:ablation3": '"Component ablation on cohort A under the current estimator: balanced "',
    "tab:rate": '"The adaptation-rate condition, single seed. Accuracies are mean $"',
    "tab:ratecurve": '"Cohort B over three data-split seeds under the current estimator. "',
    "tab:noise": '"Measurement noise. The first block repeats arms for which momentum "',
}


def main():
    s = io.open(SRC, encoding="utf-8").read()
    done = 0
    for lab, anchor in ANCHORS.items():
        if anchor not in s:
            print("  %-16s anchor not found" % lab)
            continue
        i = s.index(anchor)
        if "None of the arms in this table" in s[i - 400:i + 400]:
            print("  %-16s already marked" % lab)
            continue
        j = i + len(anchor)
        s = s[:j] + "\n        " + MARK + s[j:]
        done += 1
        print("  %-16s marked" % lab)
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(s)
    print("\n  %d captions marked in the generator" % done)


if __name__ == "__main__":
    main()
