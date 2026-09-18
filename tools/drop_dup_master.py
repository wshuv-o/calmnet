r"""Remove the duplicate master table, now that Table 3 does the job.

Pass 8 added tab:master because no table showed the reported model against the
field. Table 3 (tab:compare) has since been rewritten to be exactly that, with
parameter counts as well, so tab:master duplicates it and the paper carries 21
tables.

Only the table environment is removed; the surrounding prose is kept and
repointed, because it states the parameter trade and the cohort-B reading, which
appear nowhere else. The removal is bounded by the begin/end of the float that
contains the label, not by the next structural marker, after a Conclusion
rewrite bounded that way deleted the bibliography.
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def main():
    s = io.open(TEX, encoding="utf-8").read()
    if BS + "label{tab:master}" not in s:
        print("tab:master already gone")
        return

    k = s.index(BS + "label{tab:master}")
    start = s.rindex(BS + "begin{table*}", 0, k)
    end = s.index(BS + "end{table*}", k) + len(BS + "end{table*}")
    removed = end - start
    s = s[:start] + s[end:]

    # repoint the prose at Table 3
    s = s.replace("Table~" + BS + "ref{tab:master} is the comparison the rest "
                  "of the results build toward: the reported model, the stem "
                  "it is built on, and eight published decoders, on every "
                  "cohort, three seeds each.",
                  "Table~" + BS + "ref{tab:compare} is the comparison the rest "
                  "of the results build toward: the reported model, the stem "
                  "it is built on, and eight published decoders, on every "
                  "cohort, three seeds each.")
    s = s.replace("for which see Table~" + BS + "ref{tab:newbase}.}",
                  "for which see Table~" + BS + "ref{tab:newbase}.}")
    s = s.replace(BS + "ref{tab:master}", BS + "ref{tab:compare}")

    if BS + "ref{tab:master}" in s:
        sys.exit("a reference to tab:master survived")
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("removed %d chars (the float only); prose kept and repointed"
          % removed)
    print("  remaining refs to tab:compare: %d"
          % s.count(BS + "ref{tab:compare}"))


if __name__ == "__main__":
    main()
