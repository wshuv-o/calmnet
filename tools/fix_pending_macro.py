r"""Make \pending render in red instead of silently vanishing.

The manuscript already carried a \pending macro defined as a no-op taking a
mandatory argument, from a draft where every value was final. Calls written as
a bare \pending in a table cell then consumed the row terminator as their
argument, so the cell rendered empty rather than flagged. An empty cell in a
results table is exactly the failure this macro exists to prevent.

This redefines it with an optional argument, so a bare \pending reads
"run pending" in red and \pending[waiting on the 2060] reads that instead.
"""
import io
import os

TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "cas_calmnet.tex")
BS = chr(92)

OLD = BS + "newcommand{" + BS + "pending}[1]{#1}"
NEW = (
    "% Cells still waiting on a run. Red and labelled, so nothing outstanding\n"
    "% can be mistaken for a measured value. A bare " + BS + "pending reads\n"
    "% \"run pending\"; the optional argument overrides that text.\n"
    + BS + "definecolor{pendingred}{RGB}{200,30,30}\n"
    + BS + "newcommand{" + BS + "pending}[1][run pending]{"
    + BS + "textcolor{pendingred}{" + BS + "textbf{#1}}}")


def main():
    s = io.open(TEX, encoding="utf-8").read()
    if OLD not in s:
        raise SystemExit("no-op macro not found; nothing changed")
    s = s.replace(OLD, NEW, 1)
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("redefined pending macro; %d call sites affected"
          % s.count(BS + "pending "))


if __name__ == "__main__":
    main()
