r"""Restore the back matter my Conclusion rewrite deleted.

Pass 10 replaced everything between \section{Conclusion} and \end{document},
which was meant to be the old conclusion prose but also contained the CRediT
authorship statement, the competing-interests declaration, the data-availability
statement, the bibliography style and the \bibliography command itself. The
build still succeeded and lost three pages of references silently, which is the
worst way for this to fail.

Recovers the back matter from git HEAD and re-appends it after the new
conclusion, with the data-availability count corrected from three datasets to
five.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def main():
    head = subprocess.run(["git", "show", "HEAD:paper/cas_calmnet.tex"],
                          cwd=ROOT, capture_output=True)
    old = head.stdout.decode("utf-8")
    i = old.index(BS + "section{Conclusion}")
    j = old.index(BS + "end{document}")
    seg = old[i:j]
    k = seg.index(BS + "section*{")
    back = seg[k:]
    if BS + "bibliography{" not in back:
        sys.exit("bibliography not found in recovered back matter")

    back = back.replace("All three datasets are publicly available.",
                        "All five datasets are publicly available.")

    s = io.open(TEX, encoding="utf-8").read()
    if BS + "bibliography{" in s:
        print("bibliography already present; nothing to do")
        return
    e = s.index(BS + "end{document}")
    s = s[:e] + back + "\n" + s[e:]
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("restored %d chars of back matter" % len(back))
    for c in ("section*{CRediT", "section*{Declaration", "section*{Data",
              "bibliographystyle", "bibliography{refs}"):
        print("   %-24s %s" % (c, "ok" if BS + c in s else "MISSING"))


if __name__ == "__main__":
    main()
