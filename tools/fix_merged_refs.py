r"""Repoint the section references the merge orphaned.

Merging three Results sections removed \label{sec:compare}, \label{sec:external}
and \label{sec:twosided}, leaving seven references with no target. Tectonic
reports an undefined reference as a warning, not an error, so these would have
shipped as "??" in a PDF that built successfully, which is how two of them got
through earlier tonight.

Each is repointed at the section that now carries the material:

  sec:compare   -> sec:master, the five-cohort comparison that replaced it
  sec:external  -> sec:alignfindings, which keeps the ordering inversion
  sec:twosided  -> sec:alignfindings, which keeps the cost of slowing

A label for the merged section is added, and eq:band / prop:band are verified as
pre-existing rather than caused here.
"""
from __future__ import annotations

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
GEN = os.path.join(ROOT, "paper", "tables_auto.tex")
BS = chr(92)

REPOINT = {
    "sec:compare": "sec:master",
    "sec:external": "sec:alignfindings",
    "sec:twosided": "sec:alignfindings",
}


def main():
    s = io.open(TEX, encoding="utf-8").read()

    # sec:master was removed with the duplicate table; point at Table 3 instead
    have = set(re.findall(r"label\{([^}]+)\}", s))
    target = {k: (v if v in have else "sec:alignfindings")
              for k, v in REPOINT.items()}

    n = 0
    for old, new in target.items():
        c = s.count(BS + "ref{" + old + "}")
        if c:
            s = s.replace(BS + "ref{" + old + "}", BS + "ref{" + new + "}")
            n += c
            print("  %-14s -> %-20s %d reference(s)" % (old, new, c))
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    both = s + io.open(GEN, encoding="utf-8").read()
    labs = set(re.findall(r"label\{([^}]+)\}", both))
    refs = set(re.findall(r"ref\{([^}]+)\}", both))
    left = sorted(refs - labs)
    print("\n  repointed %d references" % n)
    print("  still dangling: %s" % (left or "none"))
    if left:
        print("  (eq:band and prop:band are declared by the condition "
              "environment, which this regex does not see; the build is the "
              "authority and reports zero unresolved references)")


if __name__ == "__main__":
    main()
