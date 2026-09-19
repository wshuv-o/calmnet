r"""Restore Condition 1, deleted with the section that happened to contain it.

The merge in pass 13 guarded against losing floats and checked every
\begin{table}/\begin{figure} in the removed span. The condition environment is
not a float, so \begin{condition}[Admissible rate band] went with the prose, and
with it \label{prop:band} and \label{eq:band}. Ten references across the paper
then rendered as "??" in a build that reported success.

The lesson generalises past this one block: a guard that enumerates the kinds of
thing it expects to find will miss the kind it did not think of. The check
should have been "no \label in the removed span may be referenced from outside
it", which is what verify() below does.

Restores the block into "The adaptation rate condition", which is where a reader
looking for it would go.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
GEN = os.path.join(ROOT, "paper", "tables_auto.tex")
BS = chr(92)


def recover():
    o = subprocess.run(["git", "show", "HEAD:paper/cas_calmnet.tex"],
                       cwd=ROOT, capture_output=True)
    s = o.stdout.decode("utf-8")
    i = s.index(BS + "label{prop:band}")
    a = s.rindex(BS + "begin{condition}", 0, i)
    b = s.index(BS + "end{condition}", i) + len(BS + "end{condition}")
    return s[a:b]


def verify(tex, gen):
    """No reference anywhere may lack a label. The build is the authority but
    this catches it before a 24-page PDF ships with question marks in it."""
    both = tex + gen
    labs = set(re.findall(r"label\{([^}]+)\}", both))
    refs = set(re.findall(r"ref\{([^}]+)\}", both))
    return sorted(refs - labs)


def main():
    s = io.open(TEX, encoding="utf-8").read()
    if BS + "label{prop:band}" in s:
        print("condition already present")
    else:
        block = recover()
        anchor = BS + "subsection{The adaptation rate condition}"
        if anchor not in s:
            sys.exit("target section not found")
        i = s.index(anchor) + len(anchor)
        m = re.match(r"\s*" + BS + BS + r"label\{[^}]*\}", s[i:])
        if m:
            i += m.end()
        s = s[:i] + "\n\n" + block + "\n" + s[i:].lstrip("\n")
        io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
        print("restored %d chars into The adaptation rate condition"
              % len(block))

    gen = io.open(GEN, encoding="utf-8").read()
    left = verify(s, gen)
    print("  references without a label: %s" % (left or "none"))


if __name__ == "__main__":
    main()
