r"""Restore \label{sec:align}, lost when the Method was restructured.

Pass 4 turned the Adaptive alignment subsection into a \paragraph inside the
dropped-components section, which dropped its \label. Two \ref{sec:align} calls
elsewhere then rendered as "??" in the PDF while the build reported success,
because an undefined reference is a warning in tectonic rather than an error.

A sed attempt at this failed silently on the backslashes, which is the third
time tonight that shell quoting has eaten a LaTeX escape. This does it in the
editor instead.
"""
import io
import os

BS = chr(92)
TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "cas_calmnet.tex")

OLD = BS + "paragraph{Adaptive alignment}"
NEW = BS + "paragraph{Adaptive alignment}" + BS + "label{sec:align}"


def main():
    s = io.open(TEX, encoding="utf-8").read()
    if BS + "label{sec:align}" in s:
        print("label already present")
        return
    if OLD not in s:
        raise SystemExit("anchor not found: " + OLD)
    s = s.replace(OLD, NEW, 1)
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("restored label{sec:align}; %d ref(s) now resolve"
          % s.count(BS + "ref{sec:align}"))


if __name__ == "__main__":
    main()
