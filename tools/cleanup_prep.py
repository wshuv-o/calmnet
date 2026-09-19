r"""Prepare for the removal: relocate what the branch needs, repair what cites it.

Two jobs, both before anything is deleted.

The running-covariance update equation is currently defined inside the
alignment material. It is the branch's reference update and the branch section
already refers to it, so it moves into the branch section with the trace
normalisation rationale that belongs with it.

Then every sentence that cites something about to be removed is rewritten or
dropped, so the removal finds no live references and the guard in
cleanup_remove.py can pass on its own terms rather than being overridden.

    python tools/cleanup_prep.py
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
GEN = os.path.join(ROOT, "paper", "tables_auto.tex")
BS = chr(92)

# The branch's reference update, written for the branch rather than the layer.
REF_UPDATE = (
    "\n\nThe reference is a running mean over the covariances of incoming "
    "windows, with each window's covariance normalised by its own trace before "
    "being accumulated, so the estimate is an average of shapes rather than an "
    "average weighted by power:\n"
    + BS + "begin{equation}\n"
    + BS + "mathbf{M} " + BS + "leftarrow (1-m)" + BS + "," + BS
    + "mathbf{M} + " + BS + "frac{m}{N} " + BS + "sum_{n=1}^{N}\n"
    + BS + "frac{" + BS + "mathbf{C}_n}{" + BS + "operatorname{tr}("
    + BS + "mathbf{C}_n)/C},\n" + BS + "qquad\n"
    + BS + "mathbf{C}_n = " + BS + "frac{" + BS + "tilde{" + BS
    + "mathbf{X}}_n " + BS + "tilde{" + BS + "mathbf{X}}_n^{" + BS + "top}}{T},\n"
    + BS + "label{eq:update}\n"
    + BS + "end{equation}\n"
    "where $" + BS + "tilde{" + BS + "mathbf{X}}_n$ is the mean-centred "
    "window, $N$ the number of windows in an update and $m$ the momentum, so "
    "the memory is $" + BS + "mu = N/m$ windows. Without the trace "
    "normalisation the estimate is power-weighted, and on a cohort whose "
    "majority class carries $1.52" + BS + "times$ the power of the minority "
    "class a power-weighted estimate is the majority class rather than an "
    "average over both.\n")

SUBS = [
    # ---- citing sentences that must go with what they cite
    ("Fig.~" + BS + "ref{fig:topo} shows the scalp distribution for one "
     "participant. Absolute", "Absolute"),
    ("(Section~" + BS + "ref{sec:align}).", "."),
    ("Section~" + BS + "ref{sec:align}. The defect is real and measurable: "
     "on cohort B, where", "On cohort B, where"),
    ("" + BS + "ref{tab:ratecurve}, Section~" + BS + "ref{sec:alignfindings})",
     BS + "ref{tab:ratecurve})"),
    ("The mechanism works as designed, and Section~" + BS
     + "ref{sec:alignfindings} shows", "Section~" + BS
     + "ref{sec:fivecohort} shows"),
    ("one is retained: an adaptive alignment layer, a causal cross-epoch "
     "transformer and a selective head. Section~" + BS + "ref{sec:dropped} "
     "describes them and points at the measurements that removed ",
     "one is retained. "),
    ("The condition was derived on the alignment layer of Section~" + BS
     + "ref{sec:dropped}, which shares this estimator, and it transfers to "
     "the branch unchanged.", ""),
    ("reduction in session drift (Section~" + BS + "ref{sec:mechanism}).", "."),
    ("Section~" + BS + "ref{sec:rejected} reports what correcting", "Correcting"),
    ("Section~" + BS + "ref{sec:rejected}. The lower figure should be read "
     "with care: the", "The lower figure should be read with care: the"),
    ("($0.878$ against $0.827$, Table~" + BS + "ref{tab:ablation})", ""),
    ("$0.06$ to $0.13$ (Table~" + BS + "ref{tab:ablation})",
     "$0.06$ to $0.13$"),
    ("(Table~" + BS + "ref{tab:external}, Fig.~" + BS + "ref{fig:inversion})",
     ""),
]
GEN_SUBS = [
    ("same pipeline as Table~" + BS + "ref{tab:external}",
     "same pipeline as Table~" + BS + "ref{tab:compare}"),
]


def move_equation(s):
    """Lift the update equation out of the alignment material and put the
    branch's own version into the branch subsection."""
    i = s.find(BS + "label{eq:update}")
    if i < 0:
        print("  equation already moved")
        return s
    a = s.rindex(BS + "begin{equation}", 0, i)
    b = s.index(BS + "end{equation}", i) + len(BS + "end{equation}")
    s = s[:a] + s[b:]

    anchor = ("A linear projection maps $" + BS + "phi(X)$ to the embedding "
              "width")
    j = s.find(anchor)
    if j < 0:
        sys.exit("branch subsection anchor not found")
    k = s.index("\n\n", j)
    return s[:k] + REF_UPDATE + s[k:]


def main():
    s = io.open(TEX, encoding="utf-8").read()
    s = move_equation(s)
    print("  update equation relocated into the branch subsection")

    ok = miss = 0
    for old, new in SUBS:
        if old in s:
            s = s.replace(old, new)
            ok += 1
        else:
            miss += 1
            print("   no match: %s..." % old[:62])
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    g = io.open(GEN, encoding="utf-8").read()
    for old, new in GEN_SUBS:
        if old in g:
            g = g.replace(old, new)
            ok += 1
        else:
            miss += 1
            print("   no match in generator: %s..." % old[:50])
    io.open(GEN, "w", encoding="utf-8", newline="\n").write(g)
    print("  %d citing sentences repaired, %d unmatched" % (ok, miss))


if __name__ == "__main__":
    main()
