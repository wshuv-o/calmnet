r"""Pass 4 of the rewrite: the Method section.

The section opened by announcing four components and led with adaptive
alignment. The reported model has two: the multi-scale log-power stem and the
tangent-space branch. Alignment, the cross-epoch transformer and the selective
head were all tested and none is retained.

Restructured so the two components that exist come first and in the order the
signal passes through them, and the three that were dropped are grouped into one
honest subsection that points at the measurements which dropped them.

The running-covariance update equation stays. It was written for the alignment
layer and now describes the branch's reference, which uses the same rule, so the
rate condition derived on the layer applies to the branch unchanged.

One correction the old text forced: the branch described its reference as the
one the alignment layer maintains. With alignment gone the branch maintains its
own, and the text now says so.

    python tools/rewrite_04_method.py
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def cut(s, start_marker, end_marker):
    i = s.index(start_marker)
    j = s.index(end_marker, i)
    return s[i:j], s[:i] + s[j:]


def main():
    s = io.open(TEX, encoding="utf-8").read()

    align, s = cut(s, BS + "subsection{Adaptive alignment}",
                   BS + "subsection{Multi-scale power stem}")
    ctx, s = cut(s, BS + "subsection{Causal cross-epoch context}",
                 BS + "subsection{Selective head}")
    sel, s = cut(s, BS + "subsection{Selective head}",
                 "%" + "=" * 77)

    # ---- opening paragraph and figure caption
    i = s.index(BS + "section{Method}")
    j = s.index(BS + "subsection{Multi-scale power stem}")
    head = (
        BS + "section{Method}\n\n"
        "The reported decoder has two components. A multi-scale log-power stem "
        "reads the diagonal of each window's spatial covariance, and a "
        "tangent-space branch reads the rest of it. Their outputs are "
        "concatenated and passed to one linear classifier. The stem carries "
        "20{,}019 parameters and the branch 270{,}924, so the branch is the "
        "larger part of the model by an order of magnitude and "
        "Section~" + BS + "ref{sec:compare} reports what that buys.\n\n"
        "Three further components were built and tested, and none is retained: "
        "an adaptive alignment layer, a causal cross-epoch transformer and a "
        "selective head. Section~" + BS + "ref{sec:dropped} describes them and "
        "points at the measurements that removed them, because each defines an "
        "arm of the ablation and one of them, the alignment layer, supplies the "
        "mechanism that explains where the branch fails.\n\n"
        + BS + "begin{figure*}[t]\n" + BS + "centering\n"
        + BS + "includegraphics[width=" + BS + "textwidth]"
        "{../results/fig_arch.pdf}\n"
        + BS + "caption{The reported decoder. A window enters both paths. The "
        "stem applies three temporal resolutions with depthwise spatial "
        "filters, squares, pools and takes the logarithm, giving log band power "
        "per channel. The branch forms the window's covariance, shrinks it, "
        "maps it to the tangent space at a running reference and projects the "
        "result. The reference update (dashed) uses no labels and stays active "
        "at inference, which is what lets the representation follow a changing "
        "session and also what makes it fail where a class block outlasts one "
        "update (Section~" + BS + "ref{sec:rate}). Parameter counts are exact "
        "and sum to the total.}\n"
        + BS + "label{fig:arch}\n"
        + BS + "end{figure*}\n\n")
    s = s[:i] + head + s[j:]

    # ---- fix the branch's reference description
    s = s.replace(
        "The reference is the running covariance the alignment layer of "
        "Section~" + BS + "ref{sec:align} already maintains, updated without "
        "labels and still updating at test time. Where alignment is disabled "
        "the branch keeps its own running mean under the same momentum, so "
        "removing the branch does not silently remove adaptation as well. "
        "Tying the two mechanisms to one statistic is what places the branch "
        "under Condition~" + BS + "ref{prop:band}: the rate that makes the "
        "alignment layer safe is the rate that makes the reference safe, and "
        "Section~" + BS + "ref{sec:tanres} tests that consequence.",
        "The reference $M$ is a running mean over the covariances of incoming "
        "windows, updated by (" + BS + "ref{eq:update}) with momentum $m$ and "
        "no labels, and it keeps updating at inference. This is what lets the "
        "representation follow a session as it changes, and it is also the "
        "component that fails: because $M$ is an average over whatever is "
        "currently streaming, a protocol that presents one class for longer "
        "than a single update spans will drive $M$ onto that class. "
        "Condition~" + BS + "ref{prop:band} states when that cannot happen and "
        "Section~" + BS + "ref{sec:fivecohort} measures it on five cohorts. "
        "The condition was derived on the alignment layer of "
        "Section~" + BS + "ref{sec:dropped}, which shares this estimator, and "
        "it transfers to the branch unchanged.", 1)

    # ---- the dropped-components subsection, appended after the branch
    anchor = "%" + "=" * 77 + "\n" + BS + "section{Experimental setup}"
    dropped = (
        BS + "subsection{Components tested and not retained}\n"
        + BS + "label{sec:dropped}\n\n"
        "Three components were built, measured, and left out of the reported "
        "model. Each is an arm of the ablation in "
        "Section~" + BS + "ref{sec:ablation}, and they are described here "
        "because a reader cannot judge an ablation against arms that are never "
        "defined.\n\n"
        + BS + "paragraph{Adaptive alignment} " + align.split(BS
        + "label{sec:align}")[1].strip() + "\n\n"
        "This layer is not in the reported model. On the cohort where the "
        "branch works best it changes accuracy by less than the spread between "
        "seeds, and on the cohort used for development the branch and the "
        "layer are substitutes rather than complements, since both read the "
        "same second-order structure; "
        "Section~" + BS + "ref{sec:fivecohort} gives the measurement. Its "
        "estimator is retained, as the branch's reference, and the rate "
        "condition derived on it is what predicts where the branch fails.\n\n"
        + BS + "paragraph{Causal cross-epoch context} " + ctx.split(
            BS + "subsection{Causal cross-epoch context}")[1].strip()
        + " This pathway is not in the reported model: it accounts for "
        "594{,}816 parameters and costs accuracy on the development cohort.\n\n"
        + BS + "paragraph{Selective head} " + sel.split(
            BS + "subsection{Selective head}")[1].strip()
        + " This head is not in the reported model. The trained selection "
        "function did not rank windows better than the classifier's own "
        "confidence, so it provided no usable reject option "
        "(Section~" + BS + "ref{sec:ablation}).\n\n")
    s = s.replace(anchor, dropped + anchor, 1)

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("PASS 4 done: Method restructured")
    print("  - opens on two components, stem then branch, with parameter split")
    print("  - branch now owns its reference; alignment no longer referenced")
    print("  - alignment, context and selective head grouped as not retained")
    print("  - fig:arch caption rewritten for the two-path model")


if __name__ == "__main__":
    main()
