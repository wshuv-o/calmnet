r"""Pass 3 of the rewrite: Related work.

Two structural problems.

There is a full subsection on calibration and selective prediction, describing a
selective head that is not in the reported model. It is cut to a short paragraph
that says the component was tested and dropped, since the ablation still reports
that result and a reader should know why it is there.

There is no subsection on Riemannian and tangent-space representations, which is
the literature the contribution actually sits in. One is added, placed directly
after alignment and transfer so the two read together: alignment moves the
covariance, the tangent map reads it.

Adds one bibliography entry, Huang and Van Gool's SPD network, because this
project ran SPD baselines and had no way to cite the family.

    python tools/rewrite_03_related.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BIB = os.path.join(ROOT, "paper", "refs.bib")
BS = chr(92)

SPDNET = """
@inproceedings{spdnet,
  title     = {A {R}iemannian Network for {SPD} Matrix Learning},
  author    = {Huang, Zhiwu and Van Gool, Luc},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {31},
  year      = {2017}
}
"""


def main():
    # ---- bibliography
    b = io.open(BIB, encoding="utf-8").read()
    if "@inproceedings{spdnet," not in b:
        io.open(BIB, "a", encoding="utf-8").write(SPDNET)
        print("added bib entry: spdnet")

    s = io.open(TEX, encoding="utf-8").read()

    # ---- new subsection, after alignment and transfer
    anchor = BS + "subsection{Test-time adaptation}"
    new_sub = (
        BS + "subsection{Riemannian and tangent-space representations}\n"
        + BS + "label{sec:rw_riem}\n\n"
        "Spatial covariances are symmetric positive-definite matrices and do "
        "not form a vector space, so the methods built on them operate on the "
        "SPD manifold. The standard recipe computes a covariance per window, "
        "maps it to the tangent space at a reference point, and classifies the "
        "resulting vector with a linear model "
        + BS + "citep{barachant,barachant2013,congedo}. The reference is "
        "usually the Riemannian or arithmetic mean of the training "
        "covariances, and the choice matters because the map is only a faithful "
        "local approximation near that point. These representations remain "
        "competitive with convolutional decoders on movement-intent tasks and "
        "are the strongest general-purpose transfer tools reported in reviews "
        + BS + "citep{lotte2018,wu_transfer}. SPD networks learn the manifold "
        "operations instead of fixing them " + BS + "citep{spdnet}.\n\n"
        "Two differences separate the branch used here from that line. The "
        "tangent map is a second input to a convolutional decoder rather than "
        "the representation the decoder is built on, so first-order band power "
        "and second-order structure are read together instead of one replacing "
        "the other; Section~" + BS + "ref{sec:fivecohort} reports that the two "
        "are complementary and that the branch alone is weaker than the stem "
        "alone. And the reference point is estimated from the incoming windows "
        "at inference rather than fixed after fitting, which is what makes the "
        "representation track a changing session and is also, as "
        "Section~" + BS + "ref{sec:rate} shows, what makes it fail under a "
        "blocked protocol.\n\n")
    s = s.replace(anchor, new_sub + anchor, 1)

    # ---- cut the selective-prediction subsection to a short paragraph
    i = s.index(BS + "subsection{Calibration and selective prediction}")
    j = s.index(BS + "subsection{EEG correlates of gait", i)
    short = (
        BS + "subsection{Calibration and selective prediction}\n\n"
        "Modern networks are overconfident and expected calibration error "
        + BS + "citep{naeini,guo} degrades further under distribution shift "
        + BS + "citep{ovadia}, which is the regime a longitudinally deployed "
        "decoder operates in. Selective classification allows a model to "
        "abstain " + BS + "citep{chow,elyaniv}, and SelectiveNet trains the "
        "classifier and the selection function jointly under a coverage "
        "constraint " + BS + "citep{selectivenet,geifman2017}. A device that "
        "must not move a wearer's legs on weak evidence is an obvious "
        "candidate for a reject option, so we built one in and tested it. It "
        "did not work: the trained selection function ranked windows no better "
        "than the classifier's own confidence, and the head is not part of the "
        "reported model. Section~" + BS + "ref{sec:ablation} gives the "
        "measurement. We claim no contribution to selective prediction and "
        "report calibration error throughout because it is what a controller "
        "would threshold on.\n\n")
    s = s[:i] + short + s[j:]

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("PASS 3 done: Related work")
    print("  + subsection on Riemannian and tangent-space representations")
    print("  - selective prediction cut to a paragraph, reframed as dropped")


if __name__ == "__main__":
    main()
