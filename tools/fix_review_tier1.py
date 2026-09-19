r"""Fix the factual errors a reviewer found. Each is checked, not assumed.

1. [10? ] in the Introduction. \citep{schirrmeister,eegnet} but there is no
   eegnet key in refs.bib; EEGNet is cited everywhere else as lawhern.

2. Parameter arithmetic. 19,505 + 238,028 + 32,896 = 290,429, not 290,943.
   The missing 514 is the classifier, which the text never names. Table 14
   gives the stem as 20,019 because a cost table must count a runnable model,
   classifier included. Both are right; neither said which convention it used.

3. AdamW cited as \citep{adam}, which is Kingma and Ba's Adam. AdamW is
   Loshchilov and Hutter, Decoupled Weight Decay Regularization, ICLR 2019.

4. Table 2 says amplitude normalisation is "global, per recording", which
   reads as though the statistic comes from the whole recording including the
   test sessions. It does not: exp_globalnorm.normalise computes mu and sd
   from Xf, the fitting split, and applies them to the calibration and test
   splits. The wording invited a fair objection to a method that is already
   causal, so the wording is what changes.

    python tools/fix_review_tier1.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BIB = os.path.join(ROOT, "paper", "refs.bib")
BS = chr(92)


def edit(s, old, new, label):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    n = len(pat.findall(s))
    if n != 1:
        sys.exit("FAILED %s: %d matches" % (label, n))
    print("  ok  " + label)
    return pat.sub(lambda _: new, s, count=1)


def main():
    tex = io.open(TEX, encoding="utf-8").read()
    bib = io.open(BIB, encoding="utf-8").read()

    # ----------------------------------------------- 1. the [10? ] artifact
    if "eegnet" in bib.split("@")[0] or re.search(r"@\w+\{eegnet,", bib):
        sys.exit("an eegnet key exists after all; check before replacing")
    tex = edit(tex, BS + "citep{schirrmeister,eegnet}",
               BS + "citep{schirrmeister,lawhern}", "[10? ] citation artifact")

    # ------------------------------------------------- 2. parameter counts
    tex = edit(tex,
               "The branch carries 238{,}028 parameters and the fusion layer "
               "it requires a further 32{,}896, against 19{,}505 in the stem "
               "path; the reported model is 290{,}943 in total, of which the "
               "branch and its fusion are 93" + BS + "," + BS + "%.",
               "The branch carries 238{,}028 parameters and the fusion layer "
               "it requires a further 32{,}896, against 19{,}505 in the stem "
               "path and 514 in the shared classifier; the reported model is "
               "290{,}943 in total, of which the branch and its fusion are "
               "93" + BS + "," + BS + "%. Counts quoted for a path exclude the "
               "classifier; counts quoted for a model include it, so the stem "
               "on its own is 20{,}019.",
               "parameter arithmetic, Method")

    tex = edit(tex,
               "The reported model is 290{,}943 parameters against 19{,}505 "
               "for the stem,",
               "The reported model is 290{,}943 parameters against 20{,}019 "
               "for the stem as a runnable model,",
               "parameter count, Results")

    tex = edit(tex,
               BS + "textbf{Parameter cost.} The branch is 238{,}028 "
               "parameters and the fusion it requires a further 32{,}896, "
               "against 19{,}505 in the stem path.",
               BS + "textbf{Parameter cost.} The branch is 238{,}028 "
               "parameters and the fusion it requires a further 32{,}896, "
               "against 19{,}505 in the stem path and 514 in the shared "
               "classifier.",
               "parameter cost, Limitations")

    # --------------------------------------------------------- 3. AdamW
    if "loshchilov" not in bib.lower():
        bib = bib.replace(
            "@inproceedings{adam,",
            "@inproceedings{adamw,\n"
            "  author    = {Loshchilov, Ilya and Hutter, Frank},\n"
            "  title     = {Decoupled weight decay regularization},\n"
            "  booktitle = {International Conference on Learning "
            "Representations (ICLR)},\n"
            "  year      = {2019},\n"
            "  url       = {https://openreview.net/forum?id=Bkg6RiCqY7},\n"
            "}\n\n@inproceedings{adam,", 1)
        io.open(BIB, "w", encoding="utf-8", newline="\n").write(bib)
        print("  ok  added the adamw entry to refs.bib")
    tex = edit(tex, "Optimiser & AdamW " + BS + "citep{adam},",
               "Optimiser & AdamW " + BS + "citep{adamw},", "AdamW citation")

    # ------------------------------------- 4. what "global" normalisation is
    tex = edit(tex, "Amplitude normalisation & global, per recording " + BS + BS,
               "Amplitude normalisation & global, fitting-split statistics "
               + BS + BS, "tab:hyper normalisation row")

    tex = edit(tex,
               "Amplitude normalisation is applied globally per recording, "
               "before the squaring stage, for the reason given in "
               "Section~" + BS + "ref{sec:stem}.",
               "Amplitude normalisation is applied globally, before the "
               "squaring stage, for the reason given in "
               "Section~" + BS + "ref{sec:stem}. The per-channel mean and "
               "standard deviation are computed on the fitting split alone "
               "and then applied unchanged to the calibration and test "
               "splits, so no statistic is taken from the recordings the "
               "model is evaluated on.",
               "normalisation scope, Preprocessing")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(tex)
    print("\n  19,505 + 238,028 + 32,896 + 514 = %d" % (19505 + 238028 + 32896 + 514))


if __name__ == "__main__":
    main()
