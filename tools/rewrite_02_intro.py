r"""Pass 2 of the rewrite: Introduction and contribution list.

Three defects fixed here.

The third framing paragraph motivated abstention through a selective head. That
head is not in the reported model, so the motivation is replaced with the one
the model actually answers: a log-power stem is the diagonal of the spatial
covariance and discards how channels covary.

The six contributions were all about the alignment layer, which is also not in
the model. They are rewritten around the branch, with the alignment work
retained as the mechanism that explains the branch's one failure.

The closing sentence disclaimed any accuracy advantage. That was true when
written and is now false: the branch beats all eight published decoders on every
cohort where the comparison was run.

    python tools/rewrite_02_intro.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def load(f):
    p = os.path.join(ROOT, "results", f)
    if not os.path.exists(p):
        sys.exit("missing: %s" % f)
    return json.load(io.open(p, encoding="utf-8"))


def paired(df, arm, cf, ctl, seeds=(0, 1, 2)):
    d, c = load(df), load(cf)
    subs = sorted(c["%s|s%d" % (ctl, seeds[0])]["per_subject"])

    def g(dd, k):
        return np.array([dd[k]["per_subject"][s]["acc"] for s in subs])

    X = np.mean([g(d, "%s|s%d" % (arm, i)) for i in seeds], axis=0)
    Y = np.mean([g(c, "%s|s%d" % (ctl, i)) for i in seeds], axis=0)
    return dict(acc=X.mean(), ctl=Y.mean(), delta=(X - Y).mean(),
                p=stats.wilcoxon(X, Y).pvalue, w=int((X - Y > 0).sum()),
                n=len(subs))


def main():
    C = paired("c_stem_tangent_3seed.json", "dn_stem_tan",
               "c_stem_tangent_3seed.json", "dn_stem")
    B = paired("b_tangent_3seed.json", "dn_stem_tan",
               "b_tangent_3seed.json", "dn_stem")

    s = io.open(TEX, encoding="utf-8").read()

    # ---- replace the abstention framing paragraph
    i = s.index(BS + "paragraph{Abstention}")
    j = s.index("Existing responses to drift", i)
    new_par = (
        BS + "paragraph{What a log-power stem cannot see} Compact EEG decoders "
        "reduce each window to log power per channel after spatial filtering "
        + BS + "citep{schirrmeister,eegnet}. That summary is the diagonal of "
        "the window's spatial covariance. The off-diagonal entries, how "
        "channels covary, are discarded, and covariance structure is the basis "
        "of the Riemannian methods that remain competitive on movement-intent "
        "tasks " + BS + "citep{barachant,congedo}. Whether a convolutional "
        "decoder is losing usable information at that step is an empirical "
        "question, and it is the one this paper answers.\n\n")
    s = s[:i] + new_par + s[j:]

    # ---- replace the contribution list
    i = s.index("Our contributions are:")
    j = s.index("%" + "=" * 77, i)
    contrib = (
        "Our contributions are:\n\n"
        + BS + "begin{enumerate}\n"
        + BS + "item Add a tangent-space branch beside the log-power stem: "
        "each window's covariance is shrunk, mapped to the tangent space at a "
        "reference estimated from the input alone, and projected into the "
        "classifier, with no labels used at inference "
        "(Section~" + BS + "ref{sec:tangent}).\n"
        + BS + "item Measure it on five cohorts and 51 participants with three "
        "seeds each. Accuracy rises on four, including $%(cd)+.3f$ in %(cw)d "
        "of %(cn)d participants of a cohort held out from every design "
        "decision (Section~" + BS + "ref{sec:fivecohort}).\n"
        + BS + "item Compare against eight published decoders trained in one "
        "pipeline with shared splits, seeds and metric code, and report the "
        "cohort where most of them fail to train rather than letting the "
        "resulting margins stand (Section~" + BS + "ref{sec:newbaselines}).\n"
        + BS + "item Identify the branch's failure mode and predict it from "
        "the protocol. The reference is a running mean over incoming windows, "
        "so it is safe only while one covariance update spans more than one "
        "class block; on the cohort where a block outlasts an update the "
        "branch costs $%(babs).3f$ and falls near chance "
        "(Condition~" + BS + "ref{prop:band}, Section~" + BS
        + "ref{sec:rate}).\n"
        + BS + "item Test that account causally by enlarging the covariance "
        "update at fixed block length, which recovers 74\\,%% of the loss "
        "without restoring parity (Section~" + BS + "ref{sec:fivecohort}).\n"
        + BS + "item Pre-register a stronger version of the same condition, "
        "one that sets the adaptation rate from the block length, and report "
        "it as falsified (Section~" + BS + "ref{sec:prereg}).\n"
        + BS + "end{enumerate}\n\n"
        "The branch costs parameters. It adds roughly 271{,}000 to a "
        "20{,}000-parameter stem, so the reported model is no longer compact "
        "and we do not claim parameter efficiency; Section~" + BS
        + "ref{sec:compare} reports the counts beside the accuracies so the "
        "trade is visible.\n\n"
    ) % dict(cd=C["delta"], cw=C["w"], cn=C["n"], babs=abs(B["delta"]))
    s = s[:i] + contrib + s[j:]

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("PASS 2 done: Introduction and contribution list")
    print("  - abstention framing replaced with what a log-power stem discards")
    print("  - six contributions rewritten around the branch")
    print("  - the no-higher-accuracy disclaimer removed; it is now false")
    print("  - parameter cost stated in the introduction: 271k on a 20k stem")


if __name__ == "__main__":
    main()
