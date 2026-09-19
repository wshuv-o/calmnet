r"""Cut every over-long caption down, and move what they said into the text.

Fourteen captions ran from 67 to 208 words, which is a paragraph inside a
caption. A caption should say what the reader is looking at; the argument about
what it means belongs in the running text, where a reader meets it in order.

Most of the discussion was already duplicated in the body, so it is simply
dropped from the caption. The parts that existed only in a caption are moved
into the nearest paragraph rather than deleted, and each move is asserted here
so that nothing is lost silently.

Also replaces the seven em-dashes in the document. All seven are the "not
applicable" marker in a table cell, so they become en-dashes, which is the
lighter conventional marker and leaves the document with no em-dash at all.

    python tools/shorten_captions.py
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
AUTO = os.path.join(ROOT, "paper", "tables_auto.tex")
BS = chr(92)

# ---------------------------------------------------------------- captions
CAPTIONS = {
    "fig:arch": (
        "The reported decoder. A window enters both paths: the stem gives log "
        "band power per channel, and the branch maps the window's covariance "
        "to the tangent space at a running reference and projects the result. "
        "The two are concatenated into one linear classifier. The dashed arrow "
        "is the reference update, which uses no labels and stays active at "
        "inference."),
    "fig:activations": (
        "What the model computes, measured rather than drawn. Forward hooks "
        "recorded activations at the four points Fig.~" + BS + "ref{fig:arch} "
        "names, on held-out windows of one cohort~A participant. (a--c) The "
        "tangent vector folded back into the $60" + BS + "times60$ symmetric "
        "matrix it came from, for Stop, Walk and their difference. (d) Class "
        "separation at each tap, as the mean standardised difference per unit. "
        "(e--h) The same four representations in their first two principal "
        "components."),
    "fig:topo": (
        "Where the branch's reference acts, and what the classes differ by. "
        "One cohort~A participant, first three sessions. The reference is "
        "built by the branch itself, with its own shrinkage and per-window "
        "trace normalisation, so these are the quantities the reported model "
        "holds at inference. (a) Channel power at the reference. (b) "
        "Per-channel gain of $" + BS + "mathbf{M}^{-1/2}$, which the tangent "
        "map applies. (c) Absolute 8--30" + BS + ",Hz power, Walk minus Stop."),
    "tab:ablation_rep": (
        "Ablation of the reported model on cohort~A, three data-split seeds. "
        "Rows vary whether the second-order branch is present and where its "
        "tangent reference comes from. SD is across seeds; $" + BS + "Delta$ "
        "and the test are paired against the stem across participants, with "
        "seeds averaged within participant."),
    "fig:cohortC": (
        "The branch on cohort~C, which was held out from every design "
        "decision, three seeds throughout. (a) Balanced accuracy, one line per "
        "participant, stem on the left and stem plus branch on the right, with "
        "the mean heavy. (b) The same change per participant, sorted, with "
        "whiskers giving that participant's standard deviation across the "
        "three seeds. (c) The same cohort for the eight published decoders of "
        "Table~" + BS + "ref{tab:compare}, trained in the same pipeline with "
        "the same splits and seeds, as mean and standard error over "
        "participants."),
    "fig:cohorts": (
        "The branch on every cohort. (a) Change in balanced accuracy against "
        "the stem alone, paired across participants with seeds averaged within "
        "participant; bars are 95" + BS + ","+ BS + "% intervals from a paired "
        "bootstrap over participants, and the annotation gives how many "
        "participants improved. Cohorts are ordered by the class-block length "
        "of their protocol. (b) The same five effects against that block "
        "length, with the estimator memory drawn as a dashed line."),
    "tab:five": (
        "The tangent-space branch against the stem alone, three seeds, paired "
        "Wilcoxon across participants. $" + BS + "tau_{" + BS + "mathrm{blk}}$ "
        "is the median single-class block in windows, read from the fitting "
        "labels before training, and the estimator memory is $" + BS + "mu = "
        "N/m = 160$ windows at the settings used throughout."),
    "tab:batchsweep": (
        "The estimator memory against the class-block length, varied directly. "
        "$" + BS + "mu = N/m$ is the memory in windows, set by the number $N$ "
        "averaged into one covariance update and the momentum $m$. Controls "
        "are matched on $N$, which changes training, while $m$ is inert for a "
        "stem that maintains no running statistic."),
    "tab:deploy": (
        "What a wearer would experience, all five cohorts, three seeds, mean "
        "over participants. ECE is expected calibration error; FA/min is "
        "spurious Walk commands per minute of standing; Missed is the fraction "
        "of true onsets never detected."),
    "tab:newbase": (
        "Published decoders on the two cohorts added last, mean and standard "
        "deviation over three seeds, same pipeline as Table~" + BS
        + "ref{tab:compare}. $^{" + BS + "dagger}$ marks a decoder left near "
        "chance for at least five of the nine cohort~D participants on at "
        "least one seed."),
    "tab:compare": (
        "The reported model, the stem it is built on, and eight published "
        "decoders trained in one pipeline with identical preprocessing, "
        "optimiser, schedule, early stopping, model selection and classifier "
        "head. Balanced accuracy, mean $" + BS + "pm$ SD across three "
        "data-split seeds. Cohorts A to E have 7, 8, 20, 9 and 7 participants. "
        "Bold marks a configuration of ours that exceeds every published "
        "decoder in that column. Parameter counts include the shared "
        "classifier at cohort~A's input shape. $" + BS + "dagger$ near chance "
        "($<0.55$) for at least five participants on a seed."),
    "tab:repr": (
        "Two-condition comparisons. The first block removes per-window "
        "amplitude normalisation across representations that differ in how "
        "much marginal power they carry. The second supplies a per-window "
        "amplitude feature, and the same feature with its window assignment "
        "shuffled. Single seed; condition and control accuracies are mean $"
        + BS + "pm$ SD across participants."),
    "tab:norm": (
        "Where amplitude normalisation is applied, across 6 published decoders "
        "on cohort~A, single seed, in an earlier training pipeline. All three "
        "columns are balanced accuracies, mean $" + BS + "pm$ SD across "
        "participants; best mean per row in bold."),
    "tab:noise": (
        "Measurement noise, for the arms this paper reports. Spread is the "
        "range of balanced accuracy across three data-split seeds with every "
        "other setting fixed."),
}

# ------------------------------------------------- what moves into the text
MOVES = [
    # (anchor already in the body, text appended after it, label)
    ("Nothing in that choice is fitted on a confirmation cohort.",
     "\n\nThe two paths are complementary inside the trained model, not only "
     "in the accuracy column. Training one cohort~A participant with the "
     "reported settings and recording activations with forward hooks on "
     "held-out windows (Fig.~" + BS + "ref{fig:activations}, held-out balanced "
     "accuracy $0.967$), class separation measured as the mean standardised "
     "difference per unit is $0.74$ at the stem and $1.22$ at the branch, and "
     "$2.41$ after the two are fused. Fusion exceeding either path alone is "
     "what complementary means here, and the tangent vector folded back into "
     "its matrix shows the structure sitting off the diagonal, which is what a "
     "log-power stem discards.",
     "fig:activations to sec:tanres"),

    ("On cohort~C, which no design decision touched, the branch adds $0.064$ "
     "and is higher in every one of the 20 participants.",
     " Two further readings of that cohort are in Fig.~" + BS
     + "ref{fig:cohortC}. The change exceeds a participant's own standard "
     "deviation across the three seeds in 16 of the 20, so for most of them "
     "the effect is larger than the noise the measurement already carries, and "
     "against the eight published decoders on the same cohort the branch leads "
     "the strongest of them, EEGNeX, by $+0.060$, higher in 18 of the 20 "
     "($p=1.3" + BS + "times10^{-4}$, paired Wilcoxon).",
     "fig:cohortC to sec:tanres"),

    ("so the control constrains the artefact account rather than excluding it "
     "(Section~" + BS + "ref{sec:limits}).",
     "\n\nWhere the reference acts is shown in Fig.~" + BS + "ref{fig:topo}. "
     "The tangent map amplifies the frontocentral midline most (AFz, Fz and F2 "
     "at $2.9$) and the temporal sites most exposed to jaw and neck muscle "
     "least (T7 and T8 at $1.9$). The class contrast is separate from that and "
     "is a property of the data: absolute 8--30" + BS + ",Hz power falls over "
     "left sensorimotor cortex during walking (C3 $-3.5$" + BS + ",dB, C1 "
     "$-3.0$, CP3 $-2.8$), which is the topography and the sign expected of mu "
     "and beta desynchronisation, and rises occipitally (O1 and O2 at $+2.1$), "
     "which is consistent with optic flow. Gross movement artefact would raise "
     "power during walking rather than lower it over sensorimotor cortex, so "
     "the sign of the contrast argues against an artefact account without "
     "settling it. It is one participant, and a power contrast is not an "
     "artefact rejection.",
     "fig:topo to sec:artefact"),

    ("Cohorts~D and~E were added after the method was fixed, and nothing was "
     "tuned on either.",
     " Both axes of Fig.~" + BS + "ref{fig:cohorts}(b) are fixed before "
     "training: the block length is read from the labels and the memory is "
     "fixed by the optimiser settings, so the separation uses no accuracy. The "
     "regime column of Table~" + BS + "ref{tab:five} follows from that "
     "comparison alone, for the same reason.",
     "fig:cohorts and tab:five to sec:fivecohort"),

    ("The upper bound is not measured here (Section~" + BS
     + "ref{sec:limits}).",
     " The branch helps wherever $" + BS + "mu$ exceeds the block length and "
     "fails wherever it does not, on both cohorts and across a 400-fold range "
     "of $" + BS + "mu$. Cohort~A at $N=8$, $m=0.2$ is the row that settles "
     "the formulation: $N$ has fallen below the block length while $" + BS
     + "mu$ has not, and the gain survives, so the governing quantity is the "
     "memory and not the update size.",
     "tab:batchsweep to sec:rate"),

    ("Those are the comparisons cohort~D actually supports, and all three hold "
     "after correction.",
     " The dagger marks a fragile regime rather than a fixed property of a "
     "decoder: running the identical criterion on an independent repeat of "
     "these cells on a second machine moves EEGNeX across the threshold, at "
     "four near-chance participants here and five there.",
     "tab:newbase to sec:newbaselines"),

    ("The exception is the spurious-activation rate on cohort~A, which rises "
     "from $0.70$ to $2.09$ per minute of standing.",
     " That rate is the quantity a per-window error rate hides, since one long "
     "false run and many scattered false windows score identically.",
     "tab:deploy to sec:deploy"),
]


def caption_span(s, label):
    """Locate the \\caption{...} belonging to \\label{label}."""
    li = s.index(BS + "label{" + label + "}")
    ci = s.rfind(BS + "caption{", 0, li)
    if ci == -1:
        sys.exit("no caption before " + label)
    i = ci + len(BS + "caption{")
    depth, j = 1, i
    while depth and j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    return i, j - 1


def words(t):
    return len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", t).split())


def main():
    tex = io.open(TEX, encoding="utf-8").read()
    auto = io.open(AUTO, encoding="utf-8").read()

    # ------------------------------------------------------ move first
    for anchor, addition, name in MOVES:
        # the .tex is hard-wrapped, so an anchor written on one line here
        # spans several there; match on a whitespace-insensitive pattern and
        # keep whatever the file actually contains
        pat = re.compile(r"\s+".join(re.escape(w) for w in anchor.split()))
        hits = pat.findall(tex)
        if len(hits) != 1:
            sys.exit("MOVE anchor %s matched %d times: %s"
                     % (name, len(hits), anchor[:70]))
        tex = pat.sub(lambda m: m.group(0) + addition, tex, count=1)
        print("  moved  %s" % name)

    # ------------------------------------------------ then shorten
    for label, new in CAPTIONS.items():
        target = "tex" if (BS + "label{" + label + "}") in tex else "auto"
        s = tex if target == "tex" else auto
        a, b = caption_span(s, label)
        old_n = words(s[a:b])
        s = s[:a] + new + s[b:]
        print("  caption %-18s %3d -> %3d words" % (label, old_n, words(new)))
        if target == "tex":
            tex = s
        else:
            auto = s

    # ------------------------------------------- em-dash to en-dash
    n_em = 0
    for name in ("tex", "auto"):
        s = tex if name == "tex" else auto
        hits = len(re.findall(r"(?<!-)---(?!-)", s))
        s = re.sub(r"(?<!-)---(?!-)", "--", s)
        n_em += hits
        if name == "tex":
            tex = s
        else:
            auto = s
    print("  em-dashes replaced: %d" % n_em)

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(tex)
    io.open(AUTO, "w", encoding="utf-8", newline="\n").write(auto)


if __name__ == "__main__":
    main()
