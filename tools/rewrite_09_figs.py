r"""Pass 9 of the rewrite: figures that still show the old architecture.

fig_cohorts is inserted into the five-cohort section as the headline figure: the
effect of the branch on every cohort, and the same five numbers against the
class-block length with the covariance update size drawn in. That is the
paper's argument, and until now it existed only as a table.

fig_ablation plots the arms of the old architecture. It is not removed, because
the ablation is reported against exactly those arms, but its caption said "the
proposed configuration" of an arm that is not proposed any more. Relabelled so a
reader knows they are looking at components that were tested and dropped.

fig_inversion has the same defect and the same fix.

    python tools/rewrite_09_figs.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
BS = chr(92)


def main():
    s = io.open(TEX, encoding="utf-8").read()

    # ---- headline figure into the five-cohort section
    anchor = ("Table~" + BS + "ref{tab:five} collects the branch against the "
              "convolutional stem")
    fig = (
        BS + "begin{figure*}[t]\n" + BS + "centering\n"
        + BS + "includegraphics[width=" + BS + "textwidth]"
        "{../results/fig_cohorts.pdf}\n"
        + BS + "caption{The branch on every cohort. (a) Change in balanced "
        "accuracy against the stem alone, paired across participants with "
        "seeds averaged within participant; bars are 95\\,% intervals from a "
        "paired bootstrap over participants, and the annotation gives how many "
        "participants improved. Cohorts are ordered by the class-block length "
        "of their protocol. (b) The same five effects against that block "
        "length, with the number of windows averaged into one covariance "
        "update drawn as a dashed line. The branch helps on every cohort whose "
        "blocks are shorter than one update and fails on the only one whose "
        "blocks are longer. Both axes of (b) are fixed before training: the "
        "block length is read from the labels and the update size is the batch "
        "size, so the separation uses no accuracy.}\n"
        + BS + "label{fig:cohorts}\n"
        + BS + "end{figure*}\n\n")
    s = s.replace(anchor, fig + anchor, 1)

    # ---- relabel the two figures that draw dropped components
    old_abl = ("Component ablation on cohort A, seed 0. (a) Balanced accuracy "
               "by arm;")
    new_abl = ("Component ablation on cohort~A, seed 0, over the components "
               "of Section~" + BS + "ref{sec:dropped} that were tested and "
               "are not in the reported model. It is retained because the "
               "ablation is reported against these arms; the reported model "
               "appears in Table~" + BS + "ref{tab:ablation_rep} and "
               "Fig.~" + BS + "ref{fig:cohorts}. (a) Balanced accuracy by "
               "arm;")
    if old_abl in s:
        s = s.replace(old_abl, new_abl, 1)
        print("  relabelled fig:ablation")

    old_inv = ("The same five arms on both cohorts, single seed. Solid heavy "
               "black is\nthe proposed configuration;")
    new_inv = ("The same five arms on both cohorts, single seed. None of these "
               "arms is the reported model; they are the components of "
               "Section~" + BS + "ref{sec:dropped}. Solid heavy black is the "
               "best of them on cohort~A;")
    if old_inv in s:
        s = s.replace(old_inv, new_inv, 1)
        print("  relabelled fig:inversion")
    else:
        # the line may be wrapped differently
        alt = "Solid heavy black is\nthe proposed configuration;"
        if alt in s:
            s = s.replace(alt, "None of these arms is the reported model; "
                          "they are the components of Section~" + BS
                          + "ref{sec:dropped}. Solid heavy black is\nthe best "
                          "of them on cohort~A;", 1)
            print("  relabelled fig:inversion (wrapped form)")
        else:
            print("  WARNING: fig:inversion caption not matched")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("PASS 9 done: figures")


if __name__ == "__main__":
    main()
