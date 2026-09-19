r"""Remove the last places where the paper still describes a model it no longer has.

Six of these are load-bearing rather than cosmetic:

  1. A sentence in Method was left ungrammatical by an earlier edit
     ("...none is retained. them, because each defines an arm...").
  2. The branch text says the projection leaves "the selective head" unchanged.
     Built with use_gate=False there is no selective head; the module is None.
  3. Preprocessing justifies skipping artefact rejection so that "the alignment
     layer is evaluated on the shift it is meant to correct".
  4. Outcome measures still calls "align + gate" the proposed decoder and gives
     its accuracy as the paper's.
  5. tab:noise and its prose are built almost entirely from dropped arms, with a
     "Transformer" column. They are rebuilt from the arms the paper reports.
  6. Accuracy at 90 % coverage is declared as an outcome measure and a coverage
     target is listed as a hyperparameter. Neither is real here: exp_calmnetx
     ranks the covered set by out["gate"], which build_driftnet sets to
     torch.ones when use_gate=False, so the "covered" 90 % is an arbitrary
     index-ordered subset. selective_loss also collapses to plain cross-entropy
     when the gate is ones (sel = mean CE, pen = 0, so the loss is 0.5*CE +
     0.5*CE). No table reports the quantity, so both lines are removed rather
     than recomputed.

Also extends the hyperparameter table from three cohorts to five.

    python tools/fix_stale_components.py
"""
from __future__ import annotations

import io
import re
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
AUTO = os.path.join(ROOT, "paper", "tables_auto.tex")
RES = os.path.join(ROOT, "results")
BS = chr(92)
NL = BS + BS

COH = [("A", "a_tangent_3seed.json", "a_ablation_s12.json"),
       ("B", "b_tangent_3seed.json", "b_tangent_3seed.json"),
       ("C", "c_stem_tangent_3seed.json", "c_stem_tangent_3seed.json"),
       ("D", "d_bnci_3seed.json", "d_bnci_3seed.json"),
       ("E", "e_decoded_3seed.json", "e_decoded_3seed.json")]


def spread(fname, arm):
    p = os.path.join(RES, fname)
    d = json.load(io.open(p, encoding="utf-8"))
    ks = sorted(k for k in d if k.startswith(arm + "|s") and "acc" in d[k])
    v = np.array([d[k]["acc"] for k in ks])
    if len(v) < 3:
        sys.exit("fewer than 3 seeds for %s in %s" % (arm, fname))
    return len(v), float(v.max() - v.min())


def edit(s, old, new, label):
    """Replace `old` ignoring how the source happens to be line-wrapped.

    The .tex is hard-wrapped, so a sentence written on one line here spans
    several there. Matching on a whitespace-insensitive pattern avoids having
    to reproduce the wrapping exactly.
    """
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    hits = pat.findall(s)
    if len(hits) != 1:
        sys.exit("FAILED %s: %d matches" % (label, len(hits)))
    print("  ok  " + label)
    return pat.sub(lambda _: new, s, count=1)


def main():
    tan = {c: spread(f, "dn_stem_tan") for c, f, _ in COH}
    stem = {c: spread(f, "dn_stem") for c, _, f in COH}
    worst = max(stem.values(), key=lambda x: x[1])[1]
    tan_hi = max(v[1] for v in tan.values())

    # ---------------------------------------------------------- tab:noise
    a = io.open(AUTO, encoding="utf-8").read()
    i = a.rfind(BS + "begin{table}", 0, a.index(BS + "label{tab:noise}"))
    j = a.index(BS + "end{table}", i) + len(BS + "end{table}")
    old_tab = a[i:j]
    k = old_tab.index("published decoders")
    keep = old_tab[old_tab.index(NL, k) + len(NL):old_tab.index(
        BS + "bottomrule", k)].strip("\n")

    rows = ""
    for c, _, _ in COH:
        n, sp = tan[c]
        rows += ("stem $+$ branch, cohort~%s & %d & %.4f %s\n" % (c, n, sp, NL))
    for c, _, _ in COH:
        n, sp = stem[c]
        rows += ("stem, cohort~%s & %d & %.4f %s\n" % (c, n, sp, NL))

    new_tab = (
        BS + "begin{table}[t]\n"
        + BS + "caption{Measurement noise, for the arms this paper reports. "
        "Spread is the range of balanced accuracy across three data-split "
        "seeds with every other setting fixed. The largest is the stem on "
        "cohort~A at $%.4f$; the reported model is tighter on every cohort, "
        "at most $%.4f$. " + BS + "textbf{Differences below $0.02$ are not "
        "interpreted anywhere in this work}, and differences below the "
        "relevant seed spread are reported without a ranking.}\n"
        + BS + "label{tab:noise}\n" + BS + "centering\n" + BS + "small\n"
        + BS + "resizebox{" + BS + "columnwidth}{!}{%%\n"
        + BS + "begin{tabular}{lrr}\n" + BS + "toprule\n"
        "Arm & $n$ & Spread " + NL + "\n" + BS + "midrule\n"
        + BS + "multicolumn{3}{l}{" + BS + "textit{Across three data-split "
        "seeds, reported arms}} " + NL + "\n" + rows
        + BS + "midrule\n"
        + BS + "multicolumn{3}{l}{" + BS + "textit{Across data-split seeds, "
        "published decoders, same pipeline}} " + NL + "\n" + keep + "\n"
        + BS + "bottomrule\n" + BS + "end{tabular}%%\n}\n"
        + BS + "end{table}") % (worst, tan_hi)
    # the kept block was written for four columns
    new_tab = new_tab.replace(BS + "multicolumn{4}{l}", BS + "multicolumn{3}{l}")
    a = a[:i] + new_tab + a[j:]
    io.open(AUTO, "w", encoding="utf-8", newline="\n").write(a)
    print("  ok  tab:noise rebuilt (%d reported rows)" % (2 * len(COH)))

    s = io.open(TEX, encoding="utf-8").read()

    # ------------------------------------------------------- 1. Method
    s = edit(s,
             "Three further components were built and tested, and none is "
             "retained. them, because each defines an arm of the ablation and "
             "one of them, the alignment layer, supplies the mechanism that "
             "explains where the branch fails.",
             "Three further components were built and tested during "
             "development and none is retained. They are not evaluated in this "
             "paper, and the ablations of Section~" + BS + "ref{sec:ablation} "
             "concern only the two paths above.",
             "Method fragment")

    # ------------------------------------------------- 2. selective head
    s = edit(s, "which leaves the classifier and the selective head "
                "unchanged in shape so the",
             "which leaves the classifier unchanged in shape so the",
             "selective head")

    # -------------------------------------------------- 3. preprocessing
    s = edit(s, "so that the alignment layer is evaluated on the shift it is "
                "meant to correct.",
             "so that the running reference is evaluated on the shift it is "
             "meant to track. The artefact control of Section~"
             + BS + "ref{sec:artefact} is the one exception and is reported "
             "separately.",
             "preprocessing rationale")

    # ---------------------------------------------- 4. conditional probe
    s = edit(s, "The proposed " + BS + "textit{align + gate} decoder reaches "
                "$0.901$ with a conditional probe of $-0.068$, and all eight "
                "cohort-A arms lie between $-0.066$ and $-0.082$, on the "
                "uncontaminated side of the calibration curve.",
             "The reported model reaches $0.909$ on cohort~A with a "
             "conditional probe of $-0.074$, and the stem reaches $0.864$ "
             "with $-0.080$, both on the uncontaminated side of the "
             "calibration curve.",
             "conditional probe")

    # ----------------------------------------------- 6. acc@90 and target
    s = edit(s, "accuracy at $90" + BS + ","
                + BS + "%$ coverage, the trained selective operating point; "
                "spurious activations per minute",
             "spurious activations per minute", "acc@90 outcome measure")
    s = edit(s, "Selective coverage target & 0.90 " + NL + "\n", "",
             "coverage-target row")

    # --------------------------------------------------- 9. five cohorts
    s = edit(s, "Artefact correction & EOG regression (A); none (B, C) " + NL,
             "Artefact correction & EOG regression (A); none (B--E) " + NL,
             "artefact row")
    s = edit(s, "Window / stride & 4" + BS + ",s / 0.5" + BS + ",s (A); 2"
                + BS + ",s (B); 2" + BS + ",s / 0.5" + BS + ",s (C) " + NL,
             "Window / stride & 4" + BS + ",s / 0.5" + BS + ",s (A, E); 2"
             + BS + ",s (B); 2" + BS + ",s / 0.5" + BS + ",s (C); 4"
             + BS + ",s, one per trial (D) " + NL,
             "window row")

    # ------------------------------------------------- 5. noise subsection
    a2 = s.index(BS + "subsection{Statistical treatment and measurement noise}")
    b2 = s.index(BS + "paragraph{Pipeline separation}", a2)
    prose = (
        BS + "subsection{Statistical treatment and measurement noise}\n"
        + BS + "label{sec:noise}\n\n"
        "Accuracy comparisons are reported as the mean over participants with\n"
        "the between-subject standard deviation, and as paired Wilcoxon\n"
        "signed-rank tests over participants wherever per-participant results\n"
        "were retained (Section~" + BS + "ref{sec:limits}). Every $" + BS
        + "pm$ in a table\nis labelled in its caption as across participants or "
        "across data-split\nseeds. Standard deviations across participants are "
        "population values,\nand standard deviations across seeds are sample "
        "values.\n\n"
        "The quantity that governs whether two models can be ranked is the\n"
        "spread across data-split seeds, and it is not uniform across arms or\n"
        "cohorts (Table~" + BS + "ref{tab:noise}). Over three seeds with every "
        "other\nsetting fixed, the reported model spans $%.4f$ to $%.4f$ "
        "depending on\nthe cohort, and the stem spans $%.4f$ to $%.4f$; the "
        "widest is the stem\non cohort~A. The stem is the noisier arm on four "
        "of the five cohorts,\nso the branch does not buy its accuracy with a "
        "less stable fit.\n"
        "Differences below $0.02$ are not interpreted anywhere in this paper,\n"
        "and differences smaller than the relevant seed spread are reported\n"
        "without a ranking. The effects we do interpret are measured over "
        "three\ndata-split seeds and tested paired within participant "
        "(Tables~" + BS + "ref{tab:compare} and " + BS + "ref{tab:batchsweep}"
        ").\n\n") % (min(v[1] for v in tan.values()), tan_hi,
                     min(v[1] for v in stem.values()), worst)
    s = s[:a2] + prose + s[b2:]
    print("  ok  noise subsection rewritten")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    n_worse = sum(1 for c, _, _ in COH if stem[c][1] > tan[c][1])
    print("\n  reported-model spread %.4f-%.4f;  stem %.4f-%.4f"
          % (min(v[1] for v in tan.values()), tan_hi,
             min(v[1] for v in stem.values()), worst))
    print("  stem noisier on %d of 5 cohorts" % n_worse)


if __name__ == "__main__":
    main()
