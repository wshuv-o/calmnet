r"""Rewrite the artefact control for the model the paper actually reports.

The control in the manuscript was measured on dn_noctx, which contained the
adaptive alignment layer and the selective gate. Neither is in the paper any
more, so that table said nothing about the reported configuration and was
presented as though it did. This replaces it with the same control run on
dn_stem_tan: same cohort, same seven participants, same cleaning, both arms
restricted to the training task so they differ only in the cleaning step.

The ICA cleaning statistics are unchanged and are recomputed here from
results/ica_components.csv, deduplicated by recording, because repeated runs
append.

    python tools/update_artefact.py
"""
from __future__ import annotations

import collections
import csv
import io
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
RES = os.path.join(ROOT, "results")
BS = chr(92)
NL = BS + BS
ARM = "dn_stem_tan"


def per_subject(fname):
    """Mean over seeds, per participant."""
    p = os.path.join(RES, fname)
    if not os.path.exists(p):
        sys.exit("missing " + fname)
    d = json.load(io.open(p, encoding="utf-8"))
    ks = sorted(k for k in d if k.startswith(ARM + "|s"))
    if not ks:
        sys.exit("no %s arm in %s" % (ARM, fname))
    subs = sorted(d[ks[0]]["per_subject"])
    M = np.array([[d[k]["per_subject"][s]["acc"] for s in subs] for k in ks])
    return subs, M.mean(axis=0), len(ks)


def cleaning_stats():
    rows = [r for r in csv.DictReader(
        io.open(os.path.join(RES, "ica_components.csv"), encoding="utf-8"))
        if "task-training" in r["recording"]]
    seen, uniq = set(), []
    for r in rows:                       # repeated runs append; keep one each
        if r["recording"] not in seen:
            seen.add(r["recording"])
            uniq.append(r)
    lab = collections.Counter(l for r in uniq
                              for l in r["removed_labels"].split(";") if l)
    tot = max(1, sum(lab.values()))
    muscle = collections.defaultdict(list)
    for r in uniq:
        muscle[r["recording"][:6]].append(
            sum(1 for l in r["removed_labels"].split(";")
                if l == "muscle artifact"))
    return (len(uniq),
            float(np.mean([int(r["n_removed"]) for r in uniq])),
            int(uniq[0]["n_components"]),
            100.0 * lab["eye blink"] / tot,
            100.0 * lab["muscle artifact"] / tot,
            {k: float(np.mean(v)) for k, v in muscle.items()})


def main():
    subs, orig, n_seed = per_subject("a_tan_icactl_orig.json")
    subs2, clean, _ = per_subject("a_tan_icactl_ica.json")
    if subs != subs2:
        sys.exit("participant sets differ: %s vs %s" % (subs, subs2))
    diff = clean - orig
    p = float(stats.wilcoxon(clean, orig).pvalue)
    n_rec, mean_rm, n_comp, pct_eye, pct_mus, mus = cleaning_stats()
    rho = float(stats.spearmanr([mus.get(s, np.nan) for s in subs], diff)[0])

    up = int((diff > 0).sum())
    dn = int((diff < 0).sum())
    worst = int(np.argmin(diff))
    most_mus = max(subs, key=lambda s: mus.get(s, -1))
    i_most = subs.index(most_mus)

    rows = ""
    for i, s in enumerate(subs):
        rows += ("%s & $%.3f$ & $%.3f$ & $%+.3f$ & %.1f %s\n"
                 % (s, orig[i], clean[i], diff[i], mus.get(s, float("nan")),
                    NL))
    rows += (BS + "midrule\nMean $" + BS + "pm$ SD & $%.3f" + BS
             + "pm%.3f$ & $%.3f" + BS + "pm%.3f$ & $%+.3f" + BS
             + "pm%.3f$ & %s\n") % (
        orig.mean(), orig.std(ddof=1), clean.mean(), clean.std(ddof=1),
        diff.mean(), diff.std(ddof=1), NL)

    sec = (
        BS + "subsection{Artefact control}\n" + BS + "label{sec:artefact}\n\n"
        "Walk and stop differ in movement, so accuracy on this task could arise\n"
        "from muscle or motion artefact. The reported configuration was therefore\n"
        "retrained after independent component analysis, with both arms\n"
        "restricted to the training task so that they differ only in the cleaning\n"
        "step. Each recording was re-referenced to the average, a copy\n"
        "band-passed to 1--45" + BS + ",Hz was decomposed into %d components with the\n"
        "extended Picard algorithm " + BS + "citep{picard}, and the components that\n"
        "ICLabel " + BS + "citep{iclabel} classified as eye, muscle, heart, line noise or\n"
        "channel noise were removed before band-passing to the decoding band.\n"
        "Across the %d training recordings, a mean of %.1f of %d components was\n"
        "removed, most often eye (%.0f" + BS + ",%% of removals) and muscle (%.0f" + BS
        + ",%%). Gait-related artefact is difficult to separate from cortical\n"
        "activity during walking " + BS + "citep{castermans2014,kline2015}, and ICLabel is\n"
        "not trained on it specifically, so this control narrows the question and\n"
        "leaves part of it open.\n\n"
        + BS + "begin{table}[t]\n"
        + BS + "caption{Artefact control on cohort~A, for the reported model.\n"
        "Balanced accuracy of stem $+$ branch on the training task, before and\n"
        "after ICA cleaning, mean over %d seeds, with the number of muscle\n"
        "components removed per recording. The last row is the mean $" + BS + "pm$ SD\n"
        "across participants.}\n"
        + BS + "label{tab:artefact}\n" + BS + "centering\n"
        + BS + "resizebox{" + BS + "columnwidth}{!}{%%\n"
        + BS + "begin{tabular}{lrrrr}\n" + BS + "toprule\n"
        "Participant & Uncleaned & Cleaned & Change & Muscle removed " + NL + "\n"
        + BS + "midrule\n" + rows + BS + "bottomrule\n"
        + BS + "end{tabular}%%\n}\n" + BS + "end{table}\n\n"
        "Cleaning changes balanced accuracy from $%.3f$ to $%.3f$, a paired\n"
        "difference of $%+.3f$ (higher in %d participants, lower in %d; Wilcoxon\n"
        "$p=%.2f$; Table~" + BS + "ref{tab:artefact}). Most of the accuracy survives, well\n"
        "above the $0.500$ chance level. The change does not follow the amount of\n"
        "muscle activity removed: the participant with the most muscle components\n"
        "removed, %.1f per recording, changes by $%+.3f$, and the Spearman\n"
        "correlation between muscle components removed and accuracy change is\n"
        "$" + BS + "rho=%+.2f$ across the seven participants. Dependence on muscle artefact\n"
        "would produce a negative correlation. Seven participants cannot settle\n"
        "the question, and average re-referencing is part of the cleaning, so the\n"
        "control constrains the artefact account rather than excluding it\n"
        "(Section~" + BS + "ref{sec:limits}).\n\n")

    sec = sec % (n_comp, n_rec, mean_rm, n_comp, pct_eye, pct_mus, n_seed,
                 orig.mean(), clean.mean(), diff.mean(), up, dn, p,
                 mus.get(most_mus, float("nan")), diff[i_most], rho)

    s = io.open(TEX, encoding="utf-8").read()
    a = s.index(BS + "subsection{Artefact control}")
    b = s.index(BS + "subsection{", a + 10)
    s = s[:a] + sec + s[b:]
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    print("tab:artefact rebuilt on %s, %d seeds" % (ARM, n_seed))
    print("  uncleaned %.4f -> cleaned %.4f   delta %+.4f  p=%.3f"
          % (orig.mean(), clean.mean(), diff.mean(), p))
    print("  higher in %d, lower in %d;  rho(muscle, change) = %+.2f"
          % (up, dn, rho))
    print("  worst: %s %+.3f" % (subs[worst], diff[worst]))


if __name__ == "__main__":
    main()
