r"""Add the two experiments a reviewer would ask for before we do.

Inference cost. The paper presents an online exoskeleton decoder and the branch
takes the model from 19,505 parameters to 290,943, so "can it run in the loop"
is a fair question that a parameter count does not answer. Measured by
src/bench_cost.py, not estimated.

Classical Riemannian baseline. The paper argues that second-order covariance
structure complements a log-power stem, and Table 3 tests that against neural
decoders only. A Riemannian reader will ask why not use the standard pipeline
instead, so src/exp_riemann_baseline.py runs it: tangent space at the
Riemannian mean with logistic regression, and MDM. Same loader, same session
splits, same seeds and the same metric code as every other arm, with CX_FULL=1
as the published decoders of Table 3 used, so the comparison is within-harness.

Cohort A only for now. Cohort C's recordings live on the other machine in this
project's split and were still downloading when the paper was finalised, so the
text says cohort A and says why.

    python tools/add_cost_and_riemann.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "paper", "cas_calmnet.tex")
RES = os.path.join(ROOT, "results")
BS = chr(92)
NL = BS + BS


def edit(s, old, new, label):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    n = len(pat.findall(s))
    if n != 1:
        sys.exit("FAILED %s: %d matches" % (label, n))
    print("  ok  " + label)
    return pat.sub(lambda m: m.group(0) + new, s, count=1)


def per(fn, arm, field="acc"):
    d = json.load(io.open(os.path.join(RES, fn), encoding="utf-8"))
    ks = sorted(k for k in d
                if k.startswith(arm + "|s") and "per_subject" in d[k])
    subs = sorted(d[ks[0]]["per_subject"])
    return subs, np.array([[d[k]["per_subject"][s][field] for s in subs]
                           for k in ks])


def main():
    B = json.load(io.open(os.path.join(RES, "bench_cost.json"),
                          encoding="utf-8"))
    st, br = B["arms"]["stem"], B["arms"]["stem+branch"]
    upd = B["branch_parts_cpu"]["update_ms"]

    sub, R_ts = per("a_riemann.json", "riem_ts")
    _, R_mdm = per("a_riemann.json", "riem_mdm")
    sub_o, O = per("a_tangent_3seed.json", "dn_stem_tan")
    sub_s, S = per("a_ablation_s12.json", "dn_stem")
    if not (sub == sub_o == sub_s):
        sys.exit("participant sets differ between arms")
    o, t, m, s_ = O.mean(0), R_ts.mean(0), R_mdm.mean(0), S.mean(0)
    w_ts = stats.wilcoxon(o, t).pvalue
    _, E_r = per("a_riemann.json", "riem_ts", "ece")
    _, E_o = per("a_tangent_3seed.json", "dn_stem_tan", "ece")

    tex = io.open(TEX, encoding="utf-8").read()

    # ------------------------------------------------ 1. inference cost
    cost = (
        "\n\n" + BS + "subsection{What the branch costs at inference}"
        + BS + "label{sec:cost}\n\n"
        "Parameters are the wrong measure of what the branch costs to run. Its "
        "expense is an eigendecomposition of a $60" + BS + "times60$ matrix, "
        "which no parameter count reflects and which a floating-point count "
        "misses as well. Table~" + BS + "ref{tab:cost} measures the model "
        "instead, at cohort~A's input shape, with no training.\n\n"
        + BS + "begin{table}[t]\n" + BS + "centering\n"
        + BS + "caption{Inference cost at cohort~A's input shape, eight "
        "windows per forward. Latency is the median over %d forwards after "
        "warm-up, batch~1, with CUDA synchronised; the 95th percentile is "
        "given for the online case. No training.}\n"
        + BS + "label{tab:cost}\n"
        + BS + "resizebox{" + BS + "columnwidth}{!}{%%\n"
        + BS + "begin{tabular}{lrrr}\n" + BS + "toprule\n"
        " & Stem & Stem $+$ branch & Ratio " + NL + "\n" + BS + "midrule\n"
        "Parameters & $%s$ & $%s$ & $%.1f" + BS + "times$ " + NL + "\n"
        "GFLOPs per forward & $%.2f$ & $%.2f$ & $%.2f" + BS + "times$ " + NL + "\n"
        "CPU latency (ms) & $%.1f$ & $%.1f$ & $%.2f" + BS + "times$ " + NL + "\n"
        "CPU 95th percentile (ms) & $%.1f$ & $%.1f$ & " + NL + "\n"
        "GPU latency (ms) & $%.2f$ & $%.2f$ & $%.1f" + BS + "times$ " + NL + "\n"
        "GPU peak memory (MB) & $%.0f$ & $%.0f$ & " + NL + "\n"
        + BS + "bottomrule\n" + BS + "end{tabular}%%\n}\n"
        + BS + "end{table}\n\n"
        "The decision rate sets the budget. Cohort~A advances one window every "
        "$0.5$" + BS + ",s, and one forward covering eight windows takes "
        "$%.1f$" + BS + ",ms on a CPU, with a 95th percentile of $%.1f$"
        + BS + ",ms. That is around %.0f" + BS + "," + BS + "%% of the "
        "interval, so the reported model runs in the loop on a CPU and needs "
        "no accelerator. The label-free reference update, the part with no "
        "analogue in the stem, is $%.2f$" + BS + ",ms of that.\n\n"
        "Two things in the table are worth stating because they cut in "
        "opposite directions. The branch multiplies the parameter count by "
        "$%.1f$ but the CPU latency by only $%.2f$, so parameters overstate "
        "what it costs to run. Floating-point operations understate it, rising "
        "by $%.2f" + BS + "times$, because the counter does not see the "
        "eigendecomposition. Latency is the honest measure and it is the one "
        "reported here. On the GPU the overhead is proportionally larger, "
        "$%.1f" + BS + "times$, because an eigendecomposition of a small "
        "matrix is latency-bound and the device cannot hide it behind "
        "throughput.\n") % (
        200,
        "{:,}".format(st["parameters"]).replace(",", "{,}"),
        "{:,}".format(br["parameters"]).replace(",", "{,}"),
        br["parameters"] / st["parameters"],
        st["flops_per_forward"] / 1e9, br["flops_per_forward"] / 1e9,
        br["flops_per_forward"] / st["flops_per_forward"],
        st["cpu_b1_ms"], br["cpu_b1_ms"], br["cpu_b1_ms"] / st["cpu_b1_ms"],
        st["cpu_b1_p95_ms"], br["cpu_b1_p95_ms"],
        st["cuda_b1_ms"], br["cuda_b1_ms"], br["cuda_b1_ms"] / st["cuda_b1_ms"],
        st["cuda_peak_mb_b1"], br["cuda_peak_mb_b1"],
        br["cpu_b1_ms"], br["cpu_b1_p95_ms"], 100 * br["cpu_b1_ms"] / 500.0,
        upd,
        br["parameters"] / st["parameters"],
        br["cpu_b1_ms"] / st["cpu_b1_ms"],
        br["flops_per_forward"] / st["flops_per_forward"],
        br["cuda_b1_ms"] / st["cuda_b1_ms"])

    tex = edit(tex, "We report it rather than the accuracy alone.", cost,
               "inference-cost subsection")

    # --------------------------------------- 2. classical Riemannian baseline
    riem = (
        "\n\nOne comparison in this table is with neural decoders only, and the "
        "branch reads a Riemannian representation, so the obvious question is "
        "whether the classical pipeline would do as well without a network. On "
        "cohort~A, in this harness and on these splits, it does not. Tangent "
        "space at the Riemannian mean of the fitting covariances followed by "
        "logistic regression reaches $%.3f$ over three seeds, and minimum "
        "distance to the Riemannian mean reaches $%.3f$, against $%.3f$ for "
        "the reported model: a paired difference of $%+.3f$, higher in %d of "
        "the %d participants ($p=%.3f$). Expected calibration error is also "
        "worse, $%.3f$ against $%.3f$. The classical pipeline sits below the "
        "convolutional stem alone as well, at $%.3f$, though with seven "
        "participants that gap is not significant. Its reference is the "
        "Riemannian mean of the fitting split, fitted once and then held "
        "fixed, which is the static counterpart of the running estimate "
        "Section~" + BS + "ref{sec:rate} studies. This was run on cohort~A "
        "only.\n") % (
        t.mean(), m.mean(), o.mean(), (o - t).mean(), int((o > t).sum()),
        len(o), w_ts, E_r.mean(0).mean(), E_o.mean(0).mean(), s_.mean())

    tex = edit(tex, "Nothing here is a claim about efficiency.", riem,
               "Riemannian paragraph")

    # ------------------------------------------------------ 3. limitations
    lim = (" " + BS + "textbf{The classical Riemannian comparison is one "
           "cohort.} Tangent space with logistic regression and MDM were run "
           "on cohort~A only (Section~" + BS + "ref{sec:master}), so the "
           "margin over them is not established on the held-out cohort where "
           "the accuracy claim rests.")
    tex = edit(tex, "and the between-subject SD of $0.06$ to $0.13$ is "
                    "large relative to the differences between arms.", lim,
               "limitations note")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(tex)
    print("\n  cost: %.1f ms CPU, %.0f%% of a 500 ms budget"
          % (br["cpu_b1_ms"], 100 * br["cpu_b1_ms"] / 500.0))
    print("  riemann: ts %.4f  mdm %.4f  vs ours %.4f  (p=%.4f)"
          % (t.mean(), m.mean(), o.mean(), w_ts))


if __name__ == "__main__":
    main()
