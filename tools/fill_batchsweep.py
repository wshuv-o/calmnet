r"""Fill tab:batchsweep, and correct the quantity the manuscript names.

The 2060's cohort A runs settle a formulation question. I had written the
boundary as the class-block length against N, the number of windows in one
covariance update, because with N fixed at 32 that separated all five cohorts.
It separated them only because no cohort has a block between 32 and 160. At
N=8 with m=0.2 the memory is still 40, above cohort A's block of 18, and the
gain survives (+0.048); at N=8 with m=1.0 the memory is 8, below it, and the
effect reverses (-0.076, worse in every participant). So the quantity is the
estimator memory mu = N/m, which is how Condition 1 was derived in the first
place, and my N framing was a wrong simplification.

Controls are matched on N, because the batch size changes training. Momentum is
inert for the stem, which maintains no running statistic, so an m-only change
may reuse the control at the same N.

    python tools/fill_batchsweep.py
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
NL = BS + BS

# cohort, tau_blk, N, m, branch file, control file (matched on N)
ROWS = [
    ("A", 18, 8, 1.0, "a_batch8_m1_3seed.json", "a_batch8_3seed.json"),
    ("A", 18, 8, 0.2, "a_batch8_3seed.json", "a_batch8_3seed.json"),
    ("A", 18, 32, 0.2, "a_tangent_3seed.json", "a_ablation_s12.json"),
    ("B", 214, 32, 0.2, "b_tangent_3seed.json", "b_tangent_3seed.json"),
    ("B", 214, 256, 0.2, "b_batch256_3seed.json", "b_batch256_3seed.json"),
    ("B", 214, 32, 0.01, "b_tangent_m001_3seed.json", "b_tangent_3seed.json"),
]


def load(f):
    p = os.path.join(ROOT, "results", f)
    if not os.path.exists(p):
        sys.exit("missing: %s" % f)
    return json.load(io.open(p, encoding="utf-8"))


def paired(bf, cf, seeds=(0, 1, 2)):
    b, c = load(bf), load(cf)
    subs = sorted(c["dn_stem|s%d" % seeds[0]]["per_subject"])

    def g(d, k):
        return np.array([d[k]["per_subject"][s]["acc"] for s in subs])

    X = np.mean([g(b, "dn_stem_tan|s%d" % i) for i in seeds], axis=0)
    Y = np.mean([g(c, "dn_stem|s%d" % i) for i in seeds], axis=0)
    d = X - Y
    return d.mean(), stats.wilcoxon(X, Y).pvalue, int((d > 0).sum()), len(subs)


def main():
    out = []
    for coh, tau, N, m, bf, cf in ROWS:
        mu = int(round(N / m))
        dl, p, w, n = paired(bf, cf)
        out.append((coh, tau, N, m, mu, dl, p, w, n))

    body = ""
    for coh, tau, N, m, mu, dl, p, w, n in out:
        adm = "yes" if mu > tau else BS + "textbf{no}"
        body += ("%s & %d & %d & %g & %d & %s & $%+.3f$ & %d/%d & $%.3f$ %s\n"
                 % (coh, tau, N, m, mu, adm, dl, w, n, p, NL))

    s = io.open(TEX, encoding="utf-8").read()
    i = s.index(BS + "label{tab:batchsweep}")
    a = s.rindex(BS + "begin{table}", 0, i)
    b = s.index(BS + "end{table}", i) + len(BS + "end{table}")
    tab = (
        BS + "begin{table}[t]\n" + BS + "centering\n"
        + BS + "caption{The estimator memory against the class-block length, "
        "varied directly. $" + BS + "mu = N/m$ is the memory in windows, set "
        "by the number $N$ averaged into one covariance update and the "
        "momentum $m$; controls are matched on $N$, which changes training, "
        "while $m$ is inert for a stem that maintains no running statistic. "
        "The branch helps wherever $" + BS + "mu$ exceeds the block length and "
        "fails wherever it does not, on both cohorts and across a 400-fold "
        "range of $" + BS + "mu$. Cohort~A at $N=8$, $m=0.2$ is the row that "
        "settles the formulation: $N$ has fallen below the block length while "
        "$" + BS + "mu$ has not, and the gain survives, so the governing "
        "quantity is the memory and not the update size.}\n"
        + BS + "label{tab:batchsweep}\n"
        + BS + "begin{tabular}{lrrrrlrrr}\n" + BS + "toprule\n"
        "Cohort & $" + BS + "tau_{" + BS + "mathrm{blk}}$ & $N$ & $m$ & $"
        + BS + "mu$ & $" + BS + "mu>" + BS + "tau_{" + BS + "mathrm{blk}}$ & "
        "$" + BS + "Delta$ & Higher & $p$ " + NL + "\n" + BS + "midrule\n"
        + body
        + BS + "bottomrule\n" + BS + "end{tabular}\n" + BS + "end{table}")
    s = s[:a] + tab + s[b:]
    io.open(TEX, "w", encoding="utf-8", newline="\n").write(s)

    print("tab:batchsweep filled, %d rows\n" % len(out))
    print("  %-3s %-8s %-4s %-6s %-6s %-6s %-9s %-8s %s"
          % ("coh", "tau_blk", "N", "m", "mu", "adm", "delta", "p", "higher"))
    for coh, tau, N, m, mu, dl, p, w, n in out:
        print("  %-3s %-8d %-4d %-6g %-6d %-6s %+-9.4f %-8.4f %d/%d"
              % (coh, tau, N, m, mu, "yes" if mu > tau else "NO", dl, p, w, n))


if __name__ == "__main__":
    main()
