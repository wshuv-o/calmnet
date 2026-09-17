"""Paired per-participant comparison of the cohort A ablation arms.

Why paired and why from one file
--------------------------------
The grand means invite the wrong conclusion. On cohort A seeds 1-2 the bare
stem reads 0.882 against the full model's 0.858, which looks like a 0.025 win,
but the seven participants differ from each other by more than the arms differ
from each other (sub-03 is at 0.99, sub-07 at 0.67). The unit of replication is
the participant, so the comparison has to be paired across participants, and a
difference of means with no test attached is not reportable under the writing
rules.

Every cell read here comes from a single JSON written by one queue, so the
estimator version, the harness and the seeds are identical across arms. That
matters: driftnet_ds.json holds a dn_stem seed-0 number of 0.827, but it was
written 2026-09-16 04:28, before the per-window covariance fix of 05:16, and
CLAUDE.md rule 2 puts it out of reach of these cells. Mixing it in would make
the stem look like it jumped 0.055 between seeds when the files are simply not
comparable.

Writes results/a_ablation_paired.json.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "a_ablation_s12.json")
OUT = os.path.join(ROOT, "results", "a_ablation_paired.json")
ARMS = ["dn_full", "dn_nogate", "dn_stem"]
SEEDS = ["1", "2"]
PAIRS = [("dn_stem", "dn_full"), ("dn_stem", "dn_nogate"),
         ("dn_nogate", "dn_full")]


def main():
    d = json.load(io.open(SRC, encoding="utf-8"))
    subs = sorted(d["%s|s%s" % (ARMS[0], SEEDS[0])]["per_subject"])

    def acc(arm, seed):
        ps = d["%s|s%s" % (arm, seed)]["per_subject"]
        return np.array([ps[s]["acc"] for s in subs])

    mean = {a: np.mean([acc(a, s) for s in SEEDS], axis=0) for a in ARMS}

    print("per-participant accuracy, cohort A, mean of seeds %s\n"
          % " and ".join(SEEDS))
    print("%-9s%s" % ("subject", "".join("%-12s" % a for a in ARMS)))
    for i, s in enumerate(subs):
        print("%-9s%s" % (s, "".join("%-12.3f" % mean[a][i] for a in ARMS)))
    print("%-9s%s" % ("MEAN", "".join("%-12.4f" % mean[a].mean()
                                      for a in ARMS)))

    out = {
        "_note": ("Paired across participants, which is the unit of "
                  "replication. Grand-mean differences without a paired test "
                  "are not reportable: the stem leads the full model by 0.025 "
                  "on the mean and the paired test does not distinguish that "
                  "from zero across seven participants. All cells from "
                  "a_ablation_s12.json, one estimator version, one harness."),
        "source": os.path.basename(SRC),
        "seeds": SEEDS,
        "subjects": subs,
        "per_arm": {}, "contrasts": {}, "seed_spread": {},
    }
    for a in ARMS:
        out["per_arm"][a] = {
            "grand_mean": float(mean[a].mean()),
            "per_subject": {s: float(mean[a][i]) for i, s in enumerate(subs)},
        }
        v = [float(acc(a, s).mean()) for s in SEEDS]
        out["seed_spread"][a] = {"per_seed": v, "range": float(max(v) - min(v))}

    print()
    for a, b in PAIRS:
        diff = mean[a] - mean[b]
        t = stats.ttest_rel(mean[a], mean[b])
        w = stats.wilcoxon(mean[a], mean[b])
        out["contrasts"]["%s_vs_%s" % (a, b)] = {
            "delta_mean": float(diff.mean()),
            "delta_sd": float(diff.std(ddof=1)),
            "paired_t_p": float(t.pvalue),
            "wilcoxon_p": float(w.pvalue),
            "wins": int((diff > 0).sum()), "n": len(subs),
            "largest_gain": [subs[int(np.argmax(diff))],
                             float(diff.max())],
            "largest_loss": [subs[int(np.argmin(diff))],
                             float(diff.min())],
        }
        print("%-10s vs %-10s %+.4f  paired t p=%.3f  wilcoxon p=%.3f  "
              "wins %d/%d  (best %s %+.3f, worst %s %+.3f)"
              % (a, b, diff.mean(), t.pvalue, w.pvalue, (diff > 0).sum(),
                 len(subs), subs[int(np.argmax(diff))], diff.max(),
                 subs[int(np.argmin(diff))], diff.min()))

    print("\nseed-to-seed spread within each arm:")
    for a in ARMS:
        v = out["seed_spread"][a]
        print("  %-10s %s  range %.4f"
              % (a, "  ".join("%.4f" % x for x in v["per_seed"]), v["range"]))

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    print("\nwrote results/a_ablation_paired.json")


if __name__ == "__main__":
    main()
