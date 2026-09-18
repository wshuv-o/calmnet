"""Can the adaptation rate be derived from the protocol instead of fitted?

The paper states Condition 1 as a band on the estimator memory, mu = 32/m, and
reports tau_blk per cohort. But the rate actually used was chosen by trying
m=0.2 and m=0.01 and keeping whichever worked, which is per-cohort tuning. A
reviewer will say so. The contribution only becomes a rule if tau_blk is
computed from the label sequence alone and the prescribed memory then predicts
which settings work, with accuracy never consulted.

This script does two things.

1. RECONCILE. Table 2 of the manuscript reports tau_blk = 18 (cohort A) and 309
   (cohort B) as median block duration in windows, over 109 and 7 blocks. A
   first pass here using per-subject MEAN run length on the fitting split gave
   47-67 and 840, the same ordering but different absolute values, and a rule
   asserted with the wrong tau_blk is worse than no rule. Every combination of
   statistic and pooling is printed so the manuscript definition can be
   identified rather than guessed.

2. TEST THE RULE. For whichever definition reproduces the published numbers,
   check whether mu > tau_blk separates the settings that worked from the ones
   that did not:

       cohort A, m=0.2   (mu=160)    measured +0.045
       cohort B, m=0.2   (mu=160)    measured -0.175
       cohort B, m=0.01  (mu=3200)   measured -0.026, recovered
       cohort C, m=0.2   (mu=160)    measured +0.064   (run on the 2060)

   Cohort C cannot be loaded on this machine, so its tau_blk has to come from
   the other machine before the rule covers all three.

Writes results/tau_blk_rule.json. Labels only; no model is trained or read.
"""
from __future__ import annotations

import io
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from exp_temporal import streams                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "tau_blk_rule.json")
BATCH = 32          # windows per estimator update, so memory mu = BATCH / m


def block_lengths(y, strm):
    """Length in windows of every contiguous single-class block."""
    runs = []
    for idx in strm:
        ys = np.asarray(y)[np.asarray(idx)]
        if len(ys) < 1:
            continue
        cut = np.flatnonzero(np.diff(ys) != 0)
        bounds = np.concatenate([[-1], cut, [len(ys) - 1]])
        runs += [int(v) for v in np.diff(bounds)]
    return runs


def summarise(name, per_subject_runs):
    """Every statistic-by-pooling combination, so the published one is
    identified rather than assumed."""
    pooled = [v for r in per_subject_runs.values() for v in r]
    per_sub_med = [float(np.median(r)) for r in per_subject_runs.values() if r]
    per_sub_mean = [float(np.mean(r)) for r in per_subject_runs.values() if r]
    return {
        "n_blocks_pooled": len(pooled),
        "pooled_median": float(np.median(pooled)) if pooled else None,
        "pooled_mean": float(np.mean(pooled)) if pooled else None,
        "longest": int(np.max(pooled)) if pooled else None,
        "median_of_per_subject_medians": float(np.median(per_sub_med))
        if per_sub_med else None,
        "mean_of_per_subject_means": float(np.mean(per_sub_mean))
        if per_sub_mean else None,
        "per_subject_median": {k: float(np.median(v))
                               for k, v in per_subject_runs.items() if v},
    }


def cohort_a(full_recording=True):
    """All windows of every session, not just the fitting split: the protocol's
    block structure is a property of the recording, not of our split."""
    from dataio import build_epochs, list_sessions
    out = {}
    for i in range(1, 8):
        sub = "sub-0%d" % i
        try:
            es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        strm = streams(es.session, es.task, es.segment)
        out[sub] = block_lengths(es.y.astype(int), strm)
    return out


def cohort_b():
    from dataio_mobi import build_subject, subjects
    out = {}
    for sub in subjects():
        try:
            es = build_subject(sub, win=2.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        g = (es.session if hasattr(es, "session") else es.trial).astype(int)
        t = getattr(es, "task", np.array(["mobi"] * len(g), object))
        seg = getattr(es, "segment", np.arange(len(g)))
        out[sub] = block_lengths(es.y.astype(int), streams(g, t, seg))
    return out


def main():
    res = {"_note": (
        "tau_blk computed from the label sequence only. No model is trained or "
        "read, and no accuracy is consulted, which is what would make a "
        "prescribed rate a derivation rather than a fit. Every "
        "statistic-by-pooling combination is reported because the manuscript "
        "value (median, 18 and 309) had to be reproduced before any rule could "
        "be built on it."),
        "batch": BATCH, "cohorts": {}}

    print("cohort A", flush=True)
    A = cohort_a()
    res["cohorts"]["A"] = summarise("A", A)
    print("cohort B", flush=True)
    B = cohort_b()
    res["cohorts"]["B"] = summarise("B", B)

    print("\n%-26s %-10s %-10s %-10s %-9s %s"
          % ("definition", "cohort A", "cohort B", "published A", "published B",
             "match"))
    pub = {"A": 18.0, "B": 309.0}
    for key, label in (("pooled_median", "pooled median"),
                       ("pooled_mean", "pooled mean"),
                       ("median_of_per_subject_medians",
                        "median of per-sub medians"),
                       ("mean_of_per_subject_means", "mean of per-sub means")):
        a, b = res["cohorts"]["A"][key], res["cohorts"]["B"][key]
        ok = (a is not None and b is not None
              and abs(a - pub["A"]) < 1.5 and abs(b - pub["B"]) < 15)
        print("%-26s %-10.1f %-10.1f %-10.0f %-9.0f %s"
              % (label, a, b, pub["A"], pub["B"], "YES" if ok else ""))
    print("\n  blocks counted: A %d (published 109), B %d (published 7)"
          % (res["cohorts"]["A"]["n_blocks_pooled"],
             res["cohorts"]["B"]["n_blocks_pooled"]))
    print("  longest block:  A %d (published 126), B %d (published 2211)"
          % (res["cohorts"]["A"]["longest"], res["cohorts"]["B"]["longest"]))

    # ---- the rule, against what was actually measured
    measured = [("A", 0.2, +0.045), ("B", 0.2, -0.175), ("B", 0.01, -0.026)]
    res["rule"] = []
    print("\n  rule: the branch is admissible when mu = %d/m exceeds tau_blk\n"
          % BATCH)
    print("  %-8s %-7s %-8s %-10s %-11s %-9s %s"
          % ("cohort", "m", "mu", "tau_blk", "mu>tau_blk", "measured", "agrees"))
    for coh, m, delta in measured:
        for key in ("pooled_median",):
            tb = res["cohorts"][coh][key]
            mu = BATCH / m
            pred = mu > tb
            ok = (pred == (delta > -0.05))
            res["rule"].append({"cohort": coh, "m": m, "mu": mu,
                                "tau_blk": tb, "admissible": bool(pred),
                                "measured_delta": delta, "agrees": bool(ok)})
            print("  %-8s %-7g %-8.0f %-10.1f %-11s %+-9.3f %s"
                  % (coh, m, mu, tb, pred, delta, "yes" if ok else "NO"))
    print("\n  cohort C cannot be loaded on this machine; its tau_blk must come "
          "from the other machine before the rule covers all three.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("\nwrote results/tau_blk_rule.json")


if __name__ == "__main__":
    main()
