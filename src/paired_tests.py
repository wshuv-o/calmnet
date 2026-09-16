"""Paired Wilcoxon signed-rank tests over the per-participant rows.

The rate-sweep JSONs store cohort means only, so nothing paired can be run on
them without a re-run. The temporal-control family is different: it has both
per-participant rows and four seeds, so a participant-paired test is available
now. Seeds are averaged per participant first, which removes run-to-run noise
from the pairing rather than treating each seed as an independent sample.

Holm correction is applied within the family, since that is the set of
comparisons the paper draws a conclusion from.

Writes results/paired_tests.json.
"""
import json
import os
from itertools import chain

import numpy as np
from scipy.stats import wilcoxon

RES = os.path.join(os.path.dirname(__file__), "..", "results")

# Four independent seeds of the same temporal-control experiment.
TEMPORAL = ["temporal_ds007788ctrl.json"] + [
    "temporal_ds007788ctrl_seed%d.json" % i for i in (1, 2, 3)]

BASELINE = "w2.0|none"
# The arms the paper draws a conclusion about, with the label used in the text.
ARMS = [
    ("w2.0|forward", "HMM forward filter"),
    ("w2.0|viterbi", "Viterbi decoding (offline only)"),
    ("w2.0|fwdshuf", "control: shuffled window order"),
]


def load(fn):
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def per_subject_acc(files, arm):
    """Mean accuracy per participant, averaged over whichever seeds ran it."""
    acc = {}
    for fn in files:
        d = load(fn)
        v = d.get(arm)
        if not isinstance(v, dict):
            continue
        for sub, r in v.get("per_subject", {}).items():
            # Some files store a metric dict per participant, others the
            # accuracy alone.
            val = r["acc"] if isinstance(r, dict) else r
            acc.setdefault(sub, []).append(float(val))
    return {s: float(np.mean(a)) for s, a in acc.items()}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    base = per_subject_acc(TEMPORAL, BASELINE)
    if not base:
        raise SystemExit("baseline arm %s carries no per-participant rows" % BASELINE)

    out, raw_p = [], []
    for arm, label in ARMS:
        cur = per_subject_acc(TEMPORAL, arm)
        subs = sorted(set(base) & set(cur))
        if len(subs) < 5:
            print("  [skip] %s: only %d paired participants" % (arm, len(subs)))
            continue
        a = np.array([cur[s] for s in subs])
        b = np.array([base[s] for s in subs])
        diff = a - b
        # Wilcoxon is undefined when every pair is identical.
        if np.allclose(diff, 0):
            print("  [skip] %s: identical to baseline in every participant" % arm)
            continue
        stat, p = wilcoxon(a, b)
        # Matched-pairs rank-biserial correlation: the standard effect size
        # for this test, and readable as "how one-sided the pairing is".
        r_rb = float(np.sign(diff).sum() / len(diff))
        out.append({
            "arm": arm,
            "label": label,
            "n": len(subs),
            "mean_delta": float(diff.mean()),
            "sd_delta": float(diff.std(ddof=1)),
            "better_in": int((diff > 0).sum()),
            "worse_in": int((diff < 0).sum()),
            "rank_biserial": r_rb,
            "W": float(stat),
            "p_raw": float(p),
        })
        raw_p.append(float(p))

    if out:
        for rec, adj in zip(out, holm(np.array(raw_p))):
            rec["p_holm"] = float(adj)

    print("Paired Wilcoxon vs %s, seeds averaged per participant, Holm within family" % BASELINE)
    print("%-34s %3s %9s %10s %8s %9s" % ("arm", "n", "mean d", "better k/n", "p raw", "p Holm"))
    for r in out:
        print("%-34s %3d %+9.3f %10s %8.4f %9.4f" % (
            r["label"], r["n"], r["mean_delta"],
            "%d/%d" % (r["better_in"], r["n"]), r["p_raw"], r["p_holm"]))

    dest = os.path.join(RES, "paired_tests.json")
    with open(dest, "w") as f:
        json.dump({"family": "temporal_controls", "baseline": BASELINE,
                   "seeds_averaged": len(TEMPORAL), "tests": out}, f, indent=1)
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
