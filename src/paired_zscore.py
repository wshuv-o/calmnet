"""Participant-paired tests for the per-window normalisation finding.

zscore.json (cohort A, n=7) and zscore_mobi.json (cohort B, n=8) both store
per-participant accuracies for three representations under z1 (per-window
z-scoring on) and z0 (off). That is a matched pair per participant, so the
dose-response the stem design rests on can be tested directly rather than
compared as cohort means.

Holm correction is applied within each cohort, across the three
representations -- the set the dose-response conclusion is drawn from.

Writes results/paired_zscore.json.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

RES = os.path.join(os.path.dirname(__file__), "..", "results")

COHORTS = [("A", "zscore.json", 7), ("B", "zscore_mobi.json", 8)]
# Ordered by how much marginal power the representation carries.
REPS = [
    ("bandpower", "band power (pure marginal power)"),
    ("tangent_ea", "tangent space (power via covariance diagonal)"),
    ("corr_only", "correlation only (scale-free)"),
]


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def subj_acc(d, key):
    v = d.get(key)
    if not isinstance(v, dict):
        return {}
    out = {}
    for sub, r in v.get("per_subject", {}).items():
        out[sub] = float(r["acc"] if isinstance(r, dict) else r)
    return out


def main():
    allrec = []
    for cname, fn, expect_n in COHORTS:
        path = os.path.join(RES, fn)
        if not os.path.exists(path):
            print("[skip] missing", fn)
            continue
        with open(path) as f:
            d = json.load(f)

        recs, praw = [], []
        for rep, label in REPS:
            # z0 = normalisation removed, z1 = normalisation applied.
            off = subj_acc(d, "w2.0|%s|z0" % rep)
            on = subj_acc(d, "w2.0|%s|z1" % rep)
            subs = sorted(set(off) & set(on))
            if len(subs) < 5:
                print("[skip] %s/%s: %d paired" % (cname, rep, len(subs)))
                continue
            a = np.array([off[s] for s in subs])
            b = np.array([on[s] for s in subs])
            diff = a - b          # gain from REMOVING per-window normalisation
            if np.allclose(diff, 0):
                print("[skip] %s/%s: identical in every participant" % (cname, rep))
                continue
            stat, p = wilcoxon(a, b)
            recs.append({
                "cohort": cname, "representation": rep, "label": label,
                "n": len(subs),
                "acc_z1": float(b.mean()), "acc_z0": float(a.mean()),
                "mean_delta": float(diff.mean()),
                "better_in": int((diff > 0).sum()),
                "rank_biserial": float(np.sign(diff).sum() / len(diff)),
                "W": float(stat), "p_raw": float(p),
            })
            praw.append(float(p))

        for r, adj in zip(recs, holm(np.array(praw))):
            r["p_holm"] = float(adj)
        allrec.extend(recs)

        print("\nCohort %s (%s), removing per-window normalisation, Holm within cohort"
              % (cname, fn))
        print("%-46s %3s %8s %9s %8s %9s"
              % ("representation", "n", "mean d", "better k/n", "p raw", "p Holm"))
        for r in recs:
            print("%-46s %3d %+8.3f %9s %8.4f %9.4f"
                  % (r["label"], r["n"], r["mean_delta"],
                     "%d/%d" % (r["better_in"], r["n"]), r["p_raw"], r["p_holm"]))

    dest = os.path.join(RES, "paired_zscore.json")
    with open(dest, "w") as f:
        json.dump({"comparison": "z0 minus z1 (removing per-window normalisation)",
                   "tests": allrec}, f, indent=1)
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
