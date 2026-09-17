"""Drift removal against adaptation rate, on COHORT B.

exp_drift_momentum.py measures this on cohort A and finds removal collapsing
from 45.9 % at m=0.2 to 0.2 % at m=0.01. That number has been used to argue
that cohort B's residual is explained: the rate that avoids class tracking
there is m=0.01, which is also the rate at which the layer removes nothing.

The argument needs the measurement on cohort B, not cohort A. The cohorts
differ in window length (2 s against 4 s), trial structure (3 trials on the
same day against 9 sessions weeks apart) and class balance, so the collapse
point is not transferable between them. This runs it where the claim is made.

Covariances only. No training, no GPU, so it runs alongside a GPU queue.

Writes results/drift_momentum_b.json.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_drift import cov, riemannian, run_subject          # noqa: E402

MOMENTA = [float(x) for x in
           os.environ.get("DM_MOMENTA", "0.2,0.1,0.05,0.02,0.01").split(",")]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "drift_momentum_b.json")


def main():
    from dataio_mobi import build_subject, subjects
    out = {"cohort": "B", "momenta": MOMENTA, "per_subject": {}}
    for sub in subjects():
        try:
            es = build_subject(sub, win=2.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        if es is None:
            continue
        X = es.X / (es.X.std() + 1e-30)
        g = (es.session if hasattr(es, "session") else es.trial).astype(int)
        order = sorted(set(int(v) for v in np.unique(g)))
        if len(order) < 2:
            print("  [skip] %s: one trial only" % sub, flush=True)
            continue
        fitg = order[:1]                      # fit trial 1, test the rest
        Cfit = cov(X[np.isin(g, fitg)])
        raw = float(np.mean([riemannian(Cfit, cov(X[g == s]))
                             for s in order[1:]]))
        row = {"raw": raw}
        for m in MOMENTA:
            a = float(np.mean(run_subject(X, g, fitg, m)))
            row["m%g" % m] = {"aligned": a,
                              "reduction": 1.0 - a / max(raw, 1e-12)}
        out["per_subject"][sub] = row
        print("  %-6s raw %.3f   " % (sub, raw)
              + "  ".join("m=%g: %.1f%%" % (m, 100 * row["m%g" % m]["reduction"])
                          for m in MOMENTA), flush=True)

    print(flush=True)
    summ = {}
    for m in MOMENTA:
        vs = [v["m%g" % m]["reduction"] for v in out["per_subject"].values()]
        if not vs:
            continue
        summ["m%g" % m] = {"mean": float(np.mean(vs)), "sd": float(np.std(vs)),
                           "n": len(vs)}
        print("m=%-5g drift reduction %.1f%% +- %.1f%%  (n=%d)"
              % (m, 100 * np.mean(vs), 100 * np.std(vs), len(vs)), flush=True)
    out["summary"] = summ
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/drift_momentum_b.json", flush=True)


if __name__ == "__main__":
    main()
