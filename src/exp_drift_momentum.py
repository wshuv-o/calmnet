"""Drift reduction as a function of adaptation rate, with the corrected metric.

Discussion item 1 rests on the adaptation rate having been chosen because it
maximised drift removal (60 % against 25 %), a choice that cost accuracy. Both
of those numbers were measured with the trace normalisation that made the
momentum-0 control move by up to 122 %, so they need re-measuring before the
argument can quote them.

Covariances only. No training, no GPU, so this runs alongside a GPU queue.

Writes results/drift_momentum.json.
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_drift import cov, riemannian, run_subject          # noqa: E402
from dataio import build_epochs, list_sessions              # noqa: E402

MOMENTA = [float(x) for x in
           os.environ.get("DM_MOMENTA", "0.2,0.1,0.05,0.02,0.01").split(",")]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "drift_momentum.json")

def main():
    out = {"momenta": MOMENTA, "per_subject": {}}
    for sub in ["sub-0%d" % i for i in range(1, 8)]:
        try:
            es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True); continue
        X = es.X / (es.X.std() + 1e-30)
        g = es.session.astype(int)
        pres = sorted(set(int(v) for v in np.unique(g)))
        sess = [s for s in list_sessions(sub) if s in pres]
        fitg = sess[:3]
        Cfit = cov(X[np.isin(g, fitg)])
        raw = float(np.mean([riemannian(Cfit, cov(X[g == s])) for s in sess[3:]]))
        row = {"raw": raw}
        for m in MOMENTA:
            a = float(np.mean(run_subject(X, g, fitg, m)))
            row["m%g" % m] = {"aligned": a, "reduction": 1.0 - a / max(raw, 1e-12)}
        out["per_subject"][sub] = row
        print("  %-8s raw %.3f  " % (sub, raw)
              + "  ".join("m=%g: %.1f%%" % (m, 100 * row["m%g" % m]["reduction"])
                          for m in MOMENTA), flush=True)
    print(flush=True)
    summ = {}
    for m in MOMENTA:
        vs = [v["m%g" % m]["reduction"] for v in out["per_subject"].values()]
        summ["m%g" % m] = {"mean": float(np.mean(vs)), "sd": float(np.std(vs)),
                           "n": len(vs)}
        print("m=%-5g drift reduction %.1f%% +- %.1f%%  (n=%d)"
              % (m, 100 * np.mean(vs), 100 * np.std(vs), len(vs)), flush=True)
    out["summary"] = summ
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/drift_momentum.json", flush=True)

if __name__ == "__main__":
    main()
