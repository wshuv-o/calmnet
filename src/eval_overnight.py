"""Evaluate the overnight experiments. Written before any of their results existed.

The third-cohort verdicts apply the decision rule in
paper/PREREGISTRATION_cohort3.md literally: a mean-difference threshold and
nothing else. Wilcoxon tests and direction counts are reported alongside and
do not alter a verdict.

Missing result files are reported as pending, so the script can be run at any
point in the night.

Writes results/overnight_eval.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parent.parent / "results"


def arm(fname, key):
    p = RES / fname
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get(key)


def per_subject(a):
    return {s: float(r["acc"]) for s, r in (a or {}).get("per_subject", {}).items()}


def paired(a, b, label):
    """b minus a, over the participants present in both."""
    pa, pb = per_subject(a), per_subject(b)
    subs = sorted(set(pa) & set(pb))
    if not subs:
        return {"label": label, "status": "pending"}
    d = np.array([pb[s] - pa[s] for s in subs])
    rec = {"label": label, "n": len(subs),
           "mean_a": float(np.mean([pa[s] for s in subs])),
           "mean_b": float(np.mean([pb[s] for s in subs])),
           "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
           "b_higher": int((d > 0).sum()), "b_lower": int((d < 0).sum()),
           "per_subject_diff": {s: float(pb[s] - pa[s]) for s in subs}}
    if len(d) >= 5 and not np.allclose(d, 0):
        rec["wilcoxon_p"] = float(wilcoxon(d).pvalue)
    return rec


def main():
    out = {}

    # ---- Third cohort, pre-registered ------------------------------------
    gate = arm("cohort3_m0.2.json", "dn_gate|s0")
    ag02 = arm("cohort3_m0.2.json", "dn_noctx|s0")
    ag001 = arm("cohort3_m0.01.json", "dn_noctx|s0")

    p1 = paired(gate, ag02, "P1: align+gate (m=0.2) minus gate")
    if "mean_diff" in p1:
        p1["threshold"] = "mean_diff > -0.05"
        p1["verdict"] = "CONFIRMED" if p1["mean_diff"] > -0.05 else "FALSIFIED"
    p2 = paired(ag02, ag001, "P2: align+gate (m=0.01) minus align+gate (m=0.2)")
    if "mean_diff" in p2:
        p2["threshold"] = "mean_diff < +0.05"
        p2["verdict"] = "CONFIRMED" if p2["mean_diff"] < 0.05 else "FALSIFIED"
    out["cohort3"] = {"P1": p1, "P2": p2}

    # ---- Artefact control, cohort A, training task only ------------------
    orig = arm("icactl_orig.json", "dn_noctx|s0")
    ica = arm("icactl_ica.json", "dn_noctx|s0")
    out["ica_control"] = paired(orig, ica, "ICA-cleaned minus uncleaned (align+gate)")

    # ---- Align-only against the headline arm -----------------------------
    al = arm("align_only.json", "dn_align|s0")
    ref = arm("driftfix_ds.json", "dn_noctx|s0")        # same estimator, full data
    if al and ref:
        out["align_only"] = {"align_only_acc": al["acc"], "align_gate_acc": ref["acc"],
                             "gate_contribution": ref["acc"] - al["acc"],
                             "note": "unpaired: the align+gate reference predates per-subject logging"}
    else:
        out["align_only"] = {"status": "pending"}

    # ---- dn_nogate under the trace-normalised estimator ------------------
    ng = arm("driftfix_ds_nogate.json", "dn_nogate|s0")
    out["dn_nogate_fixed"] = {"acc": ng["acc"]} if ng else {"status": "pending"}

    (RES / "overnight_eval.json").write_text(json.dumps(out, indent=1))

    def show(r):
        if r.get("status") == "pending" or "mean_diff" not in r:
            return "pending"
        s = "n=%d  a=%.3f  b=%.3f  diff=%+.3f  (b higher %d / lower %d)" % (
            r["n"], r["mean_a"], r["mean_b"], r["mean_diff"], r["b_higher"], r["b_lower"])
        if "wilcoxon_p" in r:
            s += "  p=%.4f" % r["wilcoxon_p"]
        if "verdict" in r:
            s += "  -> %s (%s)" % (r["verdict"], r["threshold"])
        return s

    print("Third cohort (pre-registered)")
    print("  P1 ", show(p1))
    print("  P2 ", show(p2))
    print("Artefact control")
    print("    ", show(out["ica_control"]))
    print("Align-only")
    a = out["align_only"]
    print("    ", "pending" if "status" in a else
          "align-only %.3f vs align+gate %.3f -> gate adds %+.3f" %
          (a["align_only_acc"], a["align_gate_acc"], a["gate_contribution"]))
    ng_ = out["dn_nogate_fixed"]
    print("dn_nogate (trace-normalised)")
    print("    ", "pending" if "status" in ng_ else "%.3f" % ng_["acc"])


if __name__ == "__main__":
    main()
