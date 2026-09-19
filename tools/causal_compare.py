"""Look-ahead control: the stem+branch arm rerun with DN_TAN_CAUSAL=1 (each
batch mapped at the reference from before it), paired against the same stem
control as the reported test, and against the reported look-ahead run.
Participants paired, seeds averaged, two-sided Wilcoxon signed-rank, as in
tools/holm_five_cohorts.py. Cohorts whose causal file does not exist yet are
skipped."""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

R = r"C:\Users\Shuvo\Downloads\bci-review\calm-net\results\\"
J = lambda f: json.load(open(R + f, encoding="utf-8"))

# (cohort, causal file, reported branch file, stem file)
SRC = [("A", "a_tangent_causal_3seed.json", "a_tangent_3seed.json", "a_ablation_s12.json"),
       ("B", "b_tangent_causal_3seed.json", "b_tangent_3seed.json", "b_tangent_3seed.json"),
       ("C", "c_tangent_causal_3seed.json", "c_stem_tangent_3seed.json", "c_stem_tangent_3seed.json"),
       ("D", "d_tangent_causal_3seed.json", "d_bnci_3seed.json", "d_bnci_3seed.json"),
       ("E", "e_tangent_causal_3seed.json", "e_decoded_3seed.json", "e_decoded_3seed.json")]


def seedavg(d, arm, subs):
    return np.mean([[d["%s|s%d" % (arm, s)]["per_subject"][x]["acc"] for x in subs]
                    for s in (0, 1, 2)], axis=0)


def test(diff):
    return {"delta": float(diff.mean()), "higher": int((diff > 0).sum()),
            "n": len(diff), "p": float(wilcoxon(diff).pvalue)}


out = {}
for coh, fc, fb, fs in SRC:
    if not os.path.exists(R + fc):
        continue
    dc, db, ds = J(fc), J(fb), J(fs)
    subs = sorted(dc["dn_stem_tan|s0"]["per_subject"])
    causal = seedavg(dc, "dn_stem_tan", subs)
    ahead = seedavg(db, "dn_stem_tan", subs)
    stem = seedavg(ds, "dn_stem", subs)
    r = {"acc_causal": float(causal.mean()), "acc_lookahead": float(ahead.mean()),
         "acc_stem": float(stem.mean()),
         "causal_vs_stem": test(causal - stem),
         "lookahead_vs_stem": test(ahead - stem),
         "causal_vs_lookahead": test(causal - ahead)}
    out[coh] = r
    print("%s  stem %.4f  look-ahead %.4f  causal %.4f" % (
        coh, r["acc_stem"], r["acc_lookahead"], r["acc_causal"]))
    for k in ("lookahead_vs_stem", "causal_vs_stem", "causal_vs_lookahead"):
        t = r[k]
        print("   %-20s delta %+.4f  higher %2d/%-2d  p %.2e" % (
            k, t["delta"], t["higher"], t["n"], t["p"]))

json.dump(out, open(R + "causal_compare.json", "w"), indent=1)
print("wrote results/causal_compare.json")
