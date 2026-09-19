"""The five cohort-level tests of Table tab:five, recomputed from the result files
and Holm-corrected as one family. Stem + branch against the stem alone, paired
over participants with seeds averaged, two-sided Wilcoxon signed-rank."""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, r"C:\Users\Shuvo\Downloads\bci-review\calm-net\src")
from paired_tests import holm  # noqa: E402

R = r"C:\Users\Shuvo\Downloads\bci-review\calm-net\results\\"
J = lambda f: json.load(open(R + f, encoding="utf-8"))

# (cohort, file with dn_stem_tan, file with dn_stem)
SRC = [("A", "a_tangent_3seed.json", "a_ablation_s12.json"),
       ("B", "b_tangent_3seed.json", "b_tangent_3seed.json"),
       ("C", "c_stem_tangent_3seed.json", "c_stem_tangent_3seed.json"),
       ("D", "d_bnci_3seed.json", "d_bnci_3seed.json"),
       ("E", "e_decoded_3seed.json", "e_decoded_3seed.json")]


def seedavg(d, arm, subs):
    return np.mean([[d["%s|s%d" % (arm, s)]["per_subject"][x]["acc"] for x in subs]
                    for s in (0, 1, 2)], axis=0)


rows = []
for coh, ft, fs in SRC:
    dt, ds = J(ft), J(fs)
    subs = sorted(dt["dn_stem_tan|s0"]["per_subject"])
    diff = seedavg(dt, "dn_stem_tan", subs) - seedavg(ds, "dn_stem", subs)
    rows.append((coh, diff.mean(), int((diff > 0).sum()), len(diff), float(wilcoxon(diff).pvalue)))

adj = holm(np.array([r[4] for r in rows]))
out = {}
for (coh, dm, hi, n, p), ph in zip(rows, adj):
    print("%s  delta %+.4f  higher %2d/%-2d  raw p %.2e  Holm %.2e" % (coh, dm, hi, n, p, ph))
    out[coh] = {"delta": dm, "higher": hi, "n": n, "p": p, "p_holm": float(ph)}
json.dump(out, open(R + "five_cohort_holm.json", "w"), indent=1)
print("wrote results/five_cohort_holm.json")
