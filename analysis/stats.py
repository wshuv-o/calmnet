"""Paired statistics behind the tables: seed-averaged Wilcoxon, Holm, bootstrap CIs, AUC.

    python -m analysis.stats            # recompute the derived files, compare with results/
    python -m analysis.stats --write    # and overwrite them
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parent.parent / "results"
REV = RESULTS / "revision"
DUMP = Path(os.environ.get("CALMNET_DUMP", REV / "dump"))
SEEDS = (0, 1, 2)
N_BOOT = 10000

# submitted evaluation: (branch file, stem file) per cohort
MAIN = {"A": ("a_tangent_3seed.json", "a_ablation_s12.json"),
        "B": ("b_tangent_3seed.json", "b_tangent_3seed.json"),
        "C": ("c_stem_tangent_3seed.json", "c_stem_tangent_3seed.json"),
        "D": ("d_bnci_3seed.json", "d_bnci_3seed.json"),
        "E": ("e_decoded_3seed.json", "e_decoded_3seed.json")}
# offline filters, causal reference update only
CAUSAL_UPDATE = {c: "%s_tangent_causal_3seed.json" % c.lower() for c in "ABCDE"}
# fully causal chain, N = 32, m = 0.2 (cohort B has no 3seed file, its dump run is the same run)
FULL = {"A": "a_fullcausal_3seed", "B": "b_fullcausal_dump", "C": "c_fullcausal_3seed",
        "D": "d_fullcausal_3seed", "E": "e_fullcausal_3seed"}
# fully causal memory sweep: (file, N, m); the stem is FULL[cohort]
SWEEP = {
    "C": [("c_fullcausal_mu2", 2, 1.0), ("c_fullcausal_mu4", 4, 1.0), ("c_fullcausal_mu8", 8, 1.0),
          ("c_fullcausal_mu32", 32, 1.0), ("c_fullcausal_3seed", 32, 0.2),
          ("c_fullcausal_mu640", 32, 0.05),
          # mu = 4 with the reported update order; N = 4 at mu = 160
          ("c_fullcausal_mu4_noncausalupd", 4, 1.0), ("c_fullcausal_N4_mu160", 4, 0.025)],
    "D": [("d_fullcausal_mu32", 32, 1.0), ("d_fullcausal_3seed", 32, 0.2),
          ("d_fullcausal_mu640", 32, 0.05)],
    "E": [("e_fullcausal_mu8", 8, 1.0), ("e_fullcausal_mu32", 32, 1.0),
          ("e_fullcausal_3seed", 32, 0.2), ("e_fullcausal_mu640", 32, 0.05)],
    "A": [("a_fullcausal_mu8", 8, 1.0), ("a_fullcausal_mu32", 32, 1.0),
          ("a_fullcausal_3seed", 32, 0.2), ("a_fullcausal_mu640", 32, 0.05)],
    "B": [("b_fullcausal_dump", 32, 0.2), ("b_fullcausal_mu3200", 32, 0.01)]}
TAU = {"A": 18, "B": 214, "C": 5, "D": 1, "E": 13}       # results/revision/protocol_stats.json
ONSET = ("acc", "false_onsets_per_min", "missed_onsets", "latency_s")


def load(name):
    """A result file by name, relative to results/ (.json optional)."""
    p = RESULTS / (name if name.endswith(".json") else name + ".json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def runs(d, arm):
    """Seed runs of one arm, in key order."""
    return [d[k] for k in sorted(d) if k.split("|")[0] == arm]


def seed_accs(d, arm, metric="acc"):
    return np.array([r[metric] for r in runs(d, arm)], float)


def subjects(d, arm):
    return sorted(runs(d, arm)[0]["per_subject"])


def seedavg(d, arm, metric="acc", subs=None):
    """Per-participant mean over seeds; a participant missing from a seed is skipped there."""
    subs = subs or subjects(d, arm)
    rs = runs(d, arm)
    return np.array([np.nanmean([r["per_subject"].get(x, {}).get(metric, np.nan) for r in rs])
                     for x in subs])


def per_seed(d, arm, subs, metric="acc"):
    """participants x seeds"""
    return np.array([[d["%s|s%d" % (arm, s)]["per_subject"][x][metric] for s in SEEDS]
                     for x in subs], float)


def wilcoxon_p(d, drop_zeros=True):
    """Two-sided signed-rank p; zero differences removed first unless drop_zeros=False."""
    d = np.asarray(d, float)
    if drop_zeros:
        d = d[d != 0]
        if not len(d):
            return float("nan")
    return float(wilcoxon(d).pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p, input order kept."""
    pvals = np.asarray(pvals, float)
    adj, run = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(d, rng):
    """Percentile bootstrap 95% CI of the mean over participants."""
    boot = rng.choice(d, (N_BOOT, len(d))).mean(1)
    return [float(v) for v in np.percentile(boot, [2.5, 97.5])]


def test(diff):
    """Mean, count higher and p of a paired difference (zeros kept, as in the submitted tests)."""
    return {"delta": float(np.mean(diff)), "higher": int((diff > 0).sum()),
            "n": len(diff), "p": wilcoxon_p(diff, drop_zeros=False)}


def paired(a, b, rng=None, subs=None):
    """b minus a over participants with both values; CI only when rng is given."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = b[keep] - a[keep]
    if not len(d):
        return None
    out = {"stem": float(a[keep].mean()), "branch": float(b[keep].mean()),
           "delta": float(d.mean()), "higher": int((d > 0).sum()),
           "lower": int((d < 0).sum()), "n": int(len(d)), "p": wilcoxon_p(d)}
    if rng is not None:
        out["ci95"] = boot_ci(d, rng)
    if subs is not None:
        out["per_subject"] = {s: float(v) for s, v, k in zip(subs, b - a, keep) if k}
    return out


def branch_vs_stem(fb, fs, metric="acc", stem_arm="stem", arm="stem_tangent"):
    """Seed-averaged per-participant difference, branch file against stem file."""
    db, ds = load(fb), load(fs)
    subs = subjects(db, arm)
    return subs, seedavg(db, arm, metric, subs) - seedavg(ds, stem_arm, metric, subs)


# ------------------------------------------------------------- derived files --
def five_cohort_holm():
    """Table tab:five: branch against stem, per cohort, Holm across the five."""
    out = {}
    for c, (fb, fs) in MAIN.items():
        _, d = branch_vs_stem(fb, fs)
        out[c] = test(d)
    for c, ph in zip(out, holm([v["p"] for v in out.values()])):
        out[c]["p_holm"] = float(ph)
    return out


def causal_compare():
    """Causal update order against the stem and against the reported (look-ahead) order."""
    out = {}
    for c in "ABCDE":
        dc, db, ds = load(CAUSAL_UPDATE[c]), load(MAIN[c][0]), load(MAIN[c][1])
        subs = subjects(dc, "stem_tangent")
        causal = seedavg(dc, "stem_tangent", subs=subs)
        ahead = seedavg(db, "stem_tangent", subs=subs)
        stem = seedavg(ds, "stem", subs=subs)
        r = {}
        if "stem|s0" in dc:
            # B and E: stem retrained beside the causal branch, the reported stem kept as a check
            r["acc_stem_reported"] = float(stem.mean())
            rerun = seedavg(dc, "stem", subs=subs)
            r["stem_rerun_vs_reported"] = test(rerun - stem)
            stem = rerun
        r.update(acc_causal=float(causal.mean()), acc_lookahead=float(ahead.mean()),
                 acc_stem=float(stem.mean()), causal_vs_stem=test(causal - stem),
                 lookahead_vs_stem=test(ahead - stem), causal_vs_lookahead=test(causal - ahead))
        out[c] = r
    return out


def revision_summary():
    """Fully causal chain: branch vs stem, reference initialisation, output filter (C, D, A, E)."""
    rng, out = np.random.default_rng(0), {}
    for c in "CDAE":
        fc = load(FULL[c])
        subs = subjects(fc, "stem")
        res = {"fullcausal": {m: paired(seedavg(fc, "stem", m, subs),
                                        seedavg(fc, "stem_tangent", m, subs), rng, subs)
                              for m in ONSET}}
        carried, stem = seedavg(fc, "stem_tangent", subs=subs), seedavg(fc, "stem", subs=subs)
        for mode in ("fitmean", "firstbatch"):
            v = seedavg(load("%s_fullcausal_init_%s" % (c.lower(), mode)), "stem_tangent", subs=subs)
            res["init_" + mode] = {"vs_stem": paired(stem, v, rng, subs),
                                   "vs_carried": paired(carried, v, rng, subs)}
        sm = load("%s_fullcausal_smooth" % c.lower())
        res["smooth"] = {m: paired(seedavg(sm, "stem", m, subs),
                                   seedavg(sm, "stem_tangent", m, subs), rng, subs)
                         for m in ONSET}
        out[c] = res
    return out


def init_vs_carried(c):
    """Supplement S6 row: each initialisation against the carried reference, no CI."""
    fc = load(FULL[c])
    subs = subjects(fc, "stem_tangent")
    carried = seedavg(fc, "stem_tangent", subs=subs)
    return {mode: paired(carried, seedavg(load("%s_fullcausal_init_%s" % (c.lower(), mode)),
                                          "stem_tangent", subs=subs))
            for mode in ("fitmean", "firstbatch")}


def causal_sweep():
    """Fully causal memory sweep, each row against the stem of the same chain at N = 32."""
    rng, out = np.random.default_rng(0), {}
    for c, rows in SWEEP.items():
        ds = load(FULL[c])
        subs = subjects(ds, "stem")
        stem = seedavg(ds, "stem", subs=subs)
        res = []
        for f, n, m in rows:
            db = load(f)
            assert subjects(db, "stem_tangent") == subs, f
            r = paired(stem, seedavg(db, "stem_tangent", subs=subs), rng)
            r.update(file=f, N=n, m=m, mu=round(n / m), above=bool(n / m > TAU[c]))
            res.append(r)
        out[c] = {"tau_blk": TAU[c], "rows": res}
    return out


def deploy_causal():
    """Controller-level outcomes, fully causal, raw decisions and after the output filter."""
    rng, out = np.random.default_rng(0), {}
    for c in "ABCDE":
        out[c] = {}
        for kind, f in (("raw", FULL[c]), ("filtered", "%s_fullcausal_smooth" % c.lower())):
            d, block = load(f), {}
            for m in ("acc", "ece", "false_onsets_per_min", "missed_onsets", "latency_s"):
                s, b = seedavg(d, "stem", m), seedavg(d, "stem_tangent", m)
                block[m] = paired(s, b, rng)
                block[m]["stem_median"] = float(np.nanmedian(s))
                block[m]["branch_median"] = float(np.nanmedian(b))
            out[c][kind] = block
    return out


def _dump_files(run, arm, sub="*", seed="*"):
    """Per-window dumps of one arm, under the new or the original arm names."""
    for name in ({"stem": ("stem", "dn_stem"), "stem_tangent": ("stem_tangent", "dn_stem_tan")}[arm]):
        hits = sorted(glob.glob(os.path.join(run, "%s_s%s_%s.npz" % (name, seed, sub))))
        if hits:
            return hits, name
    return [], None


def auc_compare(dump=DUMP):
    """Threshold-free comparison: ROC AUC of the per-window posterior, fully causal dumps."""
    from sklearn.metrics import roc_auc_score
    rng, out = np.random.default_rng(0), {}
    for c in "CDAEB":
        run = os.path.join(str(dump), "%s_fullcausal_dump" % c.lower())
        files, name = _dump_files(run, "stem_tangent", seed="0")
        if not files:
            return None
        subs = sorted(os.path.basename(f)[len(name) + 4:-4] for f in files)

        def auc(arm, x):
            return float(np.mean([roc_auc_score(z["y"], z["p"][:, 1]) for z in
                                  map(np.load, _dump_files(run, arm, x)[0])]))
        s = np.array([auc("stem", x) for x in subs])
        b = np.array([auc("stem_tangent", x) for x in subs])
        d = b - s
        out[c] = {"stem": float(s.mean()), "branch": float(b.mean()), "delta": float(d.mean()),
                  "ci95": boot_ci(d, rng), "higher": int((d > 0).sum()), "n": len(d),
                  "p": wilcoxon_p(d, drop_zeros=False),
                  "per_subject": {x: [float(u), float(v)] for x, u, v in zip(subs, s, b)}}
    return out


DERIVED = {"five_cohort_holm.json": five_cohort_holm,
           "causal_compare.json": causal_compare,
           "revision/revision_summary.json": revision_summary,
           "revision/causal_sweep.json": causal_sweep,
           "revision/deploy_causal.json": deploy_causal,
           "revision/auc_compare.json": auc_compare}


def _diff(new, old, path=""):
    """Largest numeric difference between two nested results; keys missing from new are listed."""
    if isinstance(old, dict):
        worst, missing = 0.0, []
        for k, v in old.items():
            if k.startswith("_"):
                continue
            # old files name the two means a / b
            k2 = {"a": "stem", "b": "branch"}.get(k, k) if isinstance(new, dict) and k not in new else k
            if not isinstance(new, dict) or k2 not in new:
                missing.append(path + "/" + k)
                continue
            w, m = _diff(new[k2], v, path + "/" + k)
            worst, missing = max(worst, w), missing + m
        return worst, missing
    if isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            return float("inf"), [path]
        res = [_diff(a, b, path) for a, b in zip(new, old)]
        return max([r[0] for r in res] + [0.0]), sum((r[1] for r in res), [])
    if isinstance(old, bool) or not isinstance(old, (int, float)):
        return (0.0 if new == old else float("inf")), []
    if old != old and new != new:
        return 0.0, []
    return abs(float(new) - float(old)), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite the derived files")
    args = ap.parse_args()
    for name, fn in DERIVED.items():
        new = fn()
        if new is None:
            print("%-34s skipped (no per-window dumps in %s)" % (name, DUMP))
            continue
        p = RESULTS / name
        if p.exists():
            worst, missing = _diff(new, json.loads(p.read_text(encoding="utf-8")))
            print("%-34s max |diff| %.2e%s" % (name, worst,
                                              "  missing %s" % missing if missing else ""))
        if args.write:
            p.write_text(json.dumps(new, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
