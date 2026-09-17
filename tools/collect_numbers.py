"""Every number in results/, grouped by harness and cohort, as one markdown file.

Written because the numbers needed to fill the manuscript's empty cells are
spread over thirty JSON files and were being transcribed by hand, which is how
the 25x parameter-count error and the mixed-estimator error both happened. This
regenerates results/ALL_NUMBERS.md from the JSONs, so a cell in the paper can
be traced to a file name and a key without anyone retyping a float.

The grouping is not cosmetic. CLAUDE.md rule 2 forbids comparing across
training harnesses (ATCNet scores 0.913 in harness C and 0.881 in harness F),
so the harnesses are separated by heading and the rule is restated at the top.

    python tools/collect_numbers.py
"""
from __future__ import annotations

import glob
import io
import json
import os
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# file -> (group, cohort, harness, note)
GROUPS = {
    "a_ablation_s12.json": ("Ablation, cohort A", "A", "C", "seeds 1-2, RTX 5080"),
    "a_ablation_s12_2060.json": ("Ablation, cohort A", "A", "C", "seeds 1-2, RTX 2060"),
    "a_align_s12.json": ("Ablation, cohort A", "A", "C", "seeds 1-2"),
    "a_aligngate_3seed.json": ("Ablation, cohort A", "A", "C", "align+gate, 3 seeds"),
    "a_gate_3seed.json": ("Ablation, cohort A", "A", "C", "gate only, 3 seeds"),
    "a_slow_noctx_3seed.json": ("Ablation, cohort A", "A", "C", "m=0.01, 3 seeds"),
    "align_only.json": ("Ablation, cohort A", "A", "C", "seed 0"),
    "driftnet_ds.json": ("Ablation, cohort A", "A", "C", "seed 0"),
    "driftseed1_ds.json": ("Ablation, cohort A", "A", "C", "seed 1"),
    "driftmom_ds_0.01.json": ("Ablation, cohort A", "A", "C", "m=0.01, seed 0"),
    "driftmom_ds_0.01_noctx.json": ("Ablation, cohort A", "A", "C", "m=0.01, seed 0"),
    "atcplus.json": ("Ablation, cohort A", "A", "C",
                     "CAUTION: the base row is STOCK ATCNet, not ours"),
    "atcours_ds.json": ("Ablation, cohort A", "A", "C", "ATCNet + our layer, seed 0"),
    "icactl_orig.json": ("ICA control, cohort A", "A", "C", "no ICA"),
    "icactl_ica.json": ("ICA control, cohort A", "A", "C", "with ICA"),

    "b_aligngate_m001_3seed.json": ("Ablation, cohort B", "B", "C", "m=0.01, 3 seeds"),
    "b_gate_aligngate_3seed.json": ("Ablation, cohort B", "B", "C", "3 seeds"),
    "b_floor_m0.02.json": ("Ablation, cohort B", "B", "C", "m=0.02, 3 seeds"),
    "b_floor_m0.10.json": ("Ablation, cohort B", "B", "C", "m=0.10, 3 seeds"),
    "driftnet_mobi.json": ("Ablation, cohort B", "B", "C", "seed 0"),
    "driftmom_0.05.json": ("Ablation, cohort B", "B", "C", "m=0.05, seed 0"),

    "bd_pipeline_c.json": ("Baselines (braindecode)", "A", "BD", "8 architectures"),
    "bd_cohort_b.json": ("Baselines (braindecode)", "B", "BD", "8 architectures"),
    "bd_cohort_c.json": ("Baselines (braindecode)", "C", "BD", "8 architectures"),
    "bd_eegnex_s12.json": ("Baselines (braindecode)", "A", "BD", "EEGNeX extra seeds"),

    "fullbench.json": ("Baselines (harness F)", "A", "F", "2 seeds"),
    "merge.json": ("Baselines (harness F)", "A", "F", "3 seeds"),
    "published.json": ("Baselines (harness F)", "A", "F",
                       "normalisation sweep, 1 seed"),
    "amp_ablation.json": ("Baselines (harness F)", "A", "F",
                          "mixed precision on/off"),
    "cohort3_m0.2.json": ("Ablation, cohort C", "C", "C", "m=0.2"),
    "cohort3_m0.01.json": ("Ablation, cohort C", "C", "C", "m=0.01"),
    "cohort3_rep_m0.2.json": ("Ablation, cohort C", "C", "C", "repeat, m=0.2"),
    "cohort3_rep_m0.01.json": ("Ablation, cohort C", "C", "C", "repeat, m=0.01"),

    "driftfix_ds.json": ("Ablation, cohort A", "A", "C", "post-fix estimator"),
    "driftfix_ds_nogate.json": ("Ablation, cohort A", "A", "C",
                                "post-fix estimator"),
    "driftmom_0.01.json": ("Ablation, cohort B", "B", "C", "m=0.01, seed 0"),
    "driftfix_mobi.json": ("Ablation, cohort B", "B", "C", "post-fix estimator"),
    "driftfix_mobi_m020.json": ("Ablation, cohort B", "B", "C",
                                "m=0.20, post-fix estimator"),

    # 2026-09-03, i.e. BEFORE the per-window covariance fix of 2026-09-16
    # 05:16. Rule 2 forbids putting this in a table beside anything above.
    "calmnet3.json": ("Superseded (pre-fix estimator)", "?", "C",
                      "PRE-FIX, do not compare with anything else here"),

    "spdnet.json": ("SPD baselines", "A", "F", "3 seeds"),
    "temporal.json": ("Window length", "A", "F", "single run"),
    "temporal_mobi.json": ("Window length", "B", "F", "single run"),
    "temporal_ds007788ctrl.json": ("Window length", "A", "F", "control"),
}
ORDER = ["Ablation, cohort A", "Ablation, cohort B", "Ablation, cohort C",
         "ICA control, cohort A",
         "Baselines (braindecode)", "Baselines (harness F)", "SPD baselines",
         "Window length", "Superseded (pre-fix estimator)"]


def arms_of(d):
    """Collapse arm|seed keys to per-arm mean, sd and n."""
    cells = {k: v for k, v in d.items()
             if "|s" in k and isinstance(v, dict) and v.get("acc") is not None}
    out = {}
    for k, v in cells.items():
        out.setdefault(k.split("|s", 1)[0], []).append(v)
    rows = []
    for a in sorted(out):
        vs = out[a]
        acc = [v["acc"] for v in vs]
        ece = [v["ece"] for v in vs if v.get("ece") is not None]
        a90 = [v["acc@90"] for v in vs if v.get("acc@90") is not None]
        rows.append({
            "arm": a, "n": len(acc), "acc": float(np.mean(acc)),
            "sd": float(np.std(acc, ddof=1)) if len(acc) > 1 else None,
            "ece": float(np.mean(ece)) if ece else None,
            "a90": float(np.mean(a90)) if a90 else None,
        })
    return rows


def fmt(r):
    sd = "%.4f" % r["sd"] if r["sd"] is not None else "single seed"
    return "| `%s` | %d | **%.4f** | %s | %s | %s |" % (
        r["arm"], r["n"], r["acc"], sd,
        "%.3f" % r["ece"] if r["ece"] is not None else "--",
        "%.3f" % r["a90"] if r["a90"] is not None else "--")


def main():
    buckets = {}
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        name = os.path.basename(p)
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rows = arms_of(d)
        if not rows:
            continue
        grp, coh, har, note = GROUPS.get(
            name, ("Unfiled", "?", "?", "not in the group table"))
        buckets.setdefault(grp, []).append((name, coh, har, note, rows))

    L = []
    L.append("# Every accuracy number in `results/`\n")
    L.append("Generated by `tools/collect_numbers.py` on %s. "
             "Do not edit by hand; regenerate.\n"
             % datetime.now().strftime("%Y-%m-%d %H:%M"))
    L.append("**Read before using any of this.** Three rules from `CLAUDE.md` "
             "decide whether two numbers here may be put in the same table.\n")
    L.append("1. **Never compare across harnesses.** Harness C "
             "(`exp_calmnetx.py`), harness F (`fullbench.json`) and the "
             "braindecode harness (BD) differ by up to 0.032 on the same "
             "architecture, which is larger than most effects here.")
    L.append("2. **Never compare across estimator versions.** The per-window "
             "covariance fix landed 2026-09-16 05:16.")
    L.append("3. **The noise floor is not uniform.** Arms with the cross-epoch "
             "transformer carry about 0.010 run-to-run spread; `dn_stem` is "
             "bit-identical across repeats. Differences below 0.02 in "
             "transformer arms are not interpreted.\n")
    L.append("`sd` is the sample standard deviation across seeds (ddof=1), so "
             "a single-seed row has none and must be labelled single-seed in "
             "the manuscript and in its caption.\n")

    for grp in ORDER + [g for g in sorted(buckets) if g not in ORDER]:
        if grp not in buckets:
            continue
        L.append("\n## %s\n" % grp)
        for name, coh, har, note, rows in sorted(buckets[grp]):
            L.append("\n**`%s`** -- cohort %s, harness %s (%s)\n"
                     % (name, coh, har, note))
            L.append("| arm | seeds | acc | sd | ECE | acc@90 |")
            L.append("|---|---|---|---|---|---|")
            L.extend(fmt(r) for r in rows)

    # ---- mechanism measurements, which are not arm|seed shaped
    L.append("\n## Mechanism measurements (covariances only, no decoder)\n")
    for f, title in (("drift_momentum.json", "Drift removal vs rate, cohort A"),
                     ("drift_momentum_b.json", "Drift removal vs rate, cohort B")):
        p = os.path.join(RES, f)
        if not os.path.exists(p):
            continue
        d = json.load(io.open(p, encoding="utf-8"))
        s = d.get("summary", {})
        L.append("\n**`%s`** -- %s\n" % (f, title))
        L.append("| momentum | memory (windows) | drift removed | sd | n |")
        L.append("|---|---|---|---|---|")
        for k in sorted(s, key=lambda x: -float(x[1:])):
            m = float(k[1:])
            L.append("| %g | %d | **%.1f%%** | %.1f%% | %d |"
                     % (m, round(32 / m), 100 * s[k]["mean"],
                        100 * s[k]["sd"], s[k]["n"]))
    L.append("\nThe two cohorts do **not** share a rate at which the layer "
             "stops removing drift: cohort A falls to 0.2% at m=0.01 while "
             "cohort B is still removing 11.1% there. Any quantitative "
             "statement about the ceiling has to be made per cohort.\n")

    out = os.path.join(RES, "ALL_NUMBERS.md")
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    n = sum(len(r) for v in buckets.values() for *_, r in v)
    print("wrote results/ALL_NUMBERS.md  (%d arm rows from %d files)"
          % (n, sum(len(v) for v in buckets.values())))
    for name, *_ in buckets.get("Unfiled", []):
        print("  UNFILED, add to GROUPS: %s" % name)


if __name__ == "__main__":
    main()
