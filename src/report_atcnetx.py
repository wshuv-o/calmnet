"""Turn the ATCNet-X result JSONs into the paper's tables.

Printing only -- no re-computation happens here, so a table can never disagree
with the run that produced it. Every figure in the draft should be reproducible
by running this and pasting.

Paired within-seed differences are the headline statistic, not the means: split
seed alone moves accuracy by +-0.01-0.02 on this data, which is the same order
as the module effects, so an unpaired comparison of means cannot resolve them.
A difference is only reported as consistent when its sign is identical on every
seed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
ARMS = ["dn_full", "dn_noalign", "dn_noctx", "dn_nogate", "dn_stem",
        "base", "rate", "rate+ctx", "rate+gate", "full"]
METRICS = [("acc", "acc", "%.3f"), ("ece", "ECE", "%.3f"),
           ("acc_at_90", "acc@90", "%.3f"),
           ("false_onsets_per_min", "onset/min", "%.2f"),
           ("latency_s", "lat_s", "%.2f"),
           ("cond_r2", "cond_R2", "%+.3f")]


def load(name):
    p = RESULTS / name
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for k, v in d.items():
        arm, seed = k.rsplit("|s", 1)
        out.setdefault(arm, {})[int(seed)] = v
    return out


def table(by, title):
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)
    if not by:
        print("  (no results yet)")
        return
    hdr = "%-11s %3s" % ("arm", "n")
    for _, lab, _ in METRICS:
        hdr += " %10s" % lab
    print(hdr)
    print("-" * 94)
    for arm in ARMS:
        if arm not in by:
            continue
        seeds = sorted(by[arm])
        row = "%-11s %3d" % (arm, len(seeds))
        for key, _, fmt in METRICS:
            vals = [by[arm][s].get(key, float("nan")) for s in seeds]
            row += " %10s" % (fmt % float(np.nanmean(vals)))
        print(row)


def paired(by, base=None):
    """Within-seed differences against the no-module arm."""
    if base is None:
        # DriftNet ablates DOWN from the full model, so the reference is
        # dn_full; the ATCNet-X arms add UP from base.
        base = "dn_full" if "dn_full" in by else "base"
    if base not in by:
        return
    print("\nPAIRED vs `%s`, within seed" % base)
    print("%-11s %3s %18s %14s %16s" %
          ("arm", "n", "d_acc", "d_onset/min", "sign"))
    print("-" * 94)
    B = by[base]
    for arm in ARMS:
        if arm == base or arm not in by:
            continue
        c = sorted(set(by[arm]) & set(B))
        if not c:
            continue
        da = np.array([by[arm][s]["acc"] - B[s]["acc"] for s in c])
        do = np.array([by[arm][s]["false_onsets_per_min"]
                       - B[s]["false_onsets_per_min"] for s in c])
        same = bool(np.all(np.sign(da) == np.sign(da[0]))) if len(da) > 1 else None
        sign = {True: "consistent", False: "MIXED", None: "n=1"}[same]
        print("%-11s %3d %+11.3f +-%.3f %+14.2f %16s"
              % (arm, len(c), da.mean(), da.std(), do.mean(), sign))


def main():
    a = load(sys.argv[1] if len(sys.argv) > 1 else "driftnet_ds.json")
    b = load(sys.argv[2] if len(sys.argv) > 2 else "driftnet_mobi.json")
    table(a, "COHORT A -- ds007788 (7 subjects, 9 sessions, 4 s windows, full data)")
    paired(a)
    table(b, "COHORT B -- MoBI treadmill (8 subjects, 2 s windows) -- EXTERNAL VALIDATION")
    paired(b)

    # does any module transfer across cohorts?
    if a and b:
        print("\n" + "=" * 94)
        print("CROSS-COHORT: does a module help in BOTH?")
        print("=" * 94)
        print("%-11s %16s %16s %14s" % ("arm", "A d_acc", "B d_acc", "transfers"))
        print("-" * 94)
        ref = "dn_full" if "dn_full" in a else "base"
        # For DriftNet the arms ablate DOWN from dn_full, so a NEGATIVE delta
        # means the removed component was carrying its weight. Flip the sign so
        # "helps" reads the same way in both conventions.
        flip = -1.0 if ref == "dn_full" else 1.0
        for arm in ARMS:
            if arm == ref or arm not in a and arm not in b:
                continue
            ds = []
            for src in (a, b):
                if arm in src and ref in src:
                    c = sorted(set(src[arm]) & set(src[ref]))
                    ds.append(flip * np.mean([src[arm][s]["acc"] - src[ref][s]["acc"]
                                              for s in c]) if c else float("nan"))
                else:
                    ds.append(float("nan"))
            ok = "yes" if all(np.isfinite(ds)) and all(x > 0 for x in ds) else "no"
            print("%-11s %+16.3f %+16.3f %14s" % (arm, ds[0], ds[1], ok))
        print("")
        print("(for DriftNet arms the sign is flipped: a positive value means")
        print(" the component REMOVED in that arm was contributing.)")
    print()


if __name__ == "__main__":
    main()
