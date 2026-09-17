"""Tabulate results/temporal*.json into the numbers that go in the paper.

Kept deliberately small and printing-only: the experiment writes JSON, this
turns JSON into tables. No re-computation happens here, so a table can never
disagree with the run that produced it.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(cohort):
    p = RESULTS / ("temporal.json" if cohort == "ds007788"
                   else "temporal_%s.json" % cohort)
    return json.loads(p.read_text()) if p.exists() else {}


def table(rows, cohort):
    if not rows:
        print("  (no results yet for %s)" % cohort)
        return
    print("\n%s" % ("=" * 104))
    print("COHORT: %s" % cohort)
    print("=" * 104)
    print("%-16s %6s %7s %9s %9s %9s %9s %8s %8s"
          % ("arm", "win", "acc", "r2_feat", "r2_post", "r2_dec",
             "onset/min", "lat_s", "missed"))
    print("-" * 104)
    for k, r in rows:
        d = r.get("deploy", {})
        print("%-16s %6s %7.3f %+9.3f %+9.3f %+9.3f %9.2f %8.1f %8.2f"
              % (r["arm"], r["win"], r["acc"], r["r2_feat"], r["r2_post"],
                 r.get("r2_dec", float("nan")),
                 d.get("false_onsets_per_min", float("nan")),
                 d.get("latency_s", float("nan")),
                 d.get("missed_onsets", float("nan"))))


def deltas(rows):
    """Per window: every arm against its own `none` baseline.

    The whole question is whether a temporal prior buys accuracy WITHOUT buying
    leakage, so accuracy and r2_post have to be read as a pair against the same
    baseline -- an arm that gains 0.05 accuracy and 0.05 r2_post has bought
    nothing.
    """
    by_win = {}
    for k, r in rows:
        by_win.setdefault(r["win"], {})[r["arm"]] = r
    print("\n%s\nDELTA vs i.i.d. baseline (same window)\n%s" % ("=" * 104, "=" * 104))
    print("%-16s %6s %9s %10s %11s %11s"
          % ("arm", "win", "d_acc", "d_r2_dec", "d_onset/min", "d_lat_s"))
    print("-" * 104)
    for win in sorted(by_win):
        base = by_win[win].get("none")
        if not base:
            continue
        bd = base.get("deploy", {})
        for arm, r in by_win[win].items():
            if arm == "none":
                continue
            d = r.get("deploy", {})
            print("%-16s %6s %+9.3f %+10.3f %+11.2f %+11.1f"
                  % (arm, win, r["acc"] - base["acc"],
                     r.get("r2_dec", np.nan) - base.get("r2_dec", np.nan),
                     d.get("false_onsets_per_min", np.nan)
                     - bd.get("false_onsets_per_min", np.nan),
                     d.get("latency_s", np.nan) - bd.get("latency_s", np.nan)))


def main():
    for cohort in (sys.argv[1:] or ["ds007788", "mobi"]):
        out = load(cohort)
        rows = sorted(out.items(), key=lambda kv: (kv[1]["win"], kv[1]["arm"]))
        table(rows, cohort)
        deltas(rows)


if __name__ == "__main__":
    main()
