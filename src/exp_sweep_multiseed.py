"""The 131-variant sweep, re-run properly: every cell on every seed, scored
with the corrected probe.

This is simultaneously step 2 of the Next list (re-scoring with
`invariance_r2_cv`) and the honest version of "run all the deep learning
configs and find a winner". Both need the same compute, so they are one job.

WHY THE ORIGINAL SWEEP CANNOT BE RE-SCORED WITHOUT RE-TRAINING
--------------------------------------------------------------
`SESSION_NOTES.md` says re-scoring is "a re-measurement, not a re-training,"
because the models are already trained. They are not: nothing in this project
ever wrote a checkpoint. `exp_ablate.py` keeps a best-state dict in memory and
drops it at the end of the call; the sweep never saves at all. The corrected
probe needs the encoder's outputs, so every model has to be retrained. The
saving grace is cost -- the whole 131-cell stage 1 was 1.2 h of recorded
compute, median 30 s per cell.

WHY MULTI-SEED IS NOT OPTIONAL HERE
-----------------------------------
Between-cell sd across the 131 recorded cells is 0.037. Within-cell seed sd is
0.028 (sweep replication) to 0.071 (CALM-Net v2, section 8). The spread of the
architecture search is the same size as its noise. Under that ratio a
single-seed leaderboard of 131 cells has an expected maximum roughly 2.9 sd
above the mean from chance alone, which is how 0.782 became 0.712 +/- 0.057.

So a "win" is only recorded here when it survives the noise:

    admissible      mean r2_cv <= 0                (corrected probe, not the
                                                    cross-split one)
    beats reference mean_acc - 2*sem > REFERENCE   (sem over seeds)

The script also reports how many cells would have been declared winners by
single-seed maximum, so the gap between the two counts is visible rather than
argued about.

REFERENCE is tangent_EA at 0.763 -- the value reproduced in this session and
matching `features.json` (`raw|tangent_ea` = 0.7628). Not 0.776, which appears
in the notes but not in the results file.

Usage:
    python src/exp_sweep_multiseed.py                 # all cells, seeds 0,1,2
    python src/exp_sweep_multiseed.py --seeds 5
    python src/exp_sweep_multiseed.py --cells bb_tcn decorr_0
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from exp_sweep100 import build_space, run_variant

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "sweep_multiseed.json"
REFERENCE = 0.763          # tangent_EA, reproduced this session
REF_NAME = "tangent_EA"


def summarise(runs):
    """runs: list of per-seed agg dicts."""
    a = np.array([r["bal_acc"] for r in runs], float)
    rc = np.array([r["r2_cv"] for r in runs], float)
    rx = np.array([r["r2"] for r in runs], float)
    n = len(a)
    sem = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n_seeds": n,
            "acc_mean": float(a.mean()), "acc_sd": float(a.std(ddof=1)) if n > 1 else 0.0,
            "acc_sem": sem, "acc_max": float(a.max()), "acc_min": float(a.min()),
            "acc_lower2sem": float(a.mean() - 2 * sem) if n > 1 else float("nan"),
            "r2_cv_mean": float(np.nanmean(rc)),
            "r2_crosssplit_mean": float(np.nanmean(rx)),
            "seed_accs": a.tolist()}


def verdict(s):
    if not np.isfinite(s["acc_lower2sem"]):
        return "insufficient-seeds"
    if s["r2_cv_mean"] > 0:
        return "leaky"                       # inadmissible whatever the accuracy
    if s["acc_lower2sem"] > REFERENCE:
        return "WIN"
    if s["acc_mean"] > REFERENCE:
        return "within-noise"                # looks better, cannot be claimed
    return "loses"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = Path(a.out) if a.out else OUT

    space = build_space()
    names = a.cells or list(space)
    seeds = list(range(a.seeds))

    store = json.loads(out_path.read_text()) if out_path.exists() else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"multi-seed sweep | {len(names)} cells x {len(seeds)} seeds | device {dev}")
    print(f"reference: {REF_NAME} = {REFERENCE:.3f} | admissible = r2_cv <= 0")
    print(f"win = acc_mean - 2*sem > {REFERENCE:.3f}\n", flush=True)

    t_start = time.time()
    for i, name in enumerate(names, 1):
        cell = store.setdefault(name, {"runs": {}})
        for sd in seeds:
            if str(sd) in cell["runs"]:
                continue
            t0 = time.time()
            try:
                agg, _ = run_variant(space[name], seed=sd)
            except Exception as e:
                print(f"  {name} seed{sd}: FAILED -- {type(e).__name__}: {e}",
                      flush=True)
                continue
            agg["secs"] = round(time.time() - t0, 1)
            cell["runs"][str(sd)] = agg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        runs = list(cell["runs"].values())
        if not runs:
            continue
        cell["summary"] = summarise(runs)
        cell["verdict"] = verdict(cell["summary"])
        out_path.write_text(json.dumps(store, indent=2))
        s = cell["summary"]
        el = time.time() - t_start
        eta = el / i * (len(names) - i)
        print(f"[{i:3d}/{len(names)}] {name:22s} acc {s['acc_mean']:.3f}"
              f" +/-{s['acc_sd']:.3f} (max {s['acc_max']:.3f})"
              f"  r2_cv {s['r2_cv_mean']:+.3f}  {cell['verdict']:14s}"
              f"  eta {eta/60:.0f}m", flush=True)

    # ---------------- report ---------------------------------------------- #
    done = {k: v for k, v in store.items() if "summary" in v}
    if not done:
        print("nothing completed")
        return
    S = {k: v["summary"] for k, v in done.items()}
    accs = np.array([S[k]["acc_mean"] for k in S])
    maxes = np.array([S[k]["acc_max"] for k in S])
    sds = np.array([S[k]["acc_sd"] for k in S])

    wins = [k for k in done if done[k]["verdict"] == "WIN"]
    noise = [k for k in done if done[k]["verdict"] == "within-noise"]
    leaky = [k for k in done if done[k]["verdict"] == "leaky"]
    adm = [k for k in done if S[k]["r2_cv_mean"] <= 0]
    fake = [k for k in done if S[k]["acc_max"] > REFERENCE]

    print(f"\n{'=' * 74}")
    print(f"{len(done)} cells x {a.seeds} seeds against {REF_NAME} = {REFERENCE:.3f}")
    print(f"  between-cell sd of mean acc   {accs.std():.3f}")
    print(f"  mean within-cell seed sd      {np.nanmean(sds):.3f}")
    print(f"  ratio (signal:noise)          {accs.std() / max(np.nanmean(sds), 1e-9):.2f}")
    print(f"\n  admissible (r2_cv <= 0)       {len(adm):3d} / {len(done)}")
    print(f"  leaky (r2_cv > 0)             {len(leaky):3d} / {len(done)}")
    print(f"\n  cells with ANY seed > ref     {len(fake):3d}   <- single-seed 'wins'")
    print(f"  cells beating ref on mean     {len(noise) + len(wins):3d}")
    print(f"  cells beating ref past noise  {len(wins):3d}   <- real wins")
    if wins:
        print("\n  WINNERS:")
        for k in sorted(wins, key=lambda k: -S[k]["acc_mean"]):
            print(f"    {k:22s} {S[k]['acc_mean']:.3f} +/- {S[k]['acc_sd']:.3f}"
                  f"  (>{S[k]['acc_lower2sem']:.3f} at 2 sem)  r2_cv {S[k]['r2_cv_mean']:+.3f}")
    else:
        print(f"\n  No cell beats {REF_NAME} past the noise floor while admissible.")
    if noise:
        print(f"\n  Beat it on the mean but NOT past noise ({len(noise)}):")
        for k in sorted(noise, key=lambda k: -S[k]["acc_mean"])[:8]:
            print(f"    {k:22s} {S[k]['acc_mean']:.3f} +/- {S[k]['acc_sd']:.3f}"
                  f"  lower bound {S[k]['acc_lower2sem']:.3f}  r2_cv {S[k]['r2_cv_mean']:+.3f}")

    print("\n  top 8 admissible cells by mean accuracy:")
    for k in sorted(adm, key=lambda k: -S[k]["acc_mean"])[:8]:
        print(f"    {k:22s} {S[k]['acc_mean']:.3f} +/- {S[k]['acc_sd']:.3f}"
              f"  r2_cv {S[k]['r2_cv_mean']:+.3f}"
              f"  [x-split {S[k]['r2_crosssplit_mean']:+.3f}]")

    # how much the old probe disagrees with the corrected one
    dr = np.array([S[k]["r2_crosssplit_mean"] - S[k]["r2_cv_mean"] for k in S])
    flip = sum(1 for k in S
               if (S[k]["r2_crosssplit_mean"] <= 0) != (S[k]["r2_cv_mean"] <= 0))
    print(f"\n  probe disagreement: cross-split minus corrected = "
          f"{dr.mean():+.3f} (sd {dr.std():.3f})")
    print(f"  admissibility verdict flips between probes: {flip} / {len(S)}")
    print("=" * 74)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
