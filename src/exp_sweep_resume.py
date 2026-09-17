"""Resume the sweep: run only variants missing from results/sweep100.json.

A CUDA illegal-memory-access poisons the whole process context -- once one
variant trips it, every later variant in the same process fails too. That is
what killed variants 99-131 (including every frequency-band variant) on the
first attempt. So this runner:

  * does only what is still missing, reading and rewriting the same file
  * frees the cached GPU allocator between variants
  * evicts the host-side epoch cache when the frequency band changes
  * exits non-zero on a CUDA fault so a shell loop can restart it with a fresh
    context and carry on from where it stopped

Run under:  until python src/exp_sweep_resume.py; do :; done
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import exp_sweep100 as S
from arch_zoo import n_params

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "sweep100.json"
SEEDS = [1, 2, 3]
TOP_K = 10


def load():
    if OUT.exists():
        try:
            return json.loads(OUT.read_text())
        except Exception:
            pass
    return {"stage1": {}, "stage2": {}}


def save(d):
    OUT.write_text(json.dumps(d, indent=2))


def is_cuda_fault(e):
    s = f"{type(e).__name__}: {e}".lower()
    return "cuda" in s or "illegal memory" in s or "device-side" in s


def band_of(delta):
    return delta.get("_band", (8.0, 30.0))


def main():
    d = load()
    done = {k for k, v in d["stage1"].items() if "mean" in v}
    todo = [(k, v) for k, v in S.VARIANTS.items() if k not in done]
    print(f"stage1: {len(done)}/{len(S.VARIANTS)} done, {len(todo)} to run", flush=True)

    cur_band = None
    for name, delta in todo:
        b = band_of(delta)
        if cur_band is not None and b != cur_band:
            S._CACHE.clear()                       # release the previous band's epochs
            print(f"  (switched band -> {b}, host cache cleared)", flush=True)
        cur_band = b
        t0 = time.time()
        try:
            agg, per_sub = S.run_variant(delta, seed=0)
            agg["secs"] = round(time.time() - t0, 1)
            agg["params"] = n_params({k: v for k, v in delta.items() if k != "_band"})
            d["stage1"][name] = {"config": {k: str(v) for k, v in delta.items()},
                                 "mean": agg, "per_subject": per_sub}
            save(d)
            flag = "LEAK" if agg["r2"] > 0 else "ok"
            print(f"  {name:22s} acc {agg['bal_acc']:.3f} R2 {agg['r2']:+.3f} {flag:5s} "
                  f"AUROC {agg['auroc']:.3f} exec80 {agg['exec80']:.3f} {agg['secs']:5.0f}s",
                  flush=True)
        except Exception as e:
            if is_cuda_fault(e):
                print(f"  {name:22s} CUDA FAULT -- restarting process to reset context",
                      flush=True)
                save(d)
                sys.exit(17)                       # shell loop restarts us
            d["stage1"][name] = {"error": f"{type(e).__name__}: {e}"}
            save(d)
            print(f"  {name:22s} FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------- stage 2: seed replication ----------------
    ok = {k: v for k, v in d["stage1"].items() if "mean" in v}
    ranked = sorted(ok, key=lambda k: -ok[k]["mean"]["bal_acc"])
    admissible = [k for k in ranked if ok[k]["mean"]["r2"] <= 0]
    bands = [k for k in ranked if k.startswith("band_")]
    want = list(dict.fromkeys(ranked[:TOP_K] + admissible[:6] + bands[:6] + ["000_baseline"]))

    for name in want:
        prev = d["stage2"].get(name, {})
        have = prev.get("seeds_done", [])
        need = [s for s in SEEDS if s not in have]
        if not need:
            continue
        accs = prev.get("accs", [ok[name]["mean"]["bal_acc"]])
        r2s = prev.get("r2s", [ok[name]["mean"]["r2"]])
        S._CACHE.clear()
        for sd in need:
            try:
                agg, _ = S.run_variant(S.VARIANTS[name], seed=sd)
                accs.append(agg["bal_acc"]); r2s.append(agg["r2"]); have.append(sd)
            except Exception as e:
                if is_cuda_fault(e):
                    d["stage2"][name] = {"accs": accs, "r2s": r2s, "seeds_done": have,
                                         "acc_mean": float(np.mean(accs)),
                                         "acc_sd": float(np.std(accs)),
                                         "r2_mean": float(np.mean(r2s)),
                                         "seed0_acc": accs[0]}
                    save(d)
                    print(f"  {name} seed{sd}: CUDA FAULT -- restarting", flush=True)
                    sys.exit(17)
                print(f"  {name} seed{sd} failed: {e}", flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        d["stage2"][name] = {"accs": accs, "r2s": r2s, "seeds_done": have,
                             "acc_mean": float(np.mean(accs)),
                             "acc_sd": float(np.std(accs)),
                             "r2_mean": float(np.mean(r2s)), "seed0_acc": accs[0]}
        save(d)
        print(f"  [rep] {name:22s} seed0 {accs[0]:.3f} -> {np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}  shrink {np.mean(accs)-accs[0]:+.3f}  "
              f"R2 {np.mean(r2s):+.3f}  (n={len(accs)})", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
