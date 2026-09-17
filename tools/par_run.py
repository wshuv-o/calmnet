"""Run exp_calmnetx (arm, seed) cells concurrently instead of one at a time.

Why
---
The models are tiny (24,181 parameters at batch 32). A forward and backward
pass is a few microseconds of GPU work against roughly 100 us of Python and
kernel-launch overhead, so the card sits at 3-4 % utilisation and 22 W of 360 W
while 30 of 32 CPU cores idle. The work is too small to fill the GPU at any
batch size; the only thing that helps is running several cells at once.

How
---
Each (arm, seed) cell is launched as its own process with its own CX_OUT file,
then the shards are merged into the target JSON. The inner computation is
untouched, so a cell's numbers are identical to what the serial run would have
produced. That matters: the paper's noise floor (0.003 for dn_noctx, 0.037 for
dn_noalign on cohort B) was measured serially, and a change in execution that
shifted results would invalidate it.

exp_calmnetx skips any key already present in CX_OUT, so a merged output also
acts as a resume point.

Usage
-----
    python tools/par_run.py --out b_floor_m0.02.json --workers 4 \\
        --arms dn_noctx,dn_noalign --seeds 0,1,2 \\
        --env CX_COHORT=mobi DN_MOMENTUM=0.02

Verify determinism against an existing serial file:

    python tools/par_run.py --verify b_floor_m0.10.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SRC = ROOT / "src"


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}


def merge_into(target, shards):
    out = load(target)
    for s in shards:
        out.update(load(s))
    Path(target).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def run_grid(out_name, arms, seeds, env_extra, workers, epochs, size):
    target = RESULTS / out_name
    have = set(load(target))
    cells = [(a, s) for a in arms for s in seeds
             if "%s|s%s" % (a, s) not in have]
    if not cells:
        print("nothing to do: all %d cells already in %s" %
              (len(arms) * len(seeds), out_name))
        return
    print("%d cells to run, %d workers, %d already present"
          % (len(cells), workers, len(have)))

    shards, procs, t0 = [], [], time.time()
    for i, (arm, seed) in enumerate(cells):
        shard = RESULTS / ("_shard_%s_%s_s%s.json" % (out_name[:-5], arm, seed))
        shards.append(shard)
        env = dict(os.environ)
        # Cap intra-op threads per worker. PyTorch defaults to ~24 threads per
        # process, so N workers spawn 24N threads on a fixed core count and
        # spend their time context-switching: measured 3 workers x 80 threads
        # on 32 cores at 54 % total CPU, each worker 2.6x slower than alone,
        # for a net throughput gain of ~10 %. One share of the cores each
        # removes the contention.
        share = max(1, (os.cpu_count() or 8) // max(workers, 1))
        env.update({"OMP_NUM_THREADS": str(share),
                    "MKL_NUM_THREADS": str(share),
                    "TORCH_NUM_THREADS": str(share)})
        env.update({"CX_ARMS": arm, "CX_SEEDS": str(seed),
                    "CX_OUT": shard.name, "CX_EPOCHS": str(epochs),
                    "CX_SIZE": size})
        env.update(env_extra)
        log = RESULTS / ("_shard_%s_%s_s%s.log" % (out_name[:-5], arm, seed))
        procs.append((arm, seed, subprocess.Popen(
            [sys.executable, "-u", str(SRC / "exp_calmnetx.py")],
            cwd=str(ROOT), env=env,
            stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
        # stagger starts so eight processes do not all build epochs at once
        time.sleep(4)
        while sum(1 for _, _, p, _ in procs if p.poll() is None) >= workers:
            time.sleep(5)

    for arm, seed, p, log in procs:
        rc = p.wait()
        print("  %-12s seed=%s  rc=%d  (%s)" % (arm, seed, rc, log.name),
              flush=True)

    merged = merge_into(target, shards)
    print("merged %d cells into %s (%.1f min)"
          % (len(merged), out_name, (time.time() - t0) / 60))
    for s in shards:
        try:
            s.unlink()
        except OSError:
            pass


def verify(name):
    """Re-run one cell of an existing file in a shard and compare."""
    target = RESULTS / name
    ref = load(target)
    if not ref:
        print("no such file: %s" % name)
        return
    key = sorted(ref)[0]
    arm, seed = key.split("|s")
    print("re-running %s from %s to check the parallel path matches" %
          (key, name))
    shard = RESULTS / ("_verify_%s.json" % key.replace("|", "_"))
    env = dict(os.environ)
    env.update({"CX_ARMS": arm, "CX_SEEDS": seed, "CX_OUT": shard.name})
    subprocess.run([sys.executable, "-u", str(SRC / "exp_calmnetx.py")],
                   cwd=str(ROOT), env=env, check=False)
    got = load(shard).get(key, {})
    a, b = ref[key].get("acc"), got.get("acc")
    print("  serial   acc = %r" % a)
    print("  parallel acc = %r" % b)
    print("  identical: %s" % (a == b))
    try:
        shard.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--arms", default="dn_noctx,dn_noalign")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", default="base")
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--verify")
    a = ap.parse_args()
    if a.verify:
        verify(a.verify)
    else:
        extra = dict(kv.split("=", 1) for kv in a.env)
        run_grid(a.out, a.arms.split(","), a.seeds.split(","), extra,
                 a.workers, a.epochs, a.size)
