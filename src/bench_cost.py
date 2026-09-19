"""What the branch costs at inference, measured rather than argued.

The paper claims a decoder for online exoskeleton control, and the branch takes
the model from 19,505 parameters to 290,943. A reviewer is entitled to ask
whether the thing can run in the loop at all, and a parameter count does not
answer that: the branch's cost is dominated by an eigendecomposition of a
60x60 covariance, which no parameter count reflects.

Measured here, for the stem alone and for stem plus branch:

  parameters      exact, from the built module
  FLOPs           per window, torch.utils.flop_counter
  latency         per window at batch 1 (the online case) and batch 32,
                  after warm-up, median and 95th percentile over many repeats
  reference       the label-free covariance update on its own, since it runs
                  at inference and is the part with no analogue in the stem
  memory          peak CUDA allocation for one forward

Nothing is retrained. The numbers describe the model the paper reports, at
cohort A's input shape.

    python src/bench_cost.py

Writes results/bench_cost.json.
"""
from __future__ import annotations

import io
import json
import os
import statistics
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from driftnet import build_driftnet                       # noqa: E402

RES = os.path.join(HERE, "..", "results")
C, T = 60, 400          # cohort A: 60 channels, 4 s at 100 Hz
K = 8                   # windows per forward, as the harness uses
REPEATS = 200
WARMUP = 20


def build(tangent):
    return build_driftnet(C, T, 2, use_align=False, use_ctx=False,
                          use_gate=False, use_tangent=tangent).eval()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def latency(model, device, batch, repeats=REPEATS):
    """Median and p95 wall-clock for one forward, in milliseconds."""
    m = model.to(device)
    x = torch.randn(batch, K, C, T, device=device)
    for _ in range(WARMUP):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return (statistics.median(ts),
            float(np.percentile(ts, 95)),
            statistics.median(ts) / batch)


@torch.no_grad()
def flops(model, batch=1):
    from torch.utils.flop_counter import FlopCounterMode
    m = model.to("cpu")
    x = torch.randn(batch, K, C, T)
    ctr = FlopCounterMode(display=False)
    with ctr:
        m(x)
    return int(ctr.get_total_flops())


@torch.no_grad()
def _time(fn, device, repeats=REPEATS):
    for _ in range(WARMUP):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


@torch.no_grad()
def reference_parts(device, batch=32):
    """Split the branch into the update and the map it feeds.

    There is no update_reference method: the update happens inside forward, so
    timing forward would report the whole branch and call it the update. These
    time the two halves separately. The update is the covariance, the shrinkage
    and the exponential moving average, which is the part that has no analogue
    in the stem and still runs when no labels are available. The map is the two
    eigendecompositions that follow it.
    """
    from driftnet import TangentBranch
    br = TangentBranch(C, 128).eval().to(device)
    x = torch.randn(batch, C, T, device=device)

    def update():
        c = br._shrunk_cov(x.detach().double())
        m = c.mean(0)
        br.run_cov.mul_(1 - br.momentum).add_(m.to(br.run_cov.dtype),
                                              alpha=br.momentum)

    return {"update_ms": round(_time(update, device), 3),
            "branch_forward_ms": round(_time(lambda: br(x), device), 3)}


@torch.no_grad()
def peak_memory(model, batch=1):
    if not torch.cuda.is_available():
        return None
    m = model.to("cuda")
    x = torch.randn(batch, K, C, T, device="cuda")
    m(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    out = {"_note": ("Inference cost of the reported model at cohort A's input "
                     "shape (%d channels, %d samples, %d windows per forward). "
                     "No training. Latency is median and 95th percentile over "
                     "%d forwards after %d warm-up, wall clock, with CUDA "
                     "synchronised. FLOPs from torch.utils.flop_counter, which "
                     "does not count the eigendecomposition in the tangent "
                     "map, so the branch's FLOP figure understates it and the "
                     "latency is the honest measure."
                     % (C, T, K, REPEATS, WARMUP)),
           "input": {"channels": C, "samples": T, "windows_per_forward": K},
           "device_cuda": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None),
           "arms": {}}

    for name, tan in (("stem", False), ("stem+branch", True)):
        m = build(tan)
        rec = {"parameters": n_params(m), "flops_per_forward": flops(m)}
        for dev in devices:
            for batch in (1, 32):
                med, p95, per_win = latency(m, dev, batch)
                rec["%s_b%d_ms" % (dev, batch)] = round(med, 3)
                rec["%s_b%d_p95_ms" % (dev, batch)] = round(p95, 3)
                rec["%s_b%d_ms_per_window" % (dev, batch)] = round(per_win, 4)
        pm = peak_memory(m)
        if pm is not None:
            rec["cuda_peak_mb_b1"] = round(pm, 2)
        out["arms"][name] = rec
        print("  %-12s %7d params  %s" % (name, rec["parameters"],
                                          " ".join(
            "%s %.2fms" % (d, rec["%s_b1_ms" % d]) for d in devices)))

    for dev in devices:
        parts = reference_parts(dev)
        out["branch_parts_%s" % dev] = parts
        print("  branch on %-4s: reference update %.3f ms, whole branch "
              "%.3f ms (32 windows)"
              % (dev, parts["update_ms"], parts["branch_forward_ms"]))

    s, b = out["arms"]["stem"], out["arms"]["stem+branch"]
    out["overhead"] = {
        "parameters_x": round(b["parameters"] / s["parameters"], 2),
        "cpu_b1_x": round(b["cpu_b1_ms"] / s["cpu_b1_ms"], 2),
    }
    if "cuda" in devices:
        out["overhead"]["cuda_b1_x"] = round(b["cuda_b1_ms"] / s["cuda_b1_ms"],
                                             2)
    io.open(os.path.join(RES, "bench_cost.json"), "w",
            encoding="utf-8").write(json.dumps(out, indent=1))
    print("\n  overhead: %.2fx parameters, %.2fx CPU latency at batch 1"
          % (out["overhead"]["parameters_x"], out["overhead"]["cpu_b1_x"]))
    print("  wrote results/bench_cost.json")


if __name__ == "__main__":
    main()
