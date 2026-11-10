"""Inference cost at cohort A's input shape: parameters, FLOPs, latency, memory.

    python -m analysis.cost          # writes results/bench_cost.json
"""
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from model import CALMNet, TangentBranch, count_parameters

OUT = Path(__file__).resolve().parent.parent / "results" / "bench_cost.json"
C, T = 60, 400          # 60 channels, 4 s at 100 Hz
K = 8                   # windows per forward
REPEATS, WARMUP = 200, 20


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
    return ts


@torch.no_grad()
def latency(model, device, batch):
    m = model.to(device)
    x = torch.randn(batch * K, C, T, device=device)
    ts = _time(lambda: m(x), device)
    return statistics.median(ts), float(np.percentile(ts, 95))


@torch.no_grad()
def flops(model):
    from torch.utils.flop_counter import FlopCounterMode
    ctr = FlopCounterMode(display=False)
    with ctr:
        model.to("cpu")(torch.randn(K, C, T))
    return int(ctr.get_total_flops())      # excludes the eigendecompositions


@torch.no_grad()
def reference_parts(device, windows=32):
    """Reference update alone vs the whole branch (the reported numbers use 32 windows)."""
    br = TangentBranch(C, 128).eval().to(device)
    x = torch.randn(windows, C, T, device=device)

    def update():
        m = br._shrunk_cov(x.double()).mean(0)
        br.run_cov.mul_(1 - br.momentum).add_(m.to(br.run_cov.dtype), alpha=br.momentum)

    return {"update_ms": round(statistics.median(_time(update, device)), 3),
            "branch_forward_ms": round(statistics.median(_time(lambda: br(x), device)), 3)}


@torch.no_grad()
def peak_memory(model):
    m = model.to("cuda")
    x = torch.randn(K, C, T, device="cuda")
    m(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    out = {"input": {"channels": C, "samples": T, "windows_per_forward": K},
           "device_cuda": torch.cuda.get_device_name(0) if "cuda" in devices else None,
           "arms": {}}
    for name, tan in (("stem", False), ("stem+branch", True)):
        m = CALMNet(C, use_tangent=tan).eval()
        rec = {"parameters": count_parameters(m), "flops_per_forward": flops(m)}
        for dev in devices:
            for batch in (1, 32):
                med, p95 = latency(m, dev, batch)
                rec["%s_b%d_ms" % (dev, batch)] = round(med, 3)
                rec["%s_b%d_p95_ms" % (dev, batch)] = round(p95, 3)
        if "cuda" in devices:
            rec["cuda_peak_mb_b1"] = round(peak_memory(m), 2)
        out["arms"][name] = rec
        print("%-12s %7d params  cpu %.2f ms" % (name, rec["parameters"], rec["cpu_b1_ms"]))
    for dev in devices:
        out["branch_parts_%s" % dev] = reference_parts(dev)
    s, b = out["arms"]["stem"], out["arms"]["stem+branch"]
    out["overhead"] = {"parameters_x": round(b["parameters"] / s["parameters"], 2),
                       "cpu_b1_x": round(b["cpu_b1_ms"] / s["cpu_b1_ms"], 2)}
    if "cuda" in devices:
        out["overhead"]["cuda_b1_x"] = round(b["cuda_b1_ms"] / s["cuda_b1_ms"], 2)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
