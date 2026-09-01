"""Run the braindecode model zoo through the CALM-Net invariance harness.

Same protocol as the hand-rolled sweep -- train on sessions 1-3, test on 4-9,
per subject, calibrate on a held-out grouped split -- so the numbers sit
directly beside `results/sweep100.json`. The point is to ask the architecture
question with published, peer-reviewed networks spanning 7k to 3.7M parameters
instead of variations on one small band-power net.

Each model is additionally run with MID off (lam_adv=0, lam_dec=0) so we can see
its raw accuracy and how much of that accuracy the invariance constraint removes.

Writes results/braindecode.json.
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
from braindecode_zoo import BD_MODELS
from arch_zoo import build_variant

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "braindecode.json"
SKIP = {"ATCNet"}                     # final_layer is a container, not a single module


def is_cuda_fault(e):
    s = f"{type(e).__name__}: {e}".lower()
    return "cuda" in s or "illegal memory" in s or "device-side" in s


def load():
    if OUT.exists():
        try:
            return json.loads(OUT.read_text())
        except Exception:
            pass
    return {}


def main():
    out = load()
    todo = []
    for name in BD_MODELS:
        if name in SKIP:
            continue
        for tag, mid in (("MID", True), ("noMID", False)):
            key = f"{name}/{tag}"
            if key not in out or "mean" not in out[key]:
                todo.append((key, name, mid))

    print(f"braindecode zoo: {len(todo)} runs pending "
          f"({len(BD_MODELS) - len(SKIP)} models x MID on/off)\n", flush=True)

    for key, name, mid in todo:
        cfg = {"backbone": f"bd:{name}"}
        if not mid:
            cfg.update({"lam_adv": 0.0, "lam_dec": 0.0})
        t0 = time.time()
        try:
            agg, per_sub = S.run_variant(cfg, seed=0)
            agg["secs"] = round(time.time() - t0, 1)
            try:
                m = build_variant(cfg)
                agg["params"] = sum(p.numel() for p in m.parameters())
                del m
            except Exception:
                agg["params"] = 0
            out[key] = {"model": name, "mid": mid, "mean": agg, "per_subject": per_sub}
            OUT.write_text(json.dumps(out, indent=2))
            flag = "LEAK" if agg["r2"] > 0 else "ok"
            print(f"  {key:28s} acc {agg['bal_acc']:.3f} R2 {agg['r2']:+.3f} {flag:5s} "
                  f"AUROC {agg['auroc']:.3f} exec80 {agg['exec80']:.3f} "
                  f"{agg['params']:>9,}p {agg['secs']:5.0f}s", flush=True)
        except Exception as e:
            if is_cuda_fault(e):
                OUT.write_text(json.dumps(out, indent=2))
                print(f"  {key:28s} CUDA FAULT -- restarting for a fresh context",
                      flush=True)
                sys.exit(17)
            out[key] = {"model": name, "mid": mid, "error": f"{type(e).__name__}: {e}"}
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  {key:28s} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------- summary ----------------
    ok = {k: v for k, v in out.items() if "mean" in v}
    if not ok:
        print("\nno results"); return
    print("\n\n" + "=" * 96)
    print("BRAINDECODE ZOO -- mean over 7 subjects, longitudinal")
    print("=" * 96)
    hdr = (f"{'model':20}{'MID':>6}{'acc':>8}{'R2':>8}{'leak':>6}{'AUROC':>7}"
           f"{'exec80':>8}{'ECE':>7}{'params':>11}")
    print(hdr); print("-" * len(hdr))
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["mean"]["bal_acc"]):
        m = v["mean"]
        print(f"{v['model']:20}{'on' if v['mid'] else 'off':>6}{m['bal_acc']:8.3f}"
              f"{m['r2']:+8.3f}{m['n_leak']:5d}/7{m['auroc']:7.3f}{m['exec80']:8.3f}"
              f"{m['ece']:7.3f}{m['params']:11,}")

    mids = {v["model"]: v["mean"] for v in ok.values() if v["mid"]}
    nomids = {v["model"]: v["mean"] for v in ok.values() if not v["mid"]}
    both = sorted(set(mids) & set(nomids))
    if both:
        print("\n--- what the invariance constraint costs each architecture ---")
        print(f"{'model':20}{'noMID acc':>11}{'MID acc':>10}{'drop':>8}"
              f"{'noMID R2':>10}{'MID R2':>9}")
        for n in both:
            a, b = nomids[n], mids[n]
            print(f"{n:20}{a['bal_acc']:11.3f}{b['bal_acc']:10.3f}"
                  f"{b['bal_acc']-a['bal_acc']:+8.3f}{a['r2']:+10.3f}{b['r2']:+9.3f}")

    adm = [(v["model"], v["mean"]) for v in ok.values() if v["mid"] and v["mean"]["r2"] <= 0]
    print(f"\n--- ADMISSIBLE (MID on, R2 <= 0): {len(adm)} ---")
    for n, m in sorted(adm, key=lambda kv: -kv[1]["bal_acc"]):
        print(f"  {n:20} acc {m['bal_acc']:.3f}  R2 {m['r2']:+.3f}  "
              f"AUROC {m['auroc']:.3f}  {m['params']:,}p")
    if not adm:
        print("  (none)")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
