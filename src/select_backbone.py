"""Pick ONE published backbone for CALM-Net v2.

Not a benchmark. IFNet -- chosen from the literature because filter-bank MI nets
suit ERD/ERS and small samples -- measured 0.551/0.647/0.602 against the
band-power baseline's 0.792/0.771/0.695. The reasoning was fine, the result was
not, so the foundation has to be chosen empirically instead of argued.

Runs each braindecode model through the standard MID harness on three subjects
spanning the confound range (sub-01 high-accuracy/leaky, sub-03 most
movement-coupled, sub-06 the one where IMU is near chance so EEG carries real
signal), and ranks by accuracy with the invariance probe attached.

The winner becomes DEFAULT_BACKBONE in calmnet_v2.py. Everything else is
discarded.

Writes results/backbone_selection.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from arch_zoo import train_variant, predict_variant, encode_variant
from abstain import balanced_accuracy, confidence_auroc
from calibrate import fit_temperature, softmax_np
from calmnet_msa import imu_valid_mask, invariance_r2
from braindecode_zoo import BD_MODELS

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "backbone_selection.json"
SUBJECTS = ["sub-01", "sub-03", "sub-06"]
SKIP = {"ATCNet"}
EPOCHS = 60

# band-power baseline on these three subjects, for reference
BASELINE = {"sub-01": 0.792, "sub-03": 0.771, "sub-06": 0.695}


def subject_data(sub):
    es = build_epochs(subject=sub)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:3])
    valid = imu_valid_mask(es.imu_feats, es.session)
    ti, ci = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    return {"Xf": es.X[tr][ti], "yf": es.y[tr][ti], "Mf": es.imu_feats[tr][ti],
            "vf": valid[tr][ti],
            "Xv": es.X[tr][ci], "yv": es.y[tr][ci], "Mv": es.imu_feats[tr][ci],
            "Xt": es.X[~tr], "yt": es.y[~tr], "Mt": es.imu_feats[~tr], "vt": valid[~tr]}


def is_cuda_fault(e):
    s = f"{type(e).__name__}: {e}".lower()
    return "cuda" in s or "illegal memory" in s


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("loading subjects ...", flush=True)
    D = {s: subject_data(s) for s in SUBJECTS}
    todo = [m for m in BD_MODELS if m not in SKIP and m not in out]
    print(f"{len(todo)} backbones to evaluate on {SUBJECTS}\n", flush=True)

    for name in todo:
        cfg = {"backbone": f"bd:{name}"}
        t0 = time.time()
        try:
            per = {}
            for sub, d in D.items():
                model, (mu, sd) = train_variant({**cfg, "_Mval": d["Mv"]},
                                                d["Xf"], d["yf"], d["Mf"],
                                                d["Xv"], d["yv"], epochs=EPOCHS, seed=0)
                lgv, _ = predict_variant(model, d["Xv"])
                T = float(np.clip(fit_temperature(lgv, d["yv"]), 0.5, 5.0))
                lg, _ = predict_variant(model, d["Xt"])
                p = softmax_np(lg, T)
                pred = p.argmax(1)
                zf = encode_variant(model, d["Xf"]); zt = encode_variant(model, d["Xt"])
                Mf = ((d["Mf"] - mu) / sd).astype(np.float32)
                Mt = ((d["Mt"] - mu) / sd).astype(np.float32)
                r2 = (invariance_r2(zf[d["vf"]], Mf[d["vf"]], zt[d["vt"]], Mt[d["vt"]])
                      if d["vf"].sum() > 20 and d["vt"].sum() > 20 else float("nan"))
                per[sub] = {"acc": balanced_accuracy(d["yt"], pred), "r2": r2,
                            "auroc": confidence_auroc(p.max(1), (pred == d["yt"]).astype(int))}
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            acc = float(np.mean([v["acc"] for v in per.values()]))
            r2m = float(np.nanmean([v["r2"] for v in per.values()]))
            out[name] = {"per_subject": per, "acc": acc, "r2": r2m,
                         "auroc": float(np.mean([v["auroc"] for v in per.values()])),
                         "secs": round(time.time() - t0, 1)}
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  {name:20} acc {acc:.3f}  R2 {r2m:+.3f}  "
                  f"({'  '.join(f"{s}:{per[s]['acc']:.3f}" for s in SUBJECTS)})"
                  f"  {out[name]['secs']:.0f}s", flush=True)
        except Exception as e:
            if is_cuda_fault(e):
                OUT.write_text(json.dumps(out, indent=2))
                print(f"  {name:20} CUDA FAULT -- restart", flush=True)
                sys.exit(17)
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  {name:20} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)

    ok = {k: v for k, v in out.items() if "acc" in v}
    if not ok:
        print("no results"); return
    bl = float(np.mean(list(BASELINE.values())))
    print("\n" + "=" * 74)
    print(f"BACKBONE SELECTION  (band-power baseline on these subjects: {bl:.3f})")
    print("=" * 74)
    print(f"{'backbone':22}{'acc':>8}{'vs base':>9}{'R2':>8}{'AUROC':>8}")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["acc"]):
        print(f"{k:22}{v['acc']:8.3f}{v['acc']-bl:+9.3f}{v['r2']:+8.3f}{v['auroc']:8.3f}")
    best = max(ok, key=lambda k: ok[k]["acc"])
    print(f"\nWINNER: {best}  (acc {ok[best]['acc']:.3f}, R2 {ok[best]['r2']:+.3f})")
    print(f"beats band-power baseline: {ok[best]['acc'] > bl}")
    print(f"\n-> set DEFAULT_BACKBONE = \"{best}\" in calmnet_v2.py")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
