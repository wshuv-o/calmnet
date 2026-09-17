"""External validation of the movement-leakage result on a second dataset.

Runs the identical analysis used on ds007788 against the MoBI treadmill BCI
cohort (Luu et al. 2018): independent lab, independent subjects, different
movement sensor (goniometers rather than IMUs), same problem class.

Three questions, in order of importance to the claim:

  1. Does the movement-only baseline match or beat EEG decoding?
     (on ds007788: IMU-only 0.870 vs EEG 0.819/0.826)

  2. Does accuracy correlate with movement leakage across architectures?
     (on ds007788: r = +0.67 over 131 variants, +0.79 over the first 16)

  3. Does any architecture beat the baseline while keeping R^2 <= 0?
     (on ds007788: 0 of 131)

If 1 and 2 reproduce, the finding is a property of movement-coupled BCI
paradigms rather than of one dataset -- which is the difference between a
single-dataset observation and a general result.

Splits are across TRIALS (fit T01, test T02+T03): the walking phase is one
contiguous block per recording, so any within-trial split would confound the
label with slow drift.

Writes results/mobi_validation.json.
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

from dataio_mobi import subjects, build_subject
from splits import grouped_split
from arch_zoo import train_variant, predict_variant, encode_variant
from abstain import balanced_accuracy, confidence_auroc
from calibrate import fit_temperature, softmax_np
from calmnet_msa import invariance_r2

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "mobi_validation.json"
EPOCHS = 30

# spread chosen to span the accuracy range seen on ds007788, including the
# published backbones at both ends of the leakage scale
VARIANTS = {
    "000_baseline":     {},
    "wide_F24":         {"F": 24},
    "mha_readout":      {"readout": "mha"},
    "gru_readout":      {"readout": "gru"},
    "mixup":            {"mixup": 0.4},
    "hsic":             {"lam_hsic": 4.0},
    "cosine":           {"cosine": True},
    "short_kernels":    {"kernels": (7, 13, 25)},
    "bd_EEGConformer":  {"backbone": "bd:EEGConformer"},
    "bd_FBMSNet":       {"backbone": "bd:FBMSNet"},
    "bd_ShallowFBCSP":  {"backbone": "bd:ShallowFBCSPNet"},
    "bd_EEGNet":        {"backbone": "bd:EEGNet"},
    "bd_TSception":     {"backbone": "bd:TSception"},
}


def motion_only_baseline(D):
    """Logistic regression on the goniometer vector alone -- how much of the
    label is simply movement? The counterpart of the IMU-only baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(D["Mf"], D["yf"])
    return balanced_accuracy(D["yt"], clf.predict(D["Mt"]))


def subject_data(sub):
    es = build_subject(sub)
    if es is None:
        return None
    fit = es.by_trials([1])
    test = es.by_trials([2, 3])
    if len(np.unique(fit.y)) < 2 or len(np.unique(test.y)) < 2:
        return None
    ti, vi = grouped_split(fit.segment, fit.y, frac=0.3, seed=0)
    return {"Xf": fit.X[ti], "yf": fit.y[ti], "Mf": fit.motion[ti],
            "Xv": fit.X[vi], "yv": fit.y[vi], "Mv": fit.motion[vi],
            "Xt": test.X, "yt": test.y, "Mt": test.motion}


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    subs = subjects()
    print(f"subjects: {subs}\nloading ...", flush=True)
    D = {}
    for s in subs:
        d = subject_data(s)
        if d is not None:
            D[s] = d
            print(f"  {s}: fit {len(d['yf'])} val {len(d['yv'])} test {len(d['yt'])} "
                  f"(walk {int(d['yt'].sum())} stand {int((d['yt']==0).sum())})", flush=True)
    if not D:
        print("no usable subjects"); return

    if "motion_only" not in out:
        mo = {s: motion_only_baseline(d) for s, d in D.items()}
        out["motion_only"] = {"per_subject": mo, "mean": float(np.mean(list(mo.values())))}
        OUT.write_text(json.dumps(out, indent=2))
    print(f"\nMOTION-ONLY baseline (goniometers, no EEG): "
          f"{out['motion_only']['mean']:.3f}", flush=True)
    for s, v in out["motion_only"]["per_subject"].items():
        print(f"    {s}: {v:.3f}", flush=True)
    print()

    for name, delta in VARIANTS.items():
        if name in out and "mean" in out[name]:
            continue
        t0 = time.time()
        try:
            per = {}
            for s, d in D.items():
                model, (mu, sd) = train_variant({**delta, "_Mval": d["Mv"]},
                                                d["Xf"], d["yf"], d["Mf"],
                                                d["Xv"], d["yv"], epochs=EPOCHS, seed=0)
                lgv, _ = predict_variant(model, d["Xv"])
                T = float(np.clip(fit_temperature(lgv, d["yv"]), 0.5, 5.0))
                lg, _ = predict_variant(model, d["Xt"])
                p = softmax_np(lg, T); pred = p.argmax(1)
                zf = encode_variant(model, d["Xf"]); zt = encode_variant(model, d["Xt"])
                Mf = ((d["Mf"] - mu) / sd).astype(np.float32)
                Mt = ((d["Mt"] - mu) / sd).astype(np.float32)
                per[s] = {"acc": balanced_accuracy(d["yt"], pred),
                          "r2": invariance_r2(zf, Mf, zt, Mt),
                          "auroc": confidence_auroc(p.max(1), (pred == d["yt"]).astype(int))}
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            agg = {k: float(np.nanmean([v[k] for v in per.values()]))
                   for k in ("acc", "r2", "auroc")}
            agg["n_leak"] = int(sum(1 for v in per.values() if v["r2"] > 0))
            agg["secs"] = round(time.time() - t0, 1)
            out[name] = {"mean": agg, "per_subject": per}
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  {name:20} acc {agg['acc']:.3f}  R2 {agg['r2']:+.3f}  "
                  f"leak {agg['n_leak']}/{len(D)}  AUROC {agg['auroc']:.3f}  "
                  f"{agg['secs']:.0f}s", flush=True)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  {name:20} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)

    # ---------------- replication verdict ----------------
    ok = {k: v for k, v in out.items() if isinstance(v, dict) and "mean" in v}
    if len(ok) < 3:
        return
    accs = [v["mean"]["acc"] for v in ok.values()]
    r2s = [v["mean"]["r2"] for v in ok.values()]
    n = len(accs); ma, mr = np.mean(accs), np.mean(r2s)
    denom = np.sqrt(np.sum((np.array(accs)-ma)**2) * np.sum((np.array(r2s)-mr)**2))
    r = float(np.sum((np.array(accs)-ma)*(np.array(r2s)-mr)) / denom) if denom else float("nan")
    base = ok["000_baseline"]["mean"]["acc"]
    real = [k for k, v in ok.items() if v["mean"]["acc"] > base and v["mean"]["r2"] <= 0]

    print("\n" + "=" * 78)
    print("EXTERNAL VALIDATION -- MoBI treadmill BCI (Luu et al.), 8 subjects")
    print("=" * 78)
    print(f"{'variant':22}{'acc':>8}{'R2':>9}{'leak':>7}{'AUROC':>8}")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["mean"]["acc"]):
        m = v["mean"]
        print(f"{k:22}{m['acc']:8.3f}{m['r2']:+9.3f}{m['n_leak']:5d}/{len(D)}{m['auroc']:8.3f}")
    print("-" * 78)
    print(f"movement-only baseline      {out['motion_only']['mean']:.3f}")
    print(f"best EEG decoder            {max(accs):.3f}")
    print(f"corr(accuracy, leakage R2)  {r:+.3f}   over {n} architectures")
    print(f"real gains (acc>base, R2<=0){len(real):>4}   {real}")
    print("\nds007788 for comparison:  movement-only 0.870, corr +0.67, real gains 0/131")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
