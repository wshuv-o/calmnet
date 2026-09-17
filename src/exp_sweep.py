"""Architecture sweep: train N variants per subject, score each on accuracy AND
movement-invariance, and report where each one actually excels.

Scoring rule: on this dataset accuracy alone is not a valid objective, because
the walk/stop label is partly a movement label. A variant is only credited with
an improvement if balanced accuracy rises while intent->IMU R^2 stays <= 0.
Variants that gain accuracy with positive R^2 are reported as LEAK.

Writes results/arch_sweep.json. Does not modify any existing results file.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from arch_zoo import (train_variant, predict_variant, encode_variant, n_params)
from calibrate import (fit_temperature, softmax_np, expected_calibration_error,
                       conformal_qhat, adaptive_conformal)
from abstain import balanced_accuracy, confidence_auroc
from calmnet_msa import imu_valid_mask, invariance_r2

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS, ALPHA, COV = 3, 80, 0.1, 0.8

# --------------------------------------------------------------------------- #
# The zoo. Each entry is a delta against the baseline config.
# --------------------------------------------------------------------------- #
VARIANTS = {
    "00_baseline":       {},
    "01_meanpool":       {"readout": "mean"},
    "02_mha_readout":    {"readout": "mha"},
    "03_squeeze_excite": {"se": True},
    "04_wide_F16":       {"F": 16},
    "05_wide_F24":       {"F": 24},
    "06_bank5":          {"kernels": (7, 13, 25, 51, 75)},
    "07_long_kernels":   {"kernels": (25, 51, 101)},
    "08_short_kernels":  {"kernels": (7, 13, 25)},
    "09_spatial_D4":     {"D": 4},
    "10_groupnorm":      {"norm": "gn"},
    "11_mlp_head":       {"head": "mlp"},
    "12_intent_25pct":   {"int_frac": 0.25},
    "13_intent_75pct":   {"int_frac": 0.75},
    "14_fine_pool":      {"pool": 15, "stride": 3},
    "15_coarse_pool":    {"pool": 50, "stride": 10},
    "16_hsic":           {"lam_hsic": 4.0},
    "17_strong_decorr":  {"lam_dec": 4.0},
    "18_label_smooth":   {"label_smooth": 0.1},
    "19_mixup":          {"mixup": 0.4},
    "20_cosine_lr":      {"cosine": True},
    "21_low_dropout":    {"p_drop": 0.25},
    "22_disent_select":  {"disent_select": True},
    "23_se_mha_wide":    {"se": True, "readout": "mha", "F": 16},
}


def subject_data(sub):
    es = build_epochs(subject=sub)
    present = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in present]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    valid = imu_valid_mask(es.imu_feats, es.session)
    tr = np.isin(es.session, train)
    ti, vi = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    return {
        "Xf": es.X[tr][ti], "yf": es.y[tr][ti], "Mf": es.imu_feats[tr][ti],
        "vf": valid[tr][ti],
        "Xv": es.X[tr][vi], "yv": es.y[tr][vi], "Mv": es.imu_feats[tr][vi],
        "Xt": es.X[~tr], "yt": es.y[~tr], "Mt": es.imu_feats[~tr],
        "vt": valid[~tr],
    }


def exec_at_cov(y, probs, cov=COV):
    conf = probs.max(1)
    k = max(1, int(round(cov * len(y))))
    idx = np.argsort(-conf)[:k]
    return balanced_accuracy(y[idx], probs[idx].argmax(1))


def run_variant(name, delta, D):
    cfg = dict(delta)
    cfg["_Mval"] = D["Mv"]
    per_sub = []
    for sub, d in D["subjects"].items():
        model, (mu, sd) = train_variant(
            {**cfg, "_Mval": d["Mv"]}, d["Xf"], d["yf"], d["Mf"], d["Xv"], d["yv"],
            epochs=EPOCHS, seed=0)
        lg, _ = predict_variant(model, d["Xt"])
        T = fit_temperature(lg, d["yt"])
        T = 1.0 if not np.isfinite(T) else float(np.clip(T, 0.5, 5.0))
        # calibrate on the held-out val split, not on test
        lgv, _ = predict_variant(model, d["Xv"])
        Tv = fit_temperature(lgv, d["yv"])
        Tv = 1.0 if not np.isfinite(Tv) else float(np.clip(Tv, 0.5, 5.0))
        p_cal = softmax_np(lg, Tv)
        pred = p_cal.argmax(1)
        correct = (pred == d["yt"]).astype(int)
        qhat = conformal_qhat(softmax_np(lgv, Tv), d["yv"], alpha=ALPHA)
        ada_cov, _, _ = adaptive_conformal(p_cal, d["yt"], q0=qhat, alpha=ALPHA)

        zf = encode_variant(model, d["Xf"]); zt = encode_variant(model, d["Xt"])
        Mf = ((d["Mf"] - mu) / sd).astype(np.float32)
        Mt = ((d["Mt"] - mu) / sd).astype(np.float32)
        r2 = (invariance_r2(zf[d["vf"]], Mf[d["vf"]], zt[d["vt"]], Mt[d["vt"]])
              if d["vf"].sum() > 20 and d["vt"].sum() > 20 else float("nan"))

        per_sub.append({
            "subject": sub, "bal_acc": balanced_accuracy(d["yt"], pred),
            "ece": expected_calibration_error(p_cal, d["yt"]),
            "auroc": confidence_auroc(p_cal.max(1), correct),
            "exec80": exec_at_cov(d["yt"], p_cal), "r2": r2,
            "cov_adaptive": ada_cov,
        })
    agg = {k: float(np.nanmean([p[k] for p in per_sub]))
           for k in ("bal_acc", "ece", "auroc", "exec80", "r2", "cov_adaptive")}
    agg["n_leak"] = int(sum(1 for p in per_sub if p["r2"] > 0))
    agg["params"] = n_params(cfg)
    return agg, per_sub


def main():
    print("Loading subjects ...", flush=True)
    subjects = {}
    for s in SUBJECTS:
        subjects[s] = subject_data(s)
    D = {"subjects": subjects, "Mv": subjects[SUBJECTS[0]]["Mv"]}
    print(f"loaded {len(subjects)} subjects\n", flush=True)

    out = {}
    for i, (name, delta) in enumerate(VARIANTS.items(), 1):
        t0 = time.time()
        try:
            agg, per_sub = run_variant(name, delta, D)
            agg["secs"] = round(time.time() - t0, 1)
            out[name] = {"config": {k: str(v) for k, v in delta.items()},
                         "mean": agg, "per_subject": per_sub}
            print(f"[{i:2d}/{len(VARIANTS)}] {name:20s} acc {agg['bal_acc']:.3f} "
                  f"R2 {agg['r2']:+.3f} leak {agg['n_leak']}/7  AUROC {agg['auroc']:.3f}  "
                  f"exec80 {agg['exec80']:.3f}  {agg['params']:>7,}p  {agg['secs']:.0f}s",
                  flush=True)
        except Exception as e:
            print(f"[{i:2d}/{len(VARIANTS)}] {name:20s} FAILED: {type(e).__name__}: {e}",
                  flush=True)
            out[name] = {"error": f"{type(e).__name__}: {e}"}
        (RESULTS / "arch_sweep.json").write_text(json.dumps(out, indent=2))

    # ---------------- summary ----------------
    ok = {k: v for k, v in out.items() if "mean" in v}
    base = ok.get("00_baseline", {}).get("mean", {})
    print("\n\n" + "=" * 104)
    print("ARCHITECTURE SWEEP -- mean over 7 subjects, longitudinal (train ses 1-3, test 4-9)")
    print("=" * 104)
    hdr = (f"{'variant':22}{'acc':>7}{'dAcc':>7}{'R2':>8}{'leak':>6}{'AUROC':>7}"
           f"{'exec80':>8}{'ECE':>7}{'cov':>7}{'params':>9}{'verdict':>12}")
    print(hdr); print("-" * len(hdr))
    rows = sorted(ok.items(), key=lambda kv: -kv[1]["mean"]["bal_acc"])
    for name, v in rows:
        m = v["mean"]
        d = m["bal_acc"] - base.get("bal_acc", m["bal_acc"])
        if name == "00_baseline":
            verdict = "reference"
        elif m["r2"] > 0:
            verdict = "LEAK"
        elif d > 0.005:
            verdict = "REAL GAIN"
        elif d < -0.005:
            verdict = "worse"
        else:
            verdict = "neutral"
        print(f"{name:22}{m['bal_acc']:7.3f}{d:+7.3f}{m['r2']:+8.3f}{m['n_leak']:5d}/7"
              f"{m['auroc']:7.3f}{m['exec80']:8.3f}{m['ece']:7.3f}{m['cov_adaptive']:7.3f}"
              f"{m['params']:9,}{verdict:>12}")

    adm = [(k, v["mean"]) for k, v in ok.items() if v["mean"]["r2"] <= 0]
    print("\n--- ADMISSIBLE ONLY (intent->IMU R2 <= 0), ranked by accuracy ---")
    for k, m in sorted(adm, key=lambda kv: -kv[1]["bal_acc"])[:10]:
        print(f"  {k:22} acc {m['bal_acc']:.3f}  R2 {m['r2']:+.3f}  "
              f"AUROC {m['auroc']:.3f}  exec80 {m['exec80']:.3f}")
    if not adm:
        print("  (none)")

    print("\n--- BEST BY METRIC ---")
    for metric, better in (("bal_acc", max), ("auroc", max), ("exec80", max),
                           ("ece", min), ("cov_adaptive", None)):
        if metric == "cov_adaptive":
            k = min(ok, key=lambda n: abs(ok[n]["mean"]["cov_adaptive"] - 0.9))
            print(f"  {'coverage closest to 0.90':28} {k:22} "
                  f"{ok[k]['mean']['cov_adaptive']:.3f}")
        else:
            k = better(ok, key=lambda n: ok[n]["mean"][metric])
            print(f"  {metric:28} {k:22} {ok[k]['mean'][metric]:.3f}"
                  f"   (R2 {ok[k]['mean']['r2']:+.3f})")
    print(f"\nSaved -> {RESULTS / 'arch_sweep.json'}")


if __name__ == "__main__":
    main()
