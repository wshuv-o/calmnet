"""100-variant architecture search with seed replication.

Two stages, because a 100-way single-seed leaderboard is mostly selection on
noise:
  Stage 1  train every variant once (seed 0) on all 7 subjects
  Stage 2  re-run the top-K and a random control set across extra seeds, and
           report how much of the stage-1 advantage survives

Scoring is unchanged: a variant only counts if balanced accuracy rises while the
intent->IMU probe stays <= 0. Everything else is leakage.

Writes results/sweep100.json incrementally.
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
from arch_zoo import train_variant, predict_variant, encode_variant, n_params
from calibrate import (fit_temperature, softmax_np, expected_calibration_error,
                       conformal_qhat, adaptive_conformal)
from abstain import balanced_accuracy, confidence_auroc
from calmnet_msa import imu_valid_mask, invariance_r2

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS, ALPHA, COV = 3, 80, 0.1, 0.8
TOP_K, REPLICATE_SEEDS = 10, [1, 2, 3]


# --------------------------------------------------------------------------- #
# Variant space
# --------------------------------------------------------------------------- #
def build_space():
    V = {}

    def add(name, **cfg):
        V[name] = cfg

    add("000_baseline")

    # --- backbones -------------------------------------------------------- #
    for bk in ("shallow", "deep", "tcn", "eegnet"):
        add(f"bb_{bk}", backbone=bk)
    add("bb_shallow_wide", backbone="shallow", F=80)
    add("bb_deep_2blk", backbone="deep", blocks=2)
    add("bb_deep_4blk", backbone="deep", blocks=4)
    add("bb_tcn_6lvl", backbone="tcn", levels=6)
    add("bb_tcn_wide", backbone="tcn", F=64)

    # --- readouts --------------------------------------------------------- #
    for ro in ("mean", "mha", "gru", "stat", "max", "edge"):
        add(f"ro_{ro}", readout=ro)
    add("ro_gru_wide", readout="gru", F=16)
    add("ro_stat_wide", readout="stat", F=16)

    # --- capacity --------------------------------------------------------- #
    for f in (4, 6, 12, 16, 24, 32):
        add(f"cap_F{f}", F=f)
    for d in (1, 3, 4):
        add(f"cap_D{d}", D=d)
    add("cap_F16_D4", F=16, D=4)
    add("cap_F32_D1", F=32, D=1)

    # --- filter banks ----------------------------------------------------- #
    banks = {"k_short": (7, 13, 25), "k_long": (25, 51, 101), "k_five": (7, 13, 25, 51, 75),
             "k_seven": (5, 9, 13, 19, 25, 37, 51), "k_single25": (25,),
             "k_single51": (51,), "k_dense": (9, 13, 17, 21, 25),
             "k_wide_span": (5, 25, 101)}
    for n, k in banks.items():
        add(n, kernels=k)

    # --- pooling ---------------------------------------------------------- #
    for p, st in ((10, 2), (15, 3), (35, 7), (50, 10), (75, 15), (100, 20)):
        add(f"pool_{p}_{st}", pool=p, stride=st)

    # --- MID geometry ----------------------------------------------------- #
    for f in (0.20, 0.30, 0.40, 0.60, 0.70, 0.80):
        add(f"int_{int(f*100)}", int_frac=f)

    # --- norm / head / dropout -------------------------------------------- #
    add("norm_group", norm="gn")
    add("head_mlp", head="mlp")
    for p in (0.15, 0.30, 0.65, 0.80):
        add(f"drop_{int(p*100)}", p_drop=p)
    add("residual", residual=True)
    add("se", se=True)

    # --- augmentation ----------------------------------------------------- #
    for aug in ("chan_drop", "time_mask", "noise", "shift", "scale"):
        for s in (0.05, 0.15, 0.30):
            add(f"aug_{aug}_{int(s*100)}", augment=aug, aug_strength=s)
    for m in (0.2, 0.4, 0.8):
        add(f"mixup_{int(m*10)}", mixup=m)
    add("ls_10", label_smooth=0.1)
    add("ls_20", label_smooth=0.2)

    # --- independence penalties ------------------------------------------- #
    for pen in ("mmd", "ortho", "coral"):
        for lam in (1.0, 4.0, 16.0):
            add(f"pen_{pen}_{int(lam)}", penalty=pen, lam_pen=lam)
    for lam in (1.0, 4.0, 16.0):
        add(f"pen_hsic_{int(lam)}", lam_hsic=lam)
    for lam in (0.0, 4.0, 16.0):
        add(f"decorr_{int(lam)}", lam_dec=lam)
    for lam in (0.0, 0.5, 2.0, 4.0):
        add(f"adv_{int(lam*10)}", lam_adv=lam)

    # --- optimisation ----------------------------------------------------- #
    for o in ("sgd", "adam", "rmsprop"):
        add(f"opt_{o}", optim=o)
    for lr in (3e-4, 3e-3, 1e-2):
        add(f"lr_{lr:g}", lr=lr)
    for wd in (0.0, 1e-2, 1e-1):
        add(f"wd_{wd:g}", wd=wd)
    for b in (32, 128, 256):
        add(f"batch_{b}", batch=b)
    add("cosine", cosine=True)
    add("cosine_long", cosine=True, lr=3e-3)

    # --- selection -------------------------------------------------------- #
    add("sel_disent", disent_select=True)
    add("sel_disent_soft", disent_select=True, lam_sel=0.3)
    add("sel_disent_hard", disent_select=True, lam_sel=3.0)

    # --- promising combinations ------------------------------------------- #
    add("cmb_honest1", cosine=True, lam_hsic=4.0, int_frac=0.3)
    add("cmb_honest2", cosine=True, penalty="ortho", lam_pen=4.0)
    add("cmb_honest3", kernels=(7, 13, 25), cosine=True, disent_select=True)
    add("cmb_honest4", augment="chan_drop", aug_strength=0.15, cosine=True, lam_hsic=4.0)
    add("cmb_cap1", F=24, readout="gru", cosine=True)
    add("cmb_cap2", F=16, se=True, readout="stat")

    # --- input representation: frequency band ------------------------------ #
    # Movement artefact is broadband and low-frequency; genuine sensorimotor
    # rhythm lives in mu/beta. A narrower band should trade accuracy for
    # invariance -- the one axis that could move the honest number.
    for tag, band in (("mu", (8.0, 13.0)), ("beta", (13.0, 30.0)),
                      ("broad", (4.0, 40.0)), ("full", (1.0, 45.0))):
        add(f"band_{tag}", _band=band)
        add(f"band_{tag}_cos", _band=band, cosine=True)
        add(f"band_{tag}_hsic", _band=band, lam_hsic=4.0)
    return V


VARIANTS = build_space()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_CACHE = {}


def subject_data(sub, band=(8.0, 30.0)):
    key = (sub, band)
    if key in _CACHE:
        return _CACHE[key]
    es = build_epochs(subject=sub, l_freq=band[0], h_freq=band[1])
    present = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in present]
    train = sess[:N_TRAIN]
    valid = imu_valid_mask(es.imu_feats, es.session)
    tr = np.isin(es.session, train)
    ti, vi = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    d = {"Xf": es.X[tr][ti], "yf": es.y[tr][ti], "Mf": es.imu_feats[tr][ti],
         "vf": valid[tr][ti],
         "Xv": es.X[tr][vi], "yv": es.y[tr][vi], "Mv": es.imu_feats[tr][vi],
         "Xt": es.X[~tr], "yt": es.y[~tr], "Mt": es.imu_feats[~tr], "vt": valid[~tr]}
    _CACHE[key] = d
    return d


def exec_at_cov(y, probs, cov=COV):
    conf = probs.max(1)
    k = max(1, int(round(cov * len(y))))
    idx = np.argsort(-conf)[:k]
    return balanced_accuracy(y[idx], probs[idx].argmax(1))


def run_variant(delta, seed=0, band=(8.0, 30.0)):
    delta = dict(delta)
    band = delta.pop("_band", band)
    per_sub = []
    for sub in SUBJECTS:
        d = subject_data(sub, band)
        cfg = {**delta, "_Mval": d["Mv"]}
        model, (mu, sd) = train_variant(cfg, d["Xf"], d["yf"], d["Mf"],
                                        d["Xv"], d["yv"], epochs=EPOCHS, seed=seed)
        lgv, _ = predict_variant(model, d["Xv"])
        T = fit_temperature(lgv, d["yv"])
        T = 1.0 if not np.isfinite(T) else float(np.clip(T, 0.5, 5.0))
        lg, _ = predict_variant(model, d["Xt"])
        p_cal = softmax_np(lg, T)
        pred = p_cal.argmax(1)
        correct = (pred == d["yt"]).astype(int)
        qhat = conformal_qhat(softmax_np(lgv, T), d["yv"], alpha=ALPHA)
        ada_cov, _, _ = adaptive_conformal(p_cal, d["yt"], q0=qhat, alpha=ALPHA)
        zf = encode_variant(model, d["Xf"]); zt = encode_variant(model, d["Xt"])
        Mf = ((d["Mf"] - mu) / sd).astype(np.float32)
        Mt = ((d["Mt"] - mu) / sd).astype(np.float32)
        r2 = (invariance_r2(zf[d["vf"]], Mf[d["vf"]], zt[d["vt"]], Mt[d["vt"]])
              if d["vf"].sum() > 20 and d["vt"].sum() > 20 else float("nan"))
        per_sub.append({"subject": sub, "bal_acc": balanced_accuracy(d["yt"], pred),
                        "ece": expected_calibration_error(p_cal, d["yt"]),
                        "auroc": confidence_auroc(p_cal.max(1), correct),
                        "exec80": exec_at_cov(d["yt"], p_cal), "r2": r2,
                        "cov_adaptive": ada_cov})
    agg = {k: float(np.nanmean([p[k] for p in per_sub]))
           for k in ("bal_acc", "ece", "auroc", "exec80", "r2", "cov_adaptive")}
    agg["n_leak"] = int(sum(1 for p in per_sub if p["r2"] > 0))
    return agg, per_sub


def main():
    out_path = RESULTS / "sweep100.json"
    out = {"stage1": {}, "stage2": {}}
    print(f"Variant space: {len(VARIANTS)} configs\n", flush=True)

    print("Pre-loading subjects (8-30 Hz) ...", flush=True)
    for s in SUBJECTS:
        subject_data(s)
    print("done\n", flush=True)

    # ---------------- stage 1 ----------------
    for i, (name, delta) in enumerate(VARIANTS.items(), 1):
        t0 = time.time()
        try:
            agg, per_sub = run_variant(delta, seed=0)
            agg["secs"] = round(time.time() - t0, 1)
            agg["params"] = n_params({k: v for k, v in delta.items() if k != "_band"})
            out["stage1"][name] = {"config": {k: str(v) for k, v in delta.items()},
                                   "mean": agg, "per_subject": per_sub}
            flag = "LEAK" if agg["r2"] > 0 else "ok"
            print(f"[{i:3d}/{len(VARIANTS)}] {name:22s} acc {agg['bal_acc']:.3f} "
                  f"R2 {agg['r2']:+.3f} {flag:5s} AUROC {agg['auroc']:.3f} "
                  f"exec80 {agg['exec80']:.3f} {agg['secs']:5.0f}s", flush=True)
        except Exception as e:
            out["stage1"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{i:3d}/{len(VARIANTS)}] {name:22s} FAILED {type(e).__name__}: {e}",
                  flush=True)
        out_path.write_text(json.dumps(out, indent=2))

    # ---------------- stage 2: seed replication ----------------
    ok = {k: v for k, v in out["stage1"].items() if "mean" in v}
    ranked = sorted(ok, key=lambda k: -ok[k]["mean"]["bal_acc"])
    admissible = [k for k in ranked if ok[k]["mean"]["r2"] <= 0]
    rng = np.random.default_rng(0)
    control = list(rng.choice([k for k in ranked if k not in ranked[:TOP_K]],
                              size=min(5, max(0, len(ranked) - TOP_K)), replace=False))
    to_rep = list(dict.fromkeys(ranked[:TOP_K] + admissible[:5] + control + ["000_baseline"]))
    print(f"\n\nStage 2: replicating {len(to_rep)} variants over seeds {REPLICATE_SEEDS}\n",
          flush=True)
    for name in to_rep:
        accs = [ok[name]["mean"]["bal_acc"]]
        r2s = [ok[name]["mean"]["r2"]]
        for sd in REPLICATE_SEEDS:
            try:
                agg, _ = run_variant(VARIANTS[name], seed=sd)
                accs.append(agg["bal_acc"]); r2s.append(agg["r2"])
            except Exception as e:
                print(f"    {name} seed{sd} failed: {e}", flush=True)
        out["stage2"][name] = {"accs": accs, "r2s": r2s,
                               "acc_mean": float(np.mean(accs)),
                               "acc_sd": float(np.std(accs)),
                               "r2_mean": float(np.mean(r2s)),
                               "seed0_acc": accs[0]}
        print(f"  {name:22s} seed0 {accs[0]:.3f} -> {np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}  (shrink {np.mean(accs)-accs[0]:+.3f})  "
              f"R2 {np.mean(r2s):+.3f}", flush=True)
        out_path.write_text(json.dumps(out, indent=2))

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
