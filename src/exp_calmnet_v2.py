"""CALM-Net v2 -- the model the paper describes, finally run end to end.

`calmnet_v2.py` has sat in the tree since the architecture sweep and nothing
ever imported it: no driver, no results file, no number. It is the only place
where the selective head, the cost-sensitive objective and the Mondrian
conformal bound that the paper claims actually exist in code. This runs it.

Per subject, on the standard split (train sessions 1-3, test the rest):

  * balanced accuracy of the intent decoder
  * BOTH movement probes -- the old cross-split `invariance_r2`, so the number
    is comparable with the ablation and the 131-sweep, and the corrected
    session-grouped `invariance_r2_cv`, which is the one to believe
  * the full three-term SAS abstention rule, evaluated whole for the first
    time: selective head AND conformal singleton AND the wrong-walk bound

Three protocol choices, each a direct consequence of a mistake recorded in
`results/SESSION_NOTES.md`. They are not incidental; they are the point.

  1. MULTI-SEED. Seed sd is 0.028, and the sweep's single-seed winner (0.782)
     replicated to 0.712 +/- 0.057. Every number here is mean +/- sd over
     `--seeds` runs. A single-seed number out of this file would repeat the
     exact error the file exists to avoid.

  2. CALIBRATION IS NOT THE EARLY-STOPPING SET. Conformal coverage is only a
     guarantee on data that had no hand in fitting the model. The 30% held out
     of the training sessions is split again, by segment, into an
     early-stopping half and a calibration half. Nothing else in this project
     did that, because nothing else in this project made a coverage claim.

  3. THE CORRECTED PROBE IS PRIMARY. `invariance_r2` fits on the training
     sessions and scores on the test sessions, so a representation that merely
     drifts scores negative and looks invariant. Both are reported; the
     cross-split one is there for comparability, not for belief.

Reference on the identical split: the classical covariance -> EA -> tangent ->
logistic pipeline, so the deep model's accuracy is read against the thing that
beat it rather than against chance. This is a same-split reference, NOT a
reproduction of the 0.776 headline, which used its own preprocessing grid.

Usage:
    python src/exp_calmnet_v2.py                    # 7 subjects, 3 seeds
    python src/exp_calmnet_v2.py --seeds 1 --subjects sub-01
    python src/exp_calmnet_v2.py --cohort mobi
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

from dataio import build_epochs, list_sessions
from splits import grouped_split
from calmnet_v2 import (train_calmnet_v2, predict_v2, encode_v2, mondrian_qhat,
                        mondrian_sets, calibrate_walk_threshold, safety_report,
                        DEFAULT_BACKBONE, WALK)
from calmnet_msa import invariance_r2, imu_valid_mask
from calibrate import fit_temperature, softmax_np
from abstain import balanced_accuracy
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "calmnet_v2.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3
EPOCHS = 80
TARGET_COV = 0.8
ALPHA = 0.1              # conformal miscoverage
MAX_WRONG_WALK = 0.05    # bound on P(commit walk | true stop)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def subject_data(sub):
    """Same split as exp_ablate / select_backbone, with the held-out 30% split
    once more into an early-stopping half and a conformal-calibration half."""
    es = build_epochs(subject=sub)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    valid = imu_valid_mask(es.imu_feats, es.session)

    seg_tr, y_tr = es.segment[tr], es.y[tr]
    ti, hi = grouped_split(seg_tr, y_tr, frac=0.3, seed=0)
    # split the held-out 30% again, by segment, into early-stop / calibration
    e_rel, c_rel = grouped_split(seg_tr[hi], y_tr[hi], frac=0.5, seed=1)
    ei, ci = hi[e_rel], hi[c_rel]

    X_tr, M_tr, v_tr = es.X[tr], es.imu_feats[tr], valid[tr]
    return {
        "Xf": X_tr[ti], "yf": y_tr[ti], "Mf": M_tr[ti], "vf": v_tr[ti],
        "Xe": X_tr[ei], "ye": y_tr[ei],                       # early stopping
        "Xc": X_tr[ci], "yc": y_tr[ci],                       # conformal cal
        "Xt": es.X[~tr], "yt": es.y[~tr], "Mt": es.imu_feats[~tr],
        "vt": valid[~tr], "st": es.session[~tr],
    }


def mobi_data(sub):
    """Second cohort, same interface. Fit on trial T01, test on T02+T03."""
    from dataio_mobi import build_subject as mobi_build
    es = mobi_build(sub)
    if es is None:
        return None
    fit, test = es.by_trials([1]), es.by_trials([2, 3])
    if len(np.unique(fit.y)) < 2 or len(np.unique(test.y)) < 2:
        return None
    ti, hi = grouped_split(fit.segment, fit.y, frac=0.3, seed=0)
    e_rel, c_rel = grouped_split(fit.segment[hi], fit.y[hi], frac=0.5, seed=1)
    ei, ci = hi[e_rel], hi[c_rel]
    return {
        "Xf": fit.X[ti], "yf": fit.y[ti], "Mf": fit.imu_feats[ti],
        "vf": np.ones(len(ti), bool),
        "Xe": fit.X[ei], "ye": fit.y[ei],
        "Xc": fit.X[ci], "yc": fit.y[ci],
        "Xt": test.X, "yt": test.y, "Mt": test.imu_feats,
        "vt": np.ones(len(test.y), bool), "st": test.session,
    }


# --------------------------------------------------------------------------- #
# Classical reference on the identical split
# --------------------------------------------------------------------------- #
def tangent_ea_reference(d):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    Ff = FE.tangent(FE.euclidean_align(FE.covariances(d["Xf"])))
    Ft = FE.tangent(FE.euclidean_align(FE.covariances(d["Xt"])))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=0.1,
                                           class_weight="balanced"))
    clf.fit(Ff, d["yf"])
    pred = clf.predict(Ft)
    vt = d["vt"]
    r2cv = (FE.invariance_r2_cv(Ft[vt], d["Mt"][vt], d["st"][vt])
            if vt.sum() > 30 else float("nan"))
    return {"acc": balanced_accuracy(d["yt"], pred), "r2_cv": r2cv,
            "n_features": int(Ft.shape[1])}


# --------------------------------------------------------------------------- #
# Which of the three gate terms actually binds
# --------------------------------------------------------------------------- #
def gate_breakdown(probs, g, theta, q, tau_walk):
    """The paper writes a three-term abstention rule and the repo never ran it
    with more than one term. Whether it is a real rule or two dead terms and a
    threshold depends entirely on which of them rejects anything, so report the
    acceptance rate of each term alone and of the conjunction."""
    sel = g >= theta
    single = mondrian_sets(probs, q).sum(1) == 1
    pred = probs.argmax(1)
    safe = ~((pred == WALK) & (probs[:, WALK] < tau_walk))
    return {"accept_selective": float(sel.mean()),
            "accept_singleton": float(single.mean()),
            "accept_walkbound": float(safe.mean()),
            "accept_all": float((sel & single & safe).mean()),
            # marginal cost of each term given the other two
            "lost_to_selective": float((single & safe & ~sel).mean()),
            "lost_to_singleton": float((sel & safe & ~single).mean()),
            "lost_to_walkbound": float((sel & single & ~safe).mean())}


# --------------------------------------------------------------------------- #
# One subject, one seed
# --------------------------------------------------------------------------- #
def run_subject(d, seed, epochs=EPOCHS, backbone=DEFAULT_BACKBONE, verbose=False):
    model, (mu, sd) = train_calmnet_v2(
        d["Xf"], d["yf"], d["Mf"], d["Xe"], d["ye"],
        backbone=backbone, epochs=epochs, target_cov=TARGET_COV,
        seed=seed, verbose=verbose)

    # temperature on the early-stopping half (already used for selection)
    lg_e, _, _ = predict_v2(model, d["Xe"])
    T = float(np.clip(fit_temperature(lg_e, d["ye"]), 0.5, 5.0))

    # conformal + thresholds on the calibration half (untouched by fitting)
    lg_c, _, g_c = predict_v2(model, d["Xc"])
    p_c = softmax_np(lg_c, T)
    q = mondrian_qhat(p_c, d["yc"], alpha=ALPHA)
    tau = calibrate_walk_threshold(p_c, d["yc"], max_wrong_walk=MAX_WRONG_WALK)
    theta = float(np.quantile(g_c, 1.0 - TARGET_COV))

    lg_t, _, g_t = predict_v2(model, d["Xt"])
    p_t = softmax_np(lg_t, T)
    if not np.isfinite(p_t).all():
        p_t = np.full_like(p_t, 0.5)
    pred = p_t.argmax(1)

    zf, zt = encode_v2(model, d["Xf"]), encode_v2(model, d["Xt"])
    vf, vt = d["vf"], d["vt"]
    Mf_z = ((d["Mf"] - mu) / sd).astype(np.float32)
    Mt_z = ((d["Mt"] - mu) / sd).astype(np.float32)
    r2_old = (invariance_r2(zf[vf], Mf_z[vf], zt[vt], Mt_z[vt])
              if vf.sum() > 20 and vt.sum() > 20 else float("nan"))
    r2_cv = (FE.invariance_r2_cv(zt[vt], d["Mt"][vt], d["st"][vt])
             if vt.sum() > 30 else float("nan"))

    out = {"acc": balanced_accuracy(d["yt"], pred),
           "r2_cv": r2_cv, "r2_crosssplit": r2_old,
           "temperature": T, "theta": theta, "tau_walk": tau,
           "n_cal": int(len(d["yc"])),
           "n_params": int(sum(p.numel() for p in model.parameters()))}
    out.update(safety_report(d["yt"], p_t, g_t, theta, q, tau))
    out.update(gate_breakdown(p_t, g_t, theta, q, tau))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def agg(vals):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if not len(v):
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--cohort", choices=["ds007788", "mobi"], default="ds007788")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    out_path = Path(a.out) if a.out else (
        OUT if a.cohort == "ds007788" else RESULTS / "calmnet_v2_mobi.json")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CALM-Net v2 | backbone {a.backbone} | cohort {a.cohort} | "
          f"{a.seeds} seed(s) | device {dev}", flush=True)

    if a.cohort == "mobi":
        from dataio_mobi import subjects as mobi_subjects
        subs = a.subjects or mobi_subjects()
        loader = mobi_data
    else:
        subs = a.subjects or SUBJECTS
        loader = subject_data

    print("loading ...", flush=True)
    D = {}
    for s in subs:
        try:
            d = loader(s)
        except Exception as e:
            print(f"  {s}: load failed -- {type(e).__name__}: {e}", flush=True)
            continue
        if d is None:
            print(f"  {s}: skipped", flush=True)
            continue
        D[s] = d
        n_walk = int((d["yc"] == 1).sum())
        thin = min(n_walk, len(d["yc"]) - n_walk) < 19
        print(f"  {s}: fit {len(d['yf'])}  es {len(d['ye'])}  "
              f"cal {len(d['yc'])} (walk {n_walk})  test {len(d['yt'])}"
              f"{'   <-- thin calibration set' if thin else ''}", flush=True)
    if not D:
        print("no subjects loaded")
        return

    res = {"config": {"backbone": a.backbone, "cohort": a.cohort,
                      "seeds": a.seeds, "epochs": a.epochs,
                      "target_cov": TARGET_COV, "alpha": ALPHA,
                      "max_wrong_walk": MAX_WRONG_WALK,
                      "n_train_sessions": N_TRAIN},
           "reference_tangent_ea": {}, "per_subject": {}}

    print("\nclassical reference (covariance -> EA -> tangent -> LR)", flush=True)
    for sub, d in D.items():
        try:
            r = tangent_ea_reference(d)
        except Exception as e:
            print(f"  {sub}: failed -- {type(e).__name__}: {e}", flush=True)
            continue
        res["reference_tangent_ea"][sub] = r
        print(f"  {sub}  acc {r['acc']:.3f}  R2_cv {r['r2_cv']:+.3f}", flush=True)

    print(f"\nCALM-Net v2, {a.seeds} seed(s) per subject", flush=True)
    for sub, d in D.items():
        runs = []
        for seed in range(a.seeds):
            t0 = time.time()
            try:
                r = run_subject(d, seed, epochs=a.epochs, backbone=a.backbone,
                                verbose=a.verbose)
            except Exception as e:
                print(f"  {sub} seed{seed}: FAILED -- {type(e).__name__}: {e}",
                      flush=True)
                continue
            r["secs"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"  {sub} seed{seed}  acc {r['acc']:.3f}  "
                  f"R2_cv {r['r2_cv']:+.3f} (x-split {r['r2_crosssplit']:+.3f})  "
                  f"cov {r['coverage']:.2f}  exec {r['executed_bal_acc']:.3f}  "
                  f"wrongwalk {r['wrong_walk_rate']:.3f}  {r['secs']:.0f}s",
                  flush=True)
        if not runs:
            continue
        keys = ("acc", "r2_cv", "r2_crosssplit", "coverage", "executed_bal_acc",
                "wrong_walk_rate", "walk_recall", "mean_set_size",
                "conformal_cov_stop", "conformal_cov_walk",
                "accept_selective", "accept_singleton", "accept_walkbound",
                "lost_to_selective", "lost_to_singleton", "lost_to_walkbound")
        res["per_subject"][sub] = {"runs": runs,
                                   **{k: agg([r[k] for r in runs]) for k in keys}}
        out_path.write_text(json.dumps(res, indent=2))

    # ---- cohort summary -------------------------------------------------- #
    P = res["per_subject"]
    if P:
        res["summary"] = {k: agg([P[s][k]["mean"] for s in P]) for k in
                          ("acc", "r2_cv", "r2_crosssplit", "coverage",
                           "executed_bal_acc", "wrong_walk_rate", "walk_recall",
                           "accept_selective", "accept_singleton",
                           "accept_walkbound", "lost_to_selective",
                           "lost_to_singleton", "lost_to_walkbound")}
        res["summary"]["mean_seed_sd_acc"] = float(
            np.mean([P[s]["acc"]["sd"] for s in P]))
        R = res["reference_tangent_ea"]
        if R:
            res["summary"]["reference_acc"] = float(
                np.mean([R[s]["acc"] for s in R]))
            res["summary"]["reference_r2_cv"] = float(
                np.nanmean([R[s]["r2_cv"] for s in R]))
        out_path.write_text(json.dumps(res, indent=2))

        S = res["summary"]
        print(f"\n{'=' * 66}")
        print(f"CALM-Net v2 ({a.backbone}), {len(P)} subjects, {a.seeds} seeds")
        print(f"  balanced accuracy   {S['acc']['mean']:.3f} +/- {S['acc']['sd']:.3f}"
              f"   (mean within-subject seed sd {S['mean_seed_sd_acc']:.3f})")
        print(f"  invariance R2_cv    {S['r2_cv']['mean']:+.3f}   "
              f"[cross-split probe {S['r2_crosssplit']['mean']:+.3f}]")
        print(f"  SAS coverage        {S['coverage']['mean']:.3f}   "
              f"executed acc {S['executed_bal_acc']['mean']:.3f}")
        print(f"  wrong-walk rate     {S['wrong_walk_rate']['mean']:.3f}   "
              f"(bound {MAX_WRONG_WALK})   walk recall {S['walk_recall']['mean']:.3f}")
        print(f"  gate, term alone    selective {S['accept_selective']['mean']:.2f}"
              f"  singleton {S['accept_singleton']['mean']:.2f}"
              f"  walk-bound {S['accept_walkbound']['mean']:.2f}")
        print(f"  gate, marginal cost selective {S['lost_to_selective']['mean']:.2f}"
              f"  singleton {S['lost_to_singleton']['mean']:.2f}"
              f"  walk-bound {S['lost_to_walkbound']['mean']:.2f}")
        if R:
            print(f"  tangent_EA ref      {S['reference_acc']:.3f}   "
                  f"R2_cv {S['reference_r2_cv']:+.3f}")
        print("=" * 66)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
