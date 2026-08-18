"""Full CALM-Net pipeline, run longitudinally per subject.

MID (motion-invariant decoder) -> temperature scaling -> split + adaptive conformal
-> selective abstention, evaluated train-early / test-late across sessions.

Reports, per subject and pooled: movement-invariant balanced accuracy, ECE
(raw -> temperature), confidence-correctness AUROC, selective executed accuracy at
80% coverage, static vs adaptive conformal coverage (target 90%), and mean
adaptive-conformal set size.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions, session_days
from splits import grouped_split
from mid import train_mid, predict_mid
from calibrate import (fit_temperature, softmax_np, expected_calibration_error,
                       conformal_qhat, adaptive_conformal)
from abstain import balanced_accuracy, confidence_auroc, selective_risk_at_coverage

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS, ALPHA, COV = 3, 100, 0.1, 0.8


def executed_accuracy_at_coverage(y, probs, coverage=COV):
    """Balanced accuracy over the most-confident `coverage` fraction (rest abstain)."""
    conf = probs.max(1)
    k = max(1, int(round(coverage * len(y))))
    idx = np.argsort(-conf)[:k]
    return balanced_accuracy(y[idx], probs[idx].argmax(1))


def run_subject(sub):
    es = build_epochs(subject=sub)
    days = session_days(sub)
    present = set(int(s) for s in np.unique(es.session))
    sess = [s for s in list_sessions(sub) if s in present]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    tr = es.by_sessions(train)
    ti, ci = grouped_split(tr.segment, tr.y, frac=0.3, seed=0)   # fit / calibration

    # 1) train MID decoder on the fit split
    model = train_mid(tr.X[ti], tr.y[ti], tr.imu_feats[ti], tr.X[ci], tr.y[ci],
                      epochs=EPOCHS, lam_adv=1.0, lam_dec=1.0, seed=0)

    # 2) calibrate: temperature + conformal q on the held-out calibration split
    cal_logits, _ = predict_mid(model, tr.X[ci])
    # clamp temperature to a sane range: an unconstrained fit on well-separated
    # training-session logits can collapse to pathological sharpening (T<<1) that
    # backfires under cross-session drift.
    T = fit_temperature(cal_logits, tr.y[ci])
    T = 1.0 if not np.isfinite(T) else float(np.clip(T, 0.5, 5.0))
    cal_probs = softmax_np(cal_logits, T)
    qhat = conformal_qhat(cal_probs, tr.y[ci], alpha=ALPHA)

    # 3) longitudinal test, in session order
    per_ses, stream_probs, stream_y = {}, [], []
    for s in test:
        te = es.by_sessions([s])
        lg, _ = predict_mid(model, te.X)
        p_raw, p_cal = softmax_np(lg, 1.0), softmax_np(lg, T)
        correct = (p_cal.argmax(1) == te.y).astype(int)
        per_ses[s] = {
            "day": days.get(s, -1),
            "bal_acc": balanced_accuracy(te.y, p_cal.argmax(1)),
            "ece_raw": expected_calibration_error(p_raw, te.y),
            "ece_cal": expected_calibration_error(p_cal, te.y),
            "conf_auroc": confidence_auroc(p_cal.max(1), correct),
            "exec_acc@80": executed_accuracy_at_coverage(te.y, p_cal, COV),
            "sel_risk@80": selective_risk_at_coverage(p_cal.max(1), correct, COV),
        }
        stream_probs.append(p_cal); stream_y.append(te.y)

    # 4) conformal coverage across the whole test stream (static q vs adaptive)
    P = np.concatenate(stream_probs); Y = np.concatenate(stream_y)
    static_cov = float(np.mean((1 - P[np.arange(len(Y)), Y]) <= qhat))
    ada_cov, ada_size, _ = adaptive_conformal(P, Y, q0=qhat, alpha=ALPHA)

    def mean(f): return float(np.nanmean([v[f] for v in per_ses.values()]))
    summary = {
        "temperature": T, "qhat": qhat,
        "bal_acc": mean("bal_acc"), "ece_raw": mean("ece_raw"), "ece_cal": mean("ece_cal"),
        "conf_auroc": mean("conf_auroc"), "exec_acc@80": mean("exec_acc@80"),
        "sel_risk@80": mean("sel_risk@80"),
        "conformal_cov_static": static_cov, "conformal_cov_adaptive": ada_cov,
        "conformal_setsize_adaptive": ada_size, "target_cov": 1 - ALPHA,
    }
    print(f"  [{sub}] invAcc {summary['bal_acc']:.3f} | ECE {summary['ece_raw']:.3f}->"
          f"{summary['ece_cal']:.3f} | AUROC {summary['conf_auroc']:.3f} | exec@80 "
          f"{summary['exec_acc@80']:.3f} | conf.cov static {static_cov:.3f} adapt "
          f"{ada_cov:.3f} (tgt {1-ALPHA:.2f})", flush=True)
    return {"subject": sub, "train": train, "test": test,
            "summary": summary, "sessions": per_ses}


if __name__ == "__main__":
    out = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} ########", flush=True)
        out[sub] = run_subject(sub)
        (RESULTS / "calmnet_full.json").write_text(json.dumps(out, indent=2))

    print("\n============== FULL CALM-Net (mean over test sessions) ==============")
    hdr = f"{'subj':8}{'invAcc':>8}{'ECEraw':>8}{'ECEcal':>8}{'AUROC':>7}{'exec@80':>9}{'cov.stat':>9}{'cov.adpt':>9}"
    print(hdr)
    agg = {}
    for sub, r in out.items():
        s = r["summary"]
        for k in ["bal_acc", "ece_raw", "ece_cal", "conf_auroc", "exec_acc@80",
                  "conformal_cov_static", "conformal_cov_adaptive"]:
            agg.setdefault(k, []).append(s[k])
        print(f"{sub:8}{s['bal_acc']:8.3f}{s['ece_raw']:8.3f}{s['ece_cal']:8.3f}"
              f"{s['conf_auroc']:7.3f}{s['exec_acc@80']:9.3f}"
              f"{s['conformal_cov_static']:9.3f}{s['conformal_cov_adaptive']:9.3f}")
    print("-" * len(hdr))
    print(f"{'MEAN':8}{np.mean(agg['bal_acc']):8.3f}{np.mean(agg['ece_raw']):8.3f}"
          f"{np.mean(agg['ece_cal']):8.3f}{np.mean(agg['conf_auroc']):7.3f}"
          f"{np.mean(agg['exec_acc@80']):9.3f}{np.mean(agg['conformal_cov_static']):9.3f}"
          f"{np.mean(agg['conformal_cov_adaptive']):9.3f}")
    print(f"\ntarget conformal coverage {1-ALPHA:.2f}; exec@80 = balanced acc of the "
          "80% most-confident (committed) windows.")
    (RESULTS / "calmnet_full.json").write_text(json.dumps(out, indent=2))
