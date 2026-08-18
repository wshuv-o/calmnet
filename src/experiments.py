"""CALM-Net experiments: longitudinal decoding + calibration + abstention,
with confound controls. Produces results/*.json and printed summary tables."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions, session_days
from splits import grouped_split, grouped_kfold
from train import train_model, predict, set_seed
from calibrate import (fit_temperature, softmax_np, expected_calibration_error,
                       brier_score, conformal_qhat, conformal_sets)
from abstain import (balanced_accuracy, confidence_auroc, aurc,
                     selective_risk_at_coverage, executed_command_accuracy)

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

MOTOR_CH = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FC3", "FC1", "FCz", "FC2",
            "FC4", "CP3", "CP1", "CPz", "CP2", "CP4"]
FRONTAL_CH = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
              "F1", "Fz", "F2", "F4", "F6", "F8"]


def select_channels(es, names):
    idx = [es.ch_names.index(n) for n in names if n in es.ch_names]
    return es.X[:, idx, :], idx


# --------------------------------------------------------------------------- #
def evaluate_session(y, logits, T, qhat, motion=None, tau=0.9, cov=0.8):
    """All decoding/calibration/abstention metrics for one test session."""
    p_raw = softmax_np(logits, 1.0)
    p_cal = softmax_np(logits, T)
    pred = p_cal.argmax(1)
    conf = p_cal.max(1)
    correct = (pred == y).astype(int)
    sets = conformal_sets(p_cal, qhat)
    exec_acc, coverage = executed_command_accuracy(y, p_cal, conf, tau)
    return {
        "n": int(len(y)),
        "bal_acc": balanced_accuracy(y, pred),
        "acc": float((pred == y).mean()),
        "ece_raw": expected_calibration_error(p_raw, y),
        "ece_cal": expected_calibration_error(p_cal, y),
        "brier_cal": brier_score(p_cal, y),
        "conf_auroc": confidence_auroc(conf, correct),
        "aurc": aurc(conf, correct),
        "sel_risk@%d" % int(cov * 100): selective_risk_at_coverage(conf, correct, cov),
        "conformal_singleton_frac": float((sets.sum(1) == 1).mean()),
        "conformal_set_size": float(sets.sum(1).mean()),
        "exec_acc@tau%.2f" % tau: exec_acc,
        "coverage@tau%.2f" % tau: coverage,
    }


def run_longitudinal(es, model_name, train_sessions, test_sessions,
                     channels=None, epochs=100, seed=0, per_session_recal=True,
                     tag="", days=None):
    """Train on early sessions, evaluate calibration + abstention on later ones."""
    days = days if days is not None else session_days()
    tr = es.by_sessions(train_sessions)
    Xtr_full = tr.X if channels is None else tr.X[:, channels, :]
    n_chan = Xtr_full.shape[1]

    ti, vi = grouped_split(tr.segment, tr.y, frac=0.2, seed=seed)
    Xtr, ytr = Xtr_full[ti], tr.y[ti]
    Xval, yval = Xtr_full[vi], tr.y[vi]

    t0 = time.time()
    model = train_model(model_name, Xtr, ytr, Xval, yval, epochs=epochs,
                        seed=seed, model_kw={"n_chan": n_chan})
    # global calibration on the training-sessions validation split
    vlog, _, _ = predict(model, Xval)
    T = fit_temperature(vlog, yval)
    vcal = softmax_np(vlog, T)
    qhat = conformal_qhat(vcal, yval, alpha=0.1)

    out = {"model": model_name, "tag": tag, "n_chan": n_chan,
           "train_sessions": list(train_sessions), "temperature": T,
           "qhat": qhat, "train_secs": round(time.time() - t0, 1),
           "val_bal_acc": balanced_accuracy(yval, vcal.argmax(1)),
           "sessions": {}}

    for s in test_sessions:
        te = es.by_sessions([s])
        if len(te) == 0 or len(np.unique(te.y)) < 2:
            continue
        Xte = te.X if channels is None else te.X[:, channels, :]
        logits, _, _ = predict(model, Xte)
        m = evaluate_session(te.y, logits, T, qhat, motion=te.motion)
        m["day"] = int(days.get(s, -1))

        if per_session_recal:
            # simulate a short per-session recalibration block: fit T on a
            # grouped 30% slice of THIS session, evaluate on the remaining 70%
            keep, cal = grouped_split(te.segment, te.y, frac=0.3, seed=seed)
            clog, _, _ = predict(model, Xte[cal])
            Ts = fit_temperature(clog, te.y[cal])
            klog, _, _ = predict(model, Xte[keep])
            p_ks = softmax_np(klog, Ts)
            m["ece_recal"] = expected_calibration_error(p_ks, te.y[keep])
        out["sessions"][str(s)] = m
        print(f"  [{model_name}{('/'+tag) if tag else ''}] test ses-{s:02d} "
              f"day{m['day']:>3}: bacc {m['bal_acc']:.3f}  ECE {m['ece_raw']:.3f}"
              f"->{m['ece_cal']:.3f}"
              + (f"->recal {m['ece_recal']:.3f}" if per_session_recal else "")
              + f"  AUROC {m['conf_auroc']:.3f}  selRisk@80 {m['sel_risk@80']:.3f}",
              flush=True)
    return out


def run_within_session(es, model_name, sessions=None, k=4, epochs=80, seed=0):
    """Grouped k-fold CV inside each session -> within-session ceiling."""
    sessions = sessions or list_sessions()
    res = {}
    for s in sessions:
        te = es.by_sessions([s])
        baccs, eces = [], []
        for tri, tei in grouped_kfold(te.segment, te.y, k=k, seed=seed):
            model = train_model(model_name, te.X[tri], te.y[tri],
                                te.X[tei], te.y[tei], epochs=epochs, seed=seed)
            logits, pred, probs = predict(model, te.X[tei])
            baccs.append(balanced_accuracy(te.y[tei], pred))
            eces.append(expected_calibration_error(probs, te.y[tei]))
        res[str(s)] = {"bal_acc": float(np.mean(baccs)), "ece": float(np.mean(eces))}
        print(f"  [{model_name}] within ses-{s:02d}: bacc {np.mean(baccs):.3f} "
              f"ECE {np.mean(eces):.3f}", flush=True)
    return res


def imu_only_baseline(es, train_sessions, test_sessions):
    """Logistic regression on head-IMU motion alone -> how much is movement?"""
    from sklearn.linear_model import LogisticRegression
    tr = es.by_sessions(train_sessions); te_all = es.by_sessions(test_sessions)
    clf = LogisticRegression().fit(tr.motion.reshape(-1, 1), tr.y)
    out = {}
    for s in test_sessions:
        te = es.by_sessions([s])
        pred = clf.predict(te.motion.reshape(-1, 1))
        out[str(s)] = balanced_accuracy(te.y, pred)
    return out


if __name__ == "__main__":
    set_seed(0)
    es = build_epochs()
    all_ses = list_sessions()
    TRAIN = [1, 2, 3]
    TEST = [s for s in all_ses if s not in TRAIN]
    results = {"train_sessions": TRAIN, "test_sessions": TEST,
               "session_days": session_days()}

    print("\n=== IMU-motion-only baseline (confound magnitude) ===")
    results["imu_only"] = imu_only_baseline(es, TRAIN, TEST)
    for s, v in results["imu_only"].items():
        print(f"  ses-{s}: bacc {v:.3f}")

    print("\n=== Longitudinal: EEGNet (all 60 ch) ===")
    results["eegnet_full"] = run_longitudinal(es, "eegnet", TRAIN, TEST, epochs=80)

    print("\n=== Longitudinal: EEG Conformer (all 60 ch) ===")
    results["conformer_full"] = run_longitudinal(es, "conformer", TRAIN, TEST, epochs=80)

    print("\n=== Control: EEGNet, MOTOR channels only ===")
    _, motor_idx = select_channels(es, MOTOR_CH)
    results["eegnet_motor"] = run_longitudinal(es, "eegnet", TRAIN, TEST,
                                               channels=motor_idx, epochs=80,
                                               per_session_recal=False, tag="motor")

    print("\n=== Control: EEGNet, FRONTAL channels only ===")
    _, frontal_idx = select_channels(es, FRONTAL_CH)
    results["eegnet_frontal"] = run_longitudinal(es, "eegnet", TRAIN, TEST,
                                                 channels=frontal_idx, epochs=80,
                                                 per_session_recal=False, tag="frontal")

    (RESULTS / "longitudinal.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {RESULTS/'longitudinal.json'}")
