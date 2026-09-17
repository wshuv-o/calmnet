"""Evaluate the multi-subject double-adversarial CALM-Net against the stored
per-subject baseline (results/calmnet_full.json).

Falsifiable target: sub-02 sits at 0.516 (chance) in the per-subject baseline
while results/extra_invariance.json shows it reaching 0.743 with more data at
R^2 -0.08, i.e. sample-limited rather than ceiling-limited. Pooling should move
it -- and the invariance probe must stay negative, otherwise the gain is
movement leakage and the result is void.

Writes results/msa.json. Does not modify any existing results file.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from calmnet_msa import (SUBJECTS, load_pooled, train_msa, predict_msa,
                         encode_msa, invariance_r2, subject_leakage)
from splits import grouped_split
from calibrate import (fit_temperature, softmax_np, expected_calibration_error,
                       conformal_qhat, adaptive_conformal)
from abstain import balanced_accuracy, confidence_auroc

RESULTS = Path(__file__).resolve().parent.parent / "results"
ALPHA, COV, EPOCHS = 0.1, 0.8, 60


def executed_acc_at_coverage(y, probs, coverage=COV):
    conf = probs.max(1)
    k = max(1, int(round(coverage * len(y))))
    idx = np.argsort(-conf)[:k]
    return balanced_accuracy(y[idx], probs[idx].argmax(1))


def main(seed=0):
    print("Loading pooled data ...", flush=True)
    data = load_pooled()
    X, y, S, M, V = data["X"], data["y"], data["subj"], data["M"], data["valid"]
    print(f"\npooled train {X.shape}  |  {len(data['subjects'])} subjects  "
          f"|  IMU-invalid {int((~V).sum())}/{len(V)}", flush=True)

    # segment-grouped fit/calibration split, leakage-free, stratified by class
    fit_idx, cal_idx = grouped_split(data["segment"], y, frac=0.3, seed=seed)
    print(f"fit {len(fit_idx)}  calib {len(cal_idx)}\n", flush=True)

    model, (mu, sd) = train_msa(data, fit_idx, cal_idx, epochs=EPOCHS, seed=seed)

    # ---- calibration on the held-out pooled calibration split ----------------
    cal_logits, _ = predict_msa(model, X[cal_idx], S[cal_idx])
    T = fit_temperature(cal_logits, y[cal_idx])
    T = 1.0 if not np.isfinite(T) else float(np.clip(T, 0.5, 5.0))
    qhat = conformal_qhat(softmax_np(cal_logits, T), y[cal_idx], alpha=ALPHA)
    print(f"\ntemperature {T:.3f}  qhat {qhat:.3f}", flush=True)

    # intent codes on the fit split, for the probes
    z_fit = encode_msa(model, X[fit_idx], S[fit_idx])
    M_fit = ((M[fit_idx] - mu) / sd).astype(np.float32)
    v_fit = V[fit_idx]

    baseline = json.loads((RESULTS / "calmnet_full.json").read_text())
    out, rows = {}, []

    for sub in data["subjects"]:
        te = data["test"][sub]
        lg, _ = predict_msa(model, te["X"], np.full(len(te["X"]), te["subj"]))
        p_raw, p_cal = softmax_np(lg, 1.0), softmax_np(lg, T)
        pred = p_cal.argmax(1)
        correct = (pred == te["y"]).astype(int)

        acc = balanced_accuracy(te["y"], pred)
        ece_raw = expected_calibration_error(p_raw, te["y"])
        ece_cal = expected_calibration_error(p_cal, te["y"])
        auroc = confidence_auroc(p_cal.max(1), correct)
        ex80 = executed_acc_at_coverage(te["y"], p_cal, COV)
        static_cov = float(np.mean((1 - p_cal[np.arange(len(te["y"])), te["y"]]) <= qhat))
        ada_cov, ada_size, _ = adaptive_conformal(p_cal, te["y"], q0=qhat, alpha=ALPHA)

        # ---- invariance probe: movement must stay unrecoverable --------------
        z_te = encode_msa(model, te["X"], np.full(len(te["X"]), te["subj"]))
        M_te = ((te["M"] - mu) / sd).astype(np.float32)
        vt = te["valid"]
        r2 = (invariance_r2(z_fit[v_fit], M_fit[v_fit], z_te[vt], M_te[vt])
              if vt.sum() > 10 else float("nan"))

        base = baseline[sub]["summary"]
        out[sub] = {
            "bal_acc": acc, "baseline_bal_acc": base["bal_acc"],
            "delta_acc": acc - base["bal_acc"],
            "ece_raw": ece_raw, "ece_cal": ece_cal,
            "baseline_ece_cal": base["ece_cal"],
            "conf_auroc": auroc, "baseline_auroc": base["conf_auroc"],
            "exec_acc@80": ex80, "baseline_exec@80": base["exec_acc@80"],
            "conformal_cov_static": static_cov,
            "conformal_cov_adaptive": ada_cov,
            "conformal_setsize_adaptive": ada_size,
            "intent_imu_r2": r2,
            "n_test": int(len(te["y"])),
        }
        rows.append((sub, acc, base["bal_acc"], r2, auroc, ex80, ada_cov))
        print(f"  [{sub}] acc {acc:.3f} (base {base['bal_acc']:.3f}, "
              f"{acc - base['bal_acc']:+.3f}) | intent->IMU R2 {r2:+.3f} | "
              f"AUROC {auroc:.3f} | exec@80 {ex80:.3f} | cov {ada_cov:.3f}", flush=True)

    # ---- subject-identity leakage probe (pooled sanity check) ----------------
    z_all = np.concatenate([encode_msa(model, data["test"][s]["X"],
                                       np.full(len(data["test"][s]["X"]),
                                               data["test"][s]["subj"]))
                            for s in data["subjects"]])
    s_all = np.concatenate([np.full(len(data["test"][s]["X"]),
                                    data["test"][s]["subj"])
                            for s in data["subjects"]])
    leak = subject_leakage(z_fit, S[fit_idx], z_all, s_all)
    chance = 1.0 / len(data["subjects"])

    print("\n================= MULTI-SUBJECT CALM-Net =================")
    hdr = f"{'subj':8}{'MSA':>8}{'base':>8}{'delta':>8}{'R2':>8}{'AUROC':>8}{'exec@80':>9}{'cov':>7}"
    print(hdr)
    for sub, a, b, r2, au, ex, cv in rows:
        print(f"{sub:8}{a:8.3f}{b:8.3f}{a - b:+8.3f}{r2:+8.3f}{au:8.3f}{ex:9.3f}{cv:7.3f}")
    print("-" * len(hdr))
    ma = np.mean([r[1] for r in rows]); mb = np.mean([r[2] for r in rows])
    print(f"{'MEAN':8}{ma:8.3f}{mb:8.3f}{ma - mb:+8.3f}"
          f"{np.mean([r[3] for r in rows]):+8.3f}"
          f"{np.mean([r[4] for r in rows]):8.3f}"
          f"{np.mean([r[5] for r in rows]):9.3f}"
          f"{np.mean([r[6] for r in rows]):7.3f}")

    print(f"\nsubject-identity recoverable from intent code: {leak:.3f} "
          f"(chance {chance:.3f})")
    n_leak = sum(1 for r in rows if r[3] > 0)
    print(f"invariance probe positive (movement recoverable) for {n_leak}/7 subjects")
    print("VERDICT: gains are only admissible where intent->IMU R2 stays <= 0.")

    payload = {"summary": {"mean_acc": float(ma), "mean_baseline_acc": float(mb),
                           "mean_delta": float(ma - mb),
                           "subject_leakage": leak, "subject_chance": chance,
                           "temperature": T, "qhat": qhat, "seed": seed},
               "per_subject": out}
    (RESULTS / "msa.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {RESULTS / 'msa.json'}")


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
