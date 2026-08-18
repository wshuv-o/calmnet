"""Riemannian+XFCA with improved disentanglement (HSIC + selection fix), full pipeline.
Compares against the previous Riemannian result (calmnet_riemann.json)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions, session_days
from splits import grouped_split
from riemann import (tangent_features, train_riemann_v2, predict_riemann, encode_riemann,
                     coral_standardise, session_coral)
from calibrate import fit_temperature, softmax_np, expected_calibration_error, conformal_qhat, adaptive_conformal
from abstain import balanced_accuracy, confidence_auroc
from mid import motion_probe_r2

RESULTS = Path(__file__).resolve().parent.parent / "results"
OLD = json.loads((RESULTS / "calmnet_riemann.json").read_text())
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS, ALPHA, COV = 3, 120, 0.1, 0.8


def exec_acc(y, probs, cov=COV):
    conf = probs.max(1); k = max(1, int(round(cov * len(y))))
    idx = np.argsort(-conf)[:k]
    return balanced_accuracy(y[idx], probs[idx].argmax(1))


def run_subject(sub):
    es = build_epochs(subject=sub); V = tangent_features(es, sub); days = session_days(sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    tri = np.isin(es.session, train)
    Vtr, ytr, Mtr, seg = V[tri], es.y[tri], es.imu_feats[tri], es.segment[tri]
    ti, ci = grouped_split(seg, ytr, frac=0.3, seed=0)
    ai, vi = grouped_split(seg[ti], ytr[ti], frac=0.2, seed=1)      # inner val for selection
    mu, sd = Vtr[ti].mean(0, keepdims=True), Vtr[ti].std(0, keepdims=True) + 1e-6
    Vfit = coral_standardise(Vtr[ti], mu, sd); Vcal = coral_standardise(Vtr[ci], mu, sd)

    m = train_riemann_v2(Vfit[ai], ytr[ti][ai], Mtr[ti][ai], Vfit[vi], ytr[ti][vi], Mtr[ti][vi],
                         epochs=EPOCHS, lam_hsic=4.0, lam_sel=1.0, seed=0)

    T = fit_temperature(predict_riemann(m, Vcal)[0], ytr[ci])
    T = 1.0 if not np.isfinite(T) else float(np.clip(T, 0.5, 5.0))
    qhat = conformal_qhat(softmax_np(predict_riemann(m, Vcal)[0], T), ytr[ci], alpha=ALPHA)

    per, Ps, Ys = {}, [], []
    for s in test:
        si = es.session == s
        lg, _ = predict_riemann(m, session_coral(V[si])); ys = es.y[si]
        p_raw, p_cal = softmax_np(lg, 1.0), softmax_np(lg, T)
        cor = (p_cal.argmax(1) == ys).astype(int)
        per[s] = {"bal_acc": balanced_accuracy(ys, p_cal.argmax(1)),
                  "ece_cal": expected_calibration_error(p_cal, ys),
                  "conf_auroc": confidence_auroc(p_cal.max(1), cor), "exec_acc@80": exec_acc(ys, p_cal)}
        Ps.append(p_cal); Ys.append(ys)
    P, Y = np.concatenate(Ps), np.concatenate(Ys)
    ada_cov, _, _ = adaptive_conformal(P, Y, q0=qhat, alpha=ALPHA)

    Mte = np.concatenate([es.imu_feats[es.session == s] for s in test])
    Vte_c = np.concatenate([session_coral(V[es.session == s]) for s in test])
    m_mu, m_sd = Mtr[ti].mean(0), Mtr[ti].std(0) + 1e-6
    r2 = motion_probe_r2(encode_riemann(m, Vfit), (Mtr[ti]-m_mu)/m_sd,
                         encode_riemann(m, Vte_c), (Mte-m_mu)/m_sd, nonlinear=True)

    mn = lambda f: float(np.nanmean([v[f] for v in per.values()]))
    o = OLD[sub]["summary"]
    summ = {"bal_acc": mn("bal_acc"), "ece_cal": mn("ece_cal"), "conf_auroc": mn("conf_auroc"),
            "exec_acc@80": mn("exec_acc@80"), "conformal_cov_adaptive": ada_cov,
            "intent_imu_r2_MID": r2, "old_acc": o["bal_acc"], "old_r2_MID": o["intent_imu_r2_MID"],
            "noMID_r2": o["intent_imu_r2_noMID"]}
    print(f"  [{sub}] acc {o['bal_acc']:.3f}->{summ['bal_acc']:.3f} | R2(MID) {o['intent_imu_r2_MID']:+.3f}"
          f"->{r2:+.3f} (noMID {o['intent_imu_r2_noMID']:+.3f}) | exec@80 {summ['exec_acc@80']:.3f} | "
          f"cov {ada_cov:.3f}", flush=True)
    return {"subject": sub, "summary": summ}


if __name__ == "__main__":
    out = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (Riemannian + HSIC-MID) ########", flush=True)
        out[sub] = run_subject(sub)
        (RESULTS / "riemann_hsic.json").write_text(json.dumps(out, indent=2))
    print("\n===== Riemannian + HSIC-MID vs old Riemannian-MID (mean over 7) =====")
    g = lambda f: np.mean([out[s]["summary"][f] for s in out])
    print(f"accuracy:          old {g('old_acc'):.3f}  ->  new {g('bal_acc'):.3f}")
    print(f"intent->IMU R2:    noMID {g('noMID_r2'):+.3f} | old-MID {g('old_r2_MID'):+.3f} | new-MID {g('intent_imu_r2_MID'):+.3f}")
    print(f"exec@80: {g('exec_acc@80'):.3f} | adaptive coverage: {g('conformal_cov_adaptive'):.3f} (target 0.90)")
    print("Goal: keep accuracy high while driving new-MID R2 well below noMID (invariance).")
