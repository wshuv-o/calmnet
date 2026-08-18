"""Full CALM-Net with the described encoder: multi-band SPD tangent + XFCA + MID,
plus temperature scaling, adaptive conformal, selective abstention, and source-free
CORAL alignment. Reports the same metrics as the band-power pipeline, per subject."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions, session_days
from splits import grouped_split
from riemann import (tangent_features, train_riemann, predict_riemann, encode_riemann,
                     coral_standardise, session_coral)
from calibrate import fit_temperature, softmax_np, expected_calibration_error, conformal_qhat, adaptive_conformal
from abstain import balanced_accuracy, confidence_auroc, selective_risk_at_coverage
from mid import motion_probe_r2

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS, ALPHA, COV = 3, 120, 0.1, 0.8


def exec_acc(y, probs, cov=COV):
    conf = probs.max(1); k = max(1, int(round(cov * len(y))))
    idx = np.argsort(-conf)[:k]
    return balanced_accuracy(y[idx], probs[idx].argmax(1))


def run_subject(sub):
    es = build_epochs(subject=sub)
    V = tangent_features(es, sub)                          # (N, nb, tri)
    days = session_days(sub)
    present = set(int(s) for s in np.unique(es.session))
    sess = [s for s in list_sessions(sub) if s in present]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    tri = np.isin(es.session, train)
    Vtr, ytr, Mtr, seg = V[tri], es.y[tri], es.imu_feats[tri], es.segment[tri]
    ti, ci = grouped_split(seg, ytr, frac=0.3, seed=0)
    mu, sd = Vtr[ti].mean(0, keepdims=True), Vtr[ti].std(0, keepdims=True) + 1e-6
    Vfit, Vcal = coral_standardise(Vtr[ti], mu, sd), coral_standardise(Vtr[ci], mu, sd)

    # MID model (full pipeline) + a no-MID twin for the invariance probe
    m = train_riemann(Vfit, ytr[ti], Mtr[ti], Vcal, ytr[ci], epochs=EPOCHS, lam_adv=1.0, lam_dec=1.0, seed=0)
    m0 = train_riemann(Vfit, ytr[ti], Mtr[ti], Vcal, ytr[ci], epochs=EPOCHS, lam_adv=0.0, lam_dec=0.0, seed=0)

    T = fit_temperature(predict_riemann(m, Vcal)[0], ytr[ci])
    T = 1.0 if not np.isfinite(T) else float(np.clip(T, 0.5, 5.0))
    qhat = conformal_qhat(softmax_np(predict_riemann(m, Vcal)[0], T), ytr[ci], alpha=ALPHA)

    per, Ps, Ys = {}, [], []
    for s in test:
        si = es.session == s
        Vs = session_coral(V[si]); ys = es.y[si]              # CORAL align (source-free)
        lg, _ = predict_riemann(m, Vs)
        p_raw, p_cal = softmax_np(lg, 1.0), softmax_np(lg, T)
        cor = (p_cal.argmax(1) == ys).astype(int)
        per[s] = {"day": days.get(s, -1), "bal_acc": balanced_accuracy(ys, p_cal.argmax(1)),
                  "ece_raw": expected_calibration_error(p_raw, ys), "ece_cal": expected_calibration_error(p_cal, ys),
                  "conf_auroc": confidence_auroc(p_cal.max(1), cor), "exec_acc@80": exec_acc(ys, p_cal),
                  "sel_risk@80": selective_risk_at_coverage(p_cal.max(1), cor, COV)}
        Ps.append(p_cal); Ys.append(ys)
    P, Y = np.concatenate(Ps), np.concatenate(Ys)
    static_cov = float(np.mean((1 - P[np.arange(len(Y)), Y]) <= qhat))
    ada_cov, ada_size, _ = adaptive_conformal(P, Y, q0=qhat, alpha=ALPHA)

    # invariance probe (nonlinear): intent -> IMU R^2, no-MID vs MID
    Mte = np.concatenate([es.imu_feats[es.session == s] for s in test])
    m_mu, m_sd = Mtr[ti].mean(0), Mtr[ti].std(0) + 1e-6
    Vte_c = np.concatenate([session_coral(V[es.session == s]) for s in test])
    def r2(model):
        zi_tr, zi_te = encode_riemann(model, Vfit), encode_riemann(model, Vte_c)
        return motion_probe_r2(zi_tr, (Mtr[ti]-m_mu)/m_sd, zi_te, (Mte-m_mu)/m_sd, nonlinear=True)
    r2_mid, r2_no = r2(m), r2(m0)

    mn = lambda f: float(np.nanmean([v[f] for v in per.values()]))
    summ = {"temperature": T, "bal_acc": mn("bal_acc"), "ece_raw": mn("ece_raw"), "ece_cal": mn("ece_cal"),
            "conf_auroc": mn("conf_auroc"), "exec_acc@80": mn("exec_acc@80"), "sel_risk@80": mn("sel_risk@80"),
            "conformal_cov_static": static_cov, "conformal_cov_adaptive": ada_cov,
            "intent_imu_r2_noMID": r2_no, "intent_imu_r2_MID": r2_mid, "target_cov": 1 - ALPHA}
    print(f"  [{sub}] invAcc {summ['bal_acc']:.3f} | ECE {summ['ece_raw']:.3f}->{summ['ece_cal']:.3f} | "
          f"AUROC {summ['conf_auroc']:.3f} | exec@80 {summ['exec_acc@80']:.3f} | cov {static_cov:.3f}/"
          f"{ada_cov:.3f} | intent->IMU R2 {r2_no:.3f}->{r2_mid:.3f}", flush=True)
    return {"subject": sub, "train": train, "test": test, "summary": summ, "sessions": per}


if __name__ == "__main__":
    out = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (Riemannian + XFCA) ########", flush=True)
        out[sub] = run_subject(sub)
        (RESULTS / "calmnet_riemann.json").write_text(json.dumps(out, indent=2))
    print("\n============ FULL CALM-Net (Riemannian+XFCA encoder) ============")
    hdr = f"{'subj':8}{'invAcc':>8}{'ECEcal':>8}{'AUROC':>7}{'exec@80':>9}{'cov.adpt':>9}{'R2 no->MID':>14}"
    print(hdr)
    agg = {}
    for sub, r in out.items():
        s = r["summary"]
        for k in ["bal_acc", "ece_cal", "conf_auroc", "exec_acc@80", "conformal_cov_adaptive",
                  "intent_imu_r2_noMID", "intent_imu_r2_MID"]:
            agg.setdefault(k, []).append(s[k])
        print(f"{sub:8}{s['bal_acc']:8.3f}{s['ece_cal']:8.3f}{s['conf_auroc']:7.3f}{s['exec_acc@80']:9.3f}"
              f"{s['conformal_cov_adaptive']:9.3f}{s['intent_imu_r2_noMID']:8.3f}->{s['intent_imu_r2_MID']:.3f}")
    print("-" * len(hdr))
    print(f"{'MEAN':8}{np.mean(agg['bal_acc']):8.3f}{np.mean(agg['ece_cal']):8.3f}{np.mean(agg['conf_auroc']):7.3f}"
          f"{np.mean(agg['exec_acc@80']):9.3f}{np.mean(agg['conformal_cov_adaptive']):9.3f}"
          f"{np.mean(agg['intent_imu_r2_noMID']):8.3f}->{np.mean(agg['intent_imu_r2_MID']):.3f}")
    (RESULTS / "calmnet_riemann.json").write_text(json.dumps(out, indent=2))
