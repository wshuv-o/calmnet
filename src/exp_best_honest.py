"""The definitive 'best honest number' experiment.

Full training data (training task + all 12 closed-loop trials, sessions 1-3) +
strongest disentanglement (HSIC + selection fix). Tested on the SAME held-out
training-task sessions 4-9 as the 0.694 baseline. Reports accuracy AND the
invariance probe together, so we can tell a real gain from movement leakage.

Reference points (mean over 7, same test set):
  EEGNet 0.819 (movement-inflated) | IMU-only 0.839 (pure movement)
  MID band-power, training-only     0.694 (honest, under-trained)
  +trials, no HSIC                  0.753 (mixed / partial leakage)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from mid import train_mid_hsic, predict_mid, encode_all, motion_probe_r2
from calibrate import fit_temperature, softmax_np, expected_calibration_error, conformal_qhat, adaptive_conformal
from abstain import balanced_accuracy, confidence_auroc

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRIALS = tuple(f"trial{i:02d}" for i in range(1, 13))
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS, ALPHA, COV, BETA = 3, 130, 0.1, 0.8, 0.05


def _T(lg, y):
    t = fit_temperature(lg, y); return 1.0 if not np.isfinite(t) else float(np.clip(t, 0.5, 5.0))


def exec_acc(y, p, cov=COV):
    c = p.max(1); k = max(1, int(round(cov * len(y)))); idx = np.argsort(-c)[:k]
    return balanced_accuracy(y[idx], p[idx].argmax(1))


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    tr = es.by_sessions(train)
    ti, ci = grouped_split(tr.segment, tr.y, frac=0.3, seed=0)        # fit / calibration
    fi, vi = grouped_split(tr.segment[ti], tr.y[ti], frac=0.2, seed=1)  # fit / selection-val
    ex = build_epochs(subject=sub, sessions=train, tasks=TRIALS)       # closed-loop trials (cached)

    Xfit = np.concatenate([tr.X[ti][fi], ex.X])
    yfit = np.concatenate([tr.y[ti][fi], ex.y])
    Mfit = np.concatenate([tr.imu_feats[ti][fi], ex.imu_feats])
    Xval, yval, Mval = tr.X[ti][vi], tr.y[ti][vi], tr.imu_feats[ti][vi]

    model = train_mid_hsic(Xfit, yfit, Mfit, Xval, yval, Mval, epochs=EPOCHS,
                           lam_hsic=4.0, lam_sel=1.0, seed=0)

    Tg = _T(predict_mid(model, tr.X[ci])[0], tr.y[ci])
    p_cal = softmax_np(predict_mid(model, tr.X[ci])[0], Tg)
    qhat = conformal_qhat(p_cal, tr.y[ci], alpha=ALPHA)
    tau_walk = float(np.quantile(p_cal[tr.y[ci] == 0, 1], 1 - BETA)) if (tr.y[ci] == 0).any() else 0.5

    accs, eces, aurocs, execs, wrongs, Ps, Ys = [], [], [], [], [], [], []
    for s in test:
        te = es.by_sessions([s]); ys = te.y
        lg = predict_mid(model, te.X)[0]; p = softmax_np(lg, Tg)
        cor = (p.argmax(1) == ys).astype(int)
        accs.append(balanced_accuracy(ys, p.argmax(1))); eces.append(expected_calibration_error(p, ys))
        aurocs.append(confidence_auroc(p.max(1), cor)); execs.append(exec_acc(ys, p))
        wrongs.append(float(np.mean((p[:, 1] >= tau_walk) & (ys == 0)))); Ps.append(p); Ys.append(ys)
    P, Y = np.concatenate(Ps), np.concatenate(Ys)
    ada_cov, _, _ = adaptive_conformal(P, Y, q0=qhat, alpha=ALPHA)

    # invariance: intent -> IMU R^2 (nonlinear) on the test set
    Mte = np.concatenate([es.by_sessions([s]).imu_feats for s in test])
    Xte = np.concatenate([es.by_sessions([s]).X for s in test])
    mu, sd = Mfit.mean(0), Mfit.std(0) + 1e-6
    r2 = motion_probe_r2(encode_all(model, Xfit)[0][:2000], ((Mfit - mu) / sd)[:2000],
                         encode_all(model, Xte)[0], (Mte - mu) / sd, nonlinear=True)

    m = lambda a: float(np.nanmean(a))
    out = {"subject": sub, "acc": m(accs), "intent_imu_r2": r2, "ece": m(eces),
           "auroc": m(aurocs), "exec@80": m(execs), "adaptive_cov": ada_cov, "wrong_walk": m(wrongs),
           "n_train": len(Xfit)}
    tag = "INVARIANT" if r2 < 0.10 else "leak-risk"
    print(f"  [{sub}] acc {out['acc']:.3f} | intent->IMU R2 {r2:+.3f} ({tag}) | ECE {out['ece']:.3f} | "
          f"AUROC {out['auroc']:.3f} | exec@80 {out['exec@80']:.3f} | cov {ada_cov:.3f} | "
          f"wrong-walk {out['wrong_walk']:.3f} | ntrain {len(Xfit)}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (best honest: +trials +HSIC) ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "best_honest.json").write_text(json.dumps(res, indent=2))
    g = lambda f: float(np.nanmean([res[s][f] for s in res]))
    print("\n================= BEST HONEST MODEL (mean over 7) =================")
    print(f"  movement-invariant accuracy : {g('acc'):.3f}")
    print(f"  intent->IMU R2 (want <=0)    : {g('intent_imu_r2'):+.3f}   <- proves it is honest")
    print(f"  ECE {g('ece'):.3f} | AUROC {g('auroc'):.3f} | exec@80 {g('exec@80'):.3f} | "
          f"adaptive cov {g('adaptive_cov'):.3f} | wrong-walk {g('wrong_walk'):.3f}")
    print("  ---- reference ----")
    print("  EEGNet 0.819 (movement) | IMU 0.839 (movement) | MID train-only 0.694 | +trials no-HSIC 0.753")
