"""Does the trial-augmentation accuracy gain stay movement-INVARIANT, or does the
extra (closed-loop, moving) data let movement leak back into the intent code?
Measures intent->IMU R^2 (nonlinear) for base vs augmented models."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from mid import train_mid, predict_mid, encode_all, motion_probe_r2
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRIALS = tuple(f"trial{i:02d}" for i in range(1, 13))
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS = 3, 100


def r2_and_acc(model, Xfit, Mfit, Xte, Mte, yte):
    _, p = predict_mid(model, Xte)
    acc = balanced_accuracy(yte, p.argmax(1))
    mu, sd = Mfit.mean(0), Mfit.std(0) + 1e-6
    r2 = motion_probe_r2(encode_all(model, Xfit)[0], (Mfit - mu) / sd,
                         encode_all(model, Xte)[0], (Mte - mu) / sd, nonlinear=True)
    return acc, r2


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    tr = es.by_sessions(train); ti, ci = grouped_split(tr.segment, tr.y, frac=0.3, seed=0)
    te = es.by_sessions(test)
    Xfit, yfit, Mfit = tr.X[ti], tr.y[ti], tr.imu_feats[ti]

    base = train_mid(Xfit, yfit, Mfit, tr.X[ci], tr.y[ci], epochs=EPOCHS, lam_adv=1.0, lam_dec=1.0, seed=0)
    a_b, r_b = r2_and_acc(base, Xfit, Mfit, te.X, te.imu_feats, te.y)

    ex = build_epochs(subject=sub, sessions=train, tasks=TRIALS)
    Xa = np.concatenate([Xfit, ex.X]); ya = np.concatenate([yfit, ex.y]); Ma = np.concatenate([Mfit, ex.imu_feats])
    aug = train_mid(Xa, ya, Ma, tr.X[ci], tr.y[ci], epochs=EPOCHS, lam_adv=1.0, lam_dec=1.0, seed=0)
    a_a, r_a = r2_and_acc(aug, Xa, Ma, te.X, te.imu_feats, te.y)

    print(f"  [{sub}] acc {a_b:.3f}->{a_a:.3f} | intent->IMU R2 {r_b:+.3f}->{r_a:+.3f}  "
          f"({'invariant kept' if r_a <= r_b + 0.05 else 'LEAKAGE: R2 rose'})", flush=True)
    return {"subject": sub, "acc_base": a_b, "acc_aug": a_a, "r2_base": r_b, "r2_aug": r_a}


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "extra_invariance.json").write_text(json.dumps(res, indent=2))
    g = lambda f: np.mean([res[s][f] for s in res])
    print("\n===== trial-augmentation: accuracy vs invariance (mean over 7) =====")
    print(f"accuracy:       {g('acc_base'):.3f} -> {g('acc_aug'):.3f}")
    print(f"intent->IMU R2: {g('r2_base'):+.3f} -> {g('r2_aug'):+.3f}")
    print("If R2 stays low/negative, the accuracy gain is genuine movement-invariant signal.")
    print("If R2 rises, the extra (moving) data leaked movement back in.")
