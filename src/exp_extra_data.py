"""Data-augmentation robustness check: add the closed-loop trials (trial01-12,
rexstate-labelled, walk/stop interleaved within each recording -> no recording-level
confound) to the training data, and re-evaluate on the SAME training-task test
sessions. walk6min/stop6min are deliberately NOT used: each is a single recording per
class, which risks a recording-identity confound.

Reports whether more data changes accuracy / calibration / confidence / the safety
(wrong-walk) metric, versus training on the training task alone."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from mid import train_mid, predict_mid
from calibrate import fit_temperature, softmax_np, expected_calibration_error
from abstain import balanced_accuracy, confidence_auroc

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
TRIALS = tuple(f"trial{i:02d}" for i in range(1, 13))
N_TRAIN, EPOCHS, COV, BETA = 3, 100, 0.8, 0.05


def _T(lg, y):
    t = fit_temperature(lg, y); return 1.0 if not np.isfinite(t) else float(np.clip(t, 0.5, 5.0))


def exec_acc(y, p, cov=COV):
    c = p.max(1); k = max(1, int(round(cov * len(y)))); idx = np.argsort(-c)[:k]
    return balanced_accuracy(y[idx], p[idx].argmax(1))


def pipeline(Xfit, yfit, Mfit, Xcal, ycal, test_epochs):
    m = train_mid(Xfit, yfit, Mfit, Xcal, ycal, epochs=EPOCHS, lam_adv=1.0, lam_dec=1.0, seed=0)
    T = _T(predict_mid(m, Xcal)[0], ycal)
    p_cal = softmax_np(predict_mid(m, Xcal)[0], T)
    tau = float(np.quantile(p_cal[ycal == 0, 1], 1 - BETA)) if (ycal == 0).any() else 0.5
    accs, eces, aurocs, execs, wrongs = [], [], [], [], []
    for te in test_epochs:
        lg = predict_mid(m, te.X)[0]; p = softmax_np(lg, T); ys = te.y
        cor = (p.argmax(1) == ys).astype(int)
        accs.append(balanced_accuracy(ys, p.argmax(1))); eces.append(expected_calibration_error(p, ys))
        aurocs.append(confidence_auroc(p.max(1), cor)); execs.append(exec_acc(ys, p))
        wrongs.append(float(np.mean((p[:, 1] >= tau) & (ys == 0))))
    m_ = lambda a: float(np.nanmean(a))
    return {"bal_acc": m_(accs), "ece": m_(eces), "auroc": m_(aurocs),
            "exec@80": m_(execs), "wrong_walk": m_(wrongs)}


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    tr = es.by_sessions(train); ti, ci = grouped_split(tr.segment, tr.y, frac=0.3, seed=0)
    test_epochs = [es.by_sessions([s]) for s in test]

    base = pipeline(tr.X[ti], tr.y[ti], tr.imu_feats[ti], tr.X[ci], tr.y[ci], test_epochs)

    extra = build_epochs(subject=sub, sessions=train, tasks=TRIALS)     # cached separately
    Xfit = np.concatenate([tr.X[ti], extra.X]); yfit = np.concatenate([tr.y[ti], extra.y])
    Mfit = np.concatenate([tr.imu_feats[ti], extra.imu_feats])
    aug = pipeline(Xfit, yfit, Mfit, tr.X[ci], tr.y[ci], test_epochs)

    print(f"  [{sub}] +{len(extra):5d} trial epochs | acc {base['bal_acc']:.3f}->{aug['bal_acc']:.3f} | "
          f"ECE {base['ece']:.3f}->{aug['ece']:.3f} | AUROC {base['auroc']:.3f}->{aug['auroc']:.3f} | "
          f"exec@80 {base['exec@80']:.3f}->{aug['exec@80']:.3f} | wrong-walk "
          f"{base['wrong_walk']:.3f}->{aug['wrong_walk']:.3f}", flush=True)
    return {"subject": sub, "n_extra": len(extra), "base": base, "aug": aug}


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (train-task vs +trials) ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "extra_data.json").write_text(json.dumps(res, indent=2))
    print("\n======= +closed-loop trials as training data (mean over 7) =======")
    for f in ["bal_acc", "ece", "auroc", "exec@80", "wrong_walk"]:
        b = np.mean([res[s]["base"][f] for s in res]); a = np.mean([res[s]["aug"][f] for s in res])
        sb = np.std([res[s]["base"][f] for s in res]); sa = np.std([res[s]["aug"][f] for s in res])
        print(f"  {f:12s}: {b:.3f} (sd {sb:.3f})  ->  {a:.3f} (sd {sa:.3f})")
    print(f"  extra trial epochs added (mean): {np.mean([res[s]['n_extra'] for s in res]):.0f}")
