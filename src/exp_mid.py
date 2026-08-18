"""Validate Motion-Invariant Disentanglement (MID).

Question: does disentangling against head-motion recover genuine neural decoding?
We compare MID (lam_adv>0) vs an identical model with MID off (lam_adv=0) and probe
how well head-motion can be linearly recovered from the intent subspace.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # allow numpy+torch OpenMP
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from mid import train_mid, predict_mid, encode_all, motion_probe_r2
from experiments import imu_only_baseline
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or ["sub-01", "sub-03", "sub-06"]
TRAIN, EPOCHS = [1, 2, 3], 100


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = list_sessions(sub)
    present = set(int(s) for s in np.unique(es.session))
    sess = [s for s in sess if s in present]
    train = sess[:len(TRAIN)]; test = sess[len(TRAIN):]
    tr = es.by_sessions(train); te = es.by_sessions(test)
    ti, vi = grouped_split(tr.segment, tr.y, frac=0.2, seed=0)

    Xtr, ytr, Mtr = tr.X[ti], tr.y[ti], tr.imu_feats[ti]     # 12-D IMU target
    Xval, yval = tr.X[vi], tr.y[vi]
    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-6
    M_tr = (Mtr - mu) / sd
    M_te = (te.imu_feats - mu) / sd

    out = {"subject": sub,
           "imu_only": float(np.mean(list(imu_only_baseline(es, train, test).values())))}
    for tag, lam in [("noMID", 0.0), ("MID", 1.0)]:
        model = train_mid(Xtr, ytr, Mtr, Xval, yval, epochs=EPOCHS,
                          lam_adv=lam, lam_dec=lam, seed=0)
        _, probs = predict_mid(model, te.X)
        acc = balanced_accuracy(te.y, probs.argmax(1))
        zi_tr, za_tr = encode_all(model, Xtr)
        zi_te, za_te = encode_all(model, te.X)
        r2_lin = motion_probe_r2(zi_tr, M_tr, zi_te, M_te, nonlinear=False)
        r2_mlp = motion_probe_r2(zi_tr, M_tr, zi_te, M_te, nonlinear=True)
        r2_art = motion_probe_r2(za_tr, M_tr, za_te, M_te, nonlinear=True)
        out[tag] = {"bal_acc": acc, "intent_imu_r2_lin": r2_lin,
                    "intent_imu_r2_mlp": r2_mlp, "artefact_imu_r2": r2_art}
        print(f"  [{sub}/{tag:5s}] bacc {acc:.3f} | intent->IMU R2 lin {r2_lin:.3f} "
              f"mlp {r2_mlp:.3f} | artefact->IMU R2 {r2_art:.3f}", flush=True)
    return out


if __name__ == "__main__":
    results = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} ########", flush=True)
        results[sub] = run_subject(sub)
        (RESULTS / "mid_validation.json").write_text(json.dumps(results, indent=2))

    print("\n=========== MID v2 VALIDATION (12-D multimodal IMU target) ===========")
    print(f"{'subj':8}{'IMU':>7}{'noMID':>7}{'MID':>7} | intent->IMU R2(mlp)  {'noMID':>7}{'MID':>7} | {'art R2':>7}")
    import numpy as _np
    d_acc, d_r2 = [], []
    for sub, r in results.items():
        n, m = r["noMID"], r["MID"]
        d_acc.append(m["bal_acc"] - n["bal_acc"])
        d_r2.append(m["intent_imu_r2_mlp"] - n["intent_imu_r2_mlp"])
        print(f"{sub:8}{r['imu_only']:7.3f}{n['bal_acc']:7.3f}{m['bal_acc']:7.3f} | "
              f"{'':18}{n['intent_imu_r2_mlp']:7.3f}{m['intent_imu_r2_mlp']:7.3f} | {m['artefact_imu_r2']:7.3f}")
    print(f"\nMean MID effect: acc {_np.mean(d_acc):+.3f} | intent->IMU R2(mlp) {_np.mean(d_r2):+.3f} "
          "(negative = invariance gained)")
    print("MID works if intent->IMU R2 drops (invariance) while accuracy is retained.")
