"""Apples-to-apples: run EEGNet and our band-power backbone BOTH ways on the same
data (training task, train ses 1-3, test 4-9) --
  (a) plain  (no disentanglement)      -> movement-INFLATED accuracy
  (b) MID    (movement-invariant)      -> honest accuracy + intent->IMU R^2
So EEGNet's 0.82 is finally compared to OUR 0.69 on equal footing: what does EEGNet
score when it, too, may not read movement?
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
from mid import train_mid, predict_mid, encode_all, motion_probe_r2
from graphnet import electrode_adjacency
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS = 3, 100


def run_cell(Xfit, yfit, Mfit, Xval, yval, Xte, Mte, yte, backbone, invariant, adj=None):
    lam = 1.0 if invariant else 0.0
    m = train_mid(Xfit, yfit, Mfit, Xval, yval, epochs=EPOCHS,
                  lam_adv=lam, lam_dec=lam, backbone=backbone, adj=adj, seed=0)
    acc = balanced_accuracy(yte, predict_mid(m, Xte)[1].argmax(1))
    mu, sd = Mfit.mean(0), Mfit.std(0) + 1e-6
    r2 = motion_probe_r2(encode_all(m, Xfit)[0], (Mfit - mu) / sd,
                         encode_all(m, Xte)[0], (Mte - mu) / sd, nonlinear=True)
    return acc, r2


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    tr = es.by_sessions(sess[:N_TRAIN]); te = es.by_sessions(sess[N_TRAIN:])
    ti, vi = grouped_split(tr.segment, tr.y, frac=0.2, seed=0)
    Xfit, yfit, Mfit = tr.X[ti], tr.y[ti], tr.imu_feats[ti]
    adj = electrode_adjacency(es.ch_names)          # scalp graph from electrodes.tsv
    o = {"subject": sub}
    for bb in ("eegnet", "calmnet", "graph"):
        for inv in (False, True):
            a, r = run_cell(Xfit, yfit, Mfit, tr.X[vi], tr.y[vi], te.X, te.imu_feats, te.y, bb, inv,
                            adj=adj if bb == "graph" else None)
            o[f"{bb}_{'inv' if inv else 'plain'}_acc"] = a
            o[f"{bb}_{'inv' if inv else 'plain'}_r2"] = r
    print(f"  [{sub}] invariant:  EEGNet {o['eegnet_inv_acc']:.3f} | CALMNet {o['calmnet_inv_acc']:.3f} | "
          f"GraphNet {o['graph_inv_acc']:.3f}   (plain: {o['eegnet_plain_acc']:.3f}/"
          f"{o['calmnet_plain_acc']:.3f}/{o['graph_plain_acc']:.3f})", flush=True)
    return o


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "apples.json").write_text(json.dumps(res, indent=2))
    g = lambda f: float(np.nanmean([res[s][f] for s in res]))
    print("\n================ APPLES-TO-APPLES (mean over 7 subjects) ================")
    print(f"{'':10}{'movement-inflated':>20}{'movement-INVARIANT':>22}{'intent->IMU R2':>18}")
    for bb, lab in [("eegnet", "EEGNet"), ("calmnet", "CALMNet"), ("graph", "GraphNet")]:
        print(f"{lab:10}{g(f'{bb}_plain_acc'):>20.3f}{g(f'{bb}_inv_acc'):>22.3f}"
              f"{g(f'{bb}_inv_r2'):>18.3f}")
    print("\nThe only number that matters: movement-INVARIANT accuracy (R2<=0 = honest).")
    print("GraphNet uses the scalp electrode geometry; the others discard it.")
