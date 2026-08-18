"""The field's benchmark, run properly.

Standard protocol (NO disentanglement - the same setting every published EEG decoder
reports), but at FULL data: training task + all 12 closed-loop trials from the train
sessions. Test on held-out training-task sessions 4-9. Architecture vs architecture at
equal data.

Reference (training-task data only, same test set): EEGNet 0.813 | CALMNet 0.814.
Question: at full data, which architecture wins, and by how much?
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
from mid import train_mid, predict_mid
from graphnet import electrode_adjacency
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRIALS = tuple(f"trial{i:02d}" for i in range(1, 13))
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS = 3, 110
BACKBONES = ("eegnet", "calmnet", "graph")


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    tr = es.by_sessions(sess[:N_TRAIN]); te = es.by_sessions(sess[N_TRAIN:])
    ti, vi = grouped_split(tr.segment, tr.y, frac=0.2, seed=0)
    ex = build_epochs(subject=sub, sessions=sess[:N_TRAIN], tasks=TRIALS)   # cached

    Xfit = np.concatenate([tr.X[ti], ex.X])
    yfit = np.concatenate([tr.y[ti], ex.y])
    Mfit = np.concatenate([tr.imu_feats[ti], ex.imu_feats])
    adj = electrode_adjacency(es.ch_names)

    out = {"subject": sub, "n_train": int(len(Xfit))}
    for bb in BACKBONES:
        m = train_mid(Xfit, yfit, Mfit, tr.X[vi], tr.y[vi], epochs=EPOCHS,
                      lam_adv=0.0, lam_dec=0.0,            # standard protocol
                      backbone=bb, adj=adj if bb == "graph" else None, seed=0)
        out[bb] = balanced_accuracy(te.y, predict_mid(m, te.X)[1].argmax(1))
    print(f"  [{sub}] n={out['n_train']:5d} | EEGNet {out['eegnet']:.3f} | "
          f"CALMNet {out['calmnet']:.3f} | GraphNet {out['graph']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (full data, standard protocol) ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "win.json").write_text(json.dumps(res, indent=2))
    g = lambda f: float(np.nanmean([res[s][f] for s in res]))
    print("\n============ FIELD BENCHMARK, FULL DATA (mean over 7) ============")
    for bb in BACKBONES:
        print(f"  {bb:10}: {g(bb):.3f}")
    print("\n  reference, training-task data only: EEGNet 0.813 | CALMNet 0.814")
    best = max(BACKBONES, key=g)
    print(f"  winner: {best} at {g(best):.3f}  (vs EEGNet baseline 0.813 -> "
          f"{g(best) - 0.813:+.3f})")
