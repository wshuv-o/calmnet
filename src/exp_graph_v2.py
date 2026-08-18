"""GraphNet v2 test: did fixing the spatial bottleneck close the gap?

v1 defect: a single node-attention vector -> ONE spatial pattern (vs EEGNet's 32).
v2 fix   : K=4 independent spatial patterns over the electrode graph (feat_dim 24->96),
           random init to break head symmetry.

Runs the graph backbone only (plain + movement-invariant) and compares against the
already-measured EEGNet / CALMNet / GraphNet-v1 numbers in apples.json.
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
REF = json.loads((RESULTS / "apples.json").read_text()) if (RESULTS / "apples.json").exists() else {}
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS = 3, 100


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    tr = es.by_sessions(sess[:N_TRAIN]); te = es.by_sessions(sess[N_TRAIN:])
    ti, vi = grouped_split(tr.segment, tr.y, frac=0.2, seed=0)
    Xfit, yfit, Mfit = tr.X[ti], tr.y[ti], tr.imu_feats[ti]
    adj = electrode_adjacency(es.ch_names)
    out = {"subject": sub}
    for inv in (False, True):
        lam = 1.0 if inv else 0.0
        m = train_mid(Xfit, yfit, Mfit, tr.X[vi], tr.y[vi], epochs=EPOCHS,
                      lam_adv=lam, lam_dec=lam, backbone="graph", adj=adj, seed=0)
        acc = balanced_accuracy(te.y, predict_mid(m, te.X)[1].argmax(1))
        mu, sd = Mfit.mean(0), Mfit.std(0) + 1e-6
        r2 = motion_probe_r2(encode_all(m, Xfit)[0], (Mfit - mu) / sd,
                             encode_all(m, te.X)[0], (te.imu_feats - mu) / sd, nonlinear=True)
        out[f"graph2_{'inv' if inv else 'plain'}_acc"] = acc
        out[f"graph2_{'inv' if inv else 'plain'}_r2"] = r2
        if inv:
            out["top_electrodes"] = m.backbone.top_electrodes(es.ch_names, n=6)
    r = REF.get(sub, {})
    print(f"  [{sub}] GraphNet v1 {r.get('graph_inv_acc', float('nan')):.3f} -> v2 "
          f"{out['graph2_inv_acc']:.3f} (R2 {out['graph2_inv_r2']:+.2f}) | ref: EEGNet "
          f"{r.get('eegnet_inv_acc', float('nan')):.3f} CALMNet {r.get('calmnet_inv_acc', float('nan')):.3f}",
          flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (GraphNet v2) ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "graph_v2.json").write_text(json.dumps(res, indent=2))
    g = lambda f, d=res: float(np.nanmean([d[s][f] for s in d if f in d[s]]))
    gr = lambda f: float(np.nanmean([REF[s][f] for s in res if s in REF and f in REF[s]]))
    print("\n=========== GraphNet v2 (bottleneck fixed) vs the field ===========")
    print(f"  EEGNet      invariant : {gr('eegnet_inv_acc'):.3f}")
    print(f"  CALMNet     invariant : {gr('calmnet_inv_acc'):.3f}")
    print(f"  GraphNet v1 invariant : {gr('graph_inv_acc'):.3f}   (single spatial pattern)")
    print(f"  GraphNet v2 invariant : {g('graph2_inv_acc'):.3f}   (K=4 patterns)  <- the test")
    print(f"    v2 plain (with movement): {g('graph2_plain_acc'):.3f} | v2 intent->IMU R2 {g('graph2_inv_r2'):+.3f}")
    print("\nVerdict: v2 must beat ~0.65 to be a real architecture contribution.")
    ex = res[SUBJECTS[0]].get("top_electrodes", [])
    if ex:
        print("  sub-01 top electrodes by graph readout:", ", ".join(f"{n}({w:.3f})" for n, w in ex))
