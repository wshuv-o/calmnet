"""Do the MRCP band and the ERD band carry COMPLEMENTARY information?

The grand-average MRCP is focal and 3.4x its surrogate, but across 7 subjects it
does not reach significance (t p=0.131, Wilcoxon p=0.109; negative in 6/7). That
is the wrong test for this purpose anyway: these decoders are trained PER
SUBJECT, so what matters is whether 0.1-3 Hz carries decodable information within
a subject, not whether a group ERP is significant across seven of them.

This asks the question the proposed architecture actually rests on. Three
representations, each matched to its generator, same windows, same splits:

  mrcp   0.1-3 Hz, TIME-DOMAIN waveform, downsampled.
         Deliberately NOT squared. The MRCP is a slow deflection whose sign and
         slope carry the information; squaring destroys exactly that, and every
         model in this project squares. This is the representation no model here
         has ever been given.

  erd    8-30 Hz, log band-power per channel.
         The conventional representation, and the one the whole project uses.

  both   concatenation of the two.

The architecture is only worth building if `both` > max(`mrcp`, `erd`). If the
MRCP band adds nothing over ERD, a dual-pathway network is two pathways carrying
one signal, and the idea is dead regardless of how good the physiology story is.

Reported per representation: balanced accuracy, and -- because the physiological
claim is specifically about TIMING -- detection latency and false-onset rate at
Stop->Walk transitions. The MRCP precedes movement, so if it is real and usable
it should show up as EARLIER detection, not only as higher accuracy. That is a
falsifiable prediction that accuracy alone cannot test.

Writes results/dualband.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from abstain import balanced_accuracy
from exp_temporal import streams, onset_metrics
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "dualband.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3
WIN = 4.0
DS = 10          # 100 Hz -> 10 Hz for the MRCP waveform; 3 Hz low-pass makes
                 # anything faster redundant, and it keeps the feature count sane


def load_band(sub, lo, hi, zscore=False):
    return build_epochs(subject=sub, win=WIN, step=0.5, l_freq=lo, h_freq=hi,
                        zscore=zscore)


def mrcp_feats(X):
    """Down-sampled time-domain waveform, per channel, mean-centred per epoch.

    Centring removes each epoch's DC offset (which is drift, not MRCP) while
    preserving the SHAPE -- the slope and polarity that carry the signal.
    """
    Xc = X - X.mean(axis=2, keepdims=True)
    n = Xc.shape[2] // DS
    return Xc[:, :, : n * DS].reshape(len(Xc), Xc.shape[1], n, DS).mean(-1) \
             .reshape(len(Xc), -1)


def erd_feats(X):
    return FE.log_var(FE.bandpass(X, 8, 30))


def evaluate(Ff, yf, Ft, yt, st, tt, gt):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=4000, C=0.05,
                                           class_weight="balanced"))
    clf.fit(Ff, yf)
    pred = clf.predict(Ft)
    strm = streams(st, tt, gt)
    m = onset_metrics(yt, pred, strm)
    return balanced_accuracy(yt, pred), m


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    per = {}
    for sub in SUBJECTS:
        try:
            es_m = load_band(sub, 0.1, 3.0)
            es_e = load_band(sub, 8.0, 30.0)
        except Exception as e:
            print("  [skip] %s: %s: %s" % (sub, type(e).__name__, e), flush=True)
            continue
        if len(es_m) != len(es_e):
            print("  [skip] %s: band epoch counts differ (%d vs %d)"
                  % (sub, len(es_m), len(es_e)), flush=True)
            continue
        pres = sorted(set(int(v) for v in np.unique(es_e.session)))
        sess = [s for s in list_sessions(sub) if s in pres]
        tr = np.isin(es_e.session, sess[:N_TRAIN])
        ti, _ = grouped_split(es_e.segment[tr], es_e.y[tr], frac=0.3, seed=0)
        y = es_e.y
        yf, yt = y[tr][ti], y[~tr]
        st, tt, gt = es_e.session[~tr], es_e.task[~tr], es_e.segment[~tr]

        M = mrcp_feats(es_m.X); E = erd_feats(es_e.X)
        reps = {"mrcp": (M[tr][ti], M[~tr]),
                "erd": (E[tr][ti], E[~tr]),
                "both": (np.concatenate([M[tr][ti], E[tr][ti]], 1),
                         np.concatenate([M[~tr], E[~tr]], 1))}
        per[sub] = {}
        for name, (Ff, Ft) in reps.items():
            acc, m = evaluate(Ff, yf, Ft, yt, st, tt, gt)
            per[sub][name] = {"acc": acc, **m}
        print("  %s  mrcp %.3f | erd %.3f | both %.3f   (lat %.1f/%.1f/%.1fs)"
              % (sub, per[sub]["mrcp"]["acc"], per[sub]["erd"]["acc"],
                 per[sub]["both"]["acc"], per[sub]["mrcp"]["latency_s"],
                 per[sub]["erd"]["latency_s"], per[sub]["both"]["latency_s"]),
              flush=True)

    if not per:
        print("no subjects"); return
    print("\n" + "=" * 72)
    print("%-8s %8s %8s %12s %10s %10s" %
          ("rep", "acc", "sd", "onsets/min", "latency", "missed"))
    print("-" * 72)
    for name in ("mrcp", "erd", "both"):
        a = [per[s][name]["acc"] for s in per]
        o = [per[s][name]["false_onsets_per_min"] for s in per]
        l = [per[s][name]["latency_s"] for s in per]
        ms = [per[s][name]["missed_onsets"] for s in per]
        out[name] = {"acc": float(np.mean(a)), "acc_sd": float(np.std(a)),
                     "onsets_per_min": float(np.nanmean(o)),
                     "latency_s": float(np.nanmean(l)),
                     "missed": float(np.nanmean(ms)),
                     "per_subject": {s: per[s][name]["acc"] for s in per}}
        print("%-8s %8.3f %8.3f %12.2f %10.2f %10.2f"
              % (name, np.mean(a), np.std(a), np.nanmean(o), np.nanmean(l),
                 np.nanmean(ms)))
    print("-" * 72)
    # the question the architecture depends on
    b = np.array([per[s]["both"]["acc"] for s in per])
    e = np.array([per[s]["erd"]["acc"] for s in per])
    m = np.array([per[s]["mrcp"]["acc"] for s in per])
    d = b - np.maximum(e, m)
    print("PAIRED both - max(erd, mrcp): %+.3f +- %.3f  (positive in %d/%d subjects)"
          % (d.mean(), d.std(), int((d > 0).sum()), len(d)))
    print("=> dual pathway is %s"
          % ("JUSTIFIED: the bands are complementary" if d.mean() > 0.005
             else "NOT justified: the MRCP band adds nothing over ERD"))
    out["_paired_both_minus_best"] = {"mean": float(d.mean()), "sd": float(d.std()),
                                      "n_positive": int((d > 0).sum()), "n": len(d)}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
