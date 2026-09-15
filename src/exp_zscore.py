"""Does per-window z-scoring delete the signal this project is trying to decode?

Found by reading a reference implementation from the lab that recorded the
second cohort (shonaka/EEG-neural-decoding, Contreras-Vidal group). They fit ONE
StandardScaler on the training set and apply it to test. `dataio.build_epochs`
does something categorically different:

    mu = X.mean(axis=2, keepdims=True)
    sd = X.std (axis=2, keepdims=True) + 1e-7
    X  = (X - mu) / sd

axis=2 is time, so this forces EVERY CHANNEL of EVERY WINDOW to exactly unit
variance, independently.

Why that is not a cosmetic choice here. Walk versus Stop is separated in motor
cortex by mu/beta event-related desynchronisation -- a power DECREASE. Power is
variance. Normalising each channel of each window to unit variance removes
absolute power by construction, and because it is done per channel it also
removes the SPATIAL power contrast, which is the part that actually carries
ERD: central electrodes desynchronise while occipital ones do not, and that
contrast is exactly what is being flattened.

For the Riemannian pipeline the consequence is sharper still. A covariance
matrix computed from per-channel unit-variance signals is a CORRELATION matrix:
unit diagonal, no variance information anywhere in it. CSP, log-band-power and
tangent-space features are all variance-based, so all three are being fed a
representation with the amplitude stripped out.

The counter-argument, which is why this needs measuring rather than asserting:
per-window normalisation is a legitimate and common defence against amplitude
drift across sessions, and this dataset spans nine sessions over months. It may
be buying more in drift robustness than it costs in ERD. Both effects are real
and only the net matters.

So: same pipeline, same splits, same everything, zscore on versus off, on the
two representations that depend on variance most (tangent space, log-band-power)
and one that depends on it least (correlation-only), across 7 subjects.

Writes results/zscore.json.
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
from calmnet_msa import imu_valid_mask
from abstain import balanced_accuracy
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "zscore.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3


def load(sub, win, zscore):
    es = build_epochs(subject=sub, win=win, step=0.5, zscore=zscore)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, _ = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    f = lambda a: a[tr][ti]
    return {"Xf": f(es.X), "yf": f(es.y),
            "Xt": es.X[~tr], "yt": es.y[~tr], "st": es.session[~tr],
            "imu_t": es.imu_feats[~tr], "vt": valid[~tr]}


def feats(kind, Xf, Xt):
    if kind == "tangent_ea":
        return (FE.tangent(FE.euclidean_align(FE.covariances(Xf))),
                FE.tangent(FE.euclidean_align(FE.covariances(Xt))))
    if kind == "bandpower":
        return FE.log_var(FE.bandpass(Xf, 8, 30)), FE.log_var(FE.bandpass(Xt, 8, 30))
    if kind == "corr_only":
        # covariance rescaled to unit diagonal: the variance-free part of the
        # representation. If zscore is harmless, this should match tangent_ea;
        # if zscore is destroying amplitude, tangent_ea on UNNORMALISED data
        # should beat this and zscored tangent_ea should roughly equal it.
        def cc(X):
            C = FE.covariances(X)
            d = np.sqrt(np.einsum("nii->ni", C))[:, :, None]
            return FE.tangent(C / (d * d.transpose(0, 2, 1) + 1e-12))
        return cc(Xf), cc(Xt)
    raise ValueError(kind)


def main():
    wins = [float(w) for w in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2.0]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for win in wins:
        for kind in ("tangent_ea", "bandpower", "corr_only"):
            for zs in (True, False):
                t0, accs, r2s = time.time(), [], []
                for sub in SUBJECTS:
                    try:
                        d = load(sub, win, zs)
                    except Exception as e:
                        print("  [skip] %s: %s" % (sub, e), flush=True)
                        continue
                    Ff, Ft = feats(kind, d["Xf"], d["Xt"])
                    clf = make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=3000, C=0.1,
                                                           class_weight="balanced"))
                    clf.fit(Ff, d["yf"])
                    accs.append(balanced_accuracy(d["yt"], clf.predict(Ft)))
                    vt = d["vt"]
                    if vt.sum() > 40:
                        r2s.append(FE.invariance_r2_conditional(
                            Ft[vt], d["imu_t"][vt], d["yt"][vt], d["st"][vt]))
                if not accs:
                    continue
                key = "w%s|%s|z%d" % (win, kind, int(zs))
                out[key] = {"win": win, "feat": kind, "zscore": bool(zs),
                            "acc": float(np.mean(accs)),
                            "acc_sd": float(np.std(accs)),
                            "cond_r2": float(np.nanmean(r2s)),
                            "per_subject": {s: a for s, a in zip(SUBJECTS, accs)}}
                print("  %-12s zscore=%-5s acc %.3f+-%.3f  cond_r2 %+.3f  (%.0fs)"
                      % (kind, zs, out[key]["acc"], out[key]["acc_sd"],
                         out[key]["cond_r2"], time.time() - t0), flush=True)
                OUT.write_text(json.dumps(out, indent=1))
        print("", flush=True)
        for kind in ("tangent_ea", "bandpower", "corr_only"):
            a = out.get("w%s|%s|z1" % (win, kind), {}).get("acc")
            b = out.get("w%s|%s|z0" % (win, kind), {}).get("acc")
            if a is not None and b is not None:
                print("  %-12s  z-scored %.3f  ->  raw %.3f   delta %+.3f"
                      % (kind, a, b, b - a), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
