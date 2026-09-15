"""How much movement contamination can the conditional probe actually detect?

The probe reports -0.11 on the tangent+EA representation: no movement
information beyond what the Walk/Stop label already implies. That number is
being used to argue the representation is clean, and as it stands it cannot
carry that argument, because a null is uninterpretable until the instrument is
shown to be capable of a positive. A probe that detects nothing would return the
same -0.11.

So: inject KNOWN artefact at KNOWN amplitude into real EEG and find the level at
which the probe starts to see it.

  artefact model   the real motion reference, spatially mixed into the EEG with
                   a fixed per-subject random gain matrix. Real motion artefact
                   reaches different electrodes with different gains, so a
                   spatially structured injection is closer to the thing being
                   detected than adding the same waveform to every channel.

  amplitude        swept as a fraction of total signal power,
                   var(alpha*artefact) / var(X + alpha*artefact), from 0 to 50%.

  measured         conditional R^2 (the claim), raw R^2 (the confounded metric,
                   for contrast), and balanced accuracy -- because contamination
                   should also make the task EASIER, which is the whole reason
                   it is dangerous.

What the output means. If conditional R^2 lifts off the floor at, say, 2%
injected power, then the reported -0.11 bounds real contamination below ~2% and
the claim is quantitative. If it only responds at 30%, the -0.11 means very
little and that needs saying before anyone writes it down.

Contamination is applied to fit and test alike: a decoder trained on clean data
and tested on contaminated data would measure transfer failure, not the
artefact-exploitation pathway that is under test here.

Writes results/sensitivity.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json, sys, time, zlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from motion_ts import build_motion_ts
from splits import grouped_split
from calmnet_msa import imu_valid_mask
from abstain import balanced_accuracy
from exp_temporal import _tangent
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "sensitivity.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3
# fraction of TOTAL power that is injected artefact
LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def load(sub, win=2.0):
    es = build_epochs(subject=sub, win=win, step=0.5)
    M, _ = build_motion_ts(sub, win=win, step=0.5)
    if len(M) != len(es):
        raise RuntimeError("motion misaligned: %d vs %d" % (len(M), len(es)))
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, _ = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    f = lambda a: a[tr][ti]
    return {"Xf": f(es.X), "Mf": f(M), "yf": f(es.y),
            "Xt": es.X[~tr], "Mt": M[~tr], "yt": es.y[~tr],
            "st": es.session[~tr], "imu_t": es.imu_feats[~tr], "vt": valid[~tr]}


def artefact(M, n_chan, seed=0):
    """Spatially mix the motion reference into n_chan EEG channels.

    A fixed random gain per (channel, reference) pair, drawn once per subject.
    Real motion artefact is spatially structured -- it does not arrive at every
    electrode identically -- so a per-channel mixing is a closer stand-in than a
    common additive waveform, and it is harder for a spatial filter (which is
    what the tangent-space pipeline builds) to reject than a rank-1 injection.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_chan, M.shape[1])).astype(np.float32)
    A = np.einsum("cr,nrt->nct", W, M.astype(np.float32))
    return A / (A.std() + 1e-12)          # unit variance, scaled later


def inject(X, A, frac):
    """Add A to X so that A accounts for `frac` of total power."""
    if frac <= 0:
        return X
    vx, va = float(X.var()), float(A.var())
    alpha = np.sqrt(frac / max(1.0 - frac, 1e-9) * vx / max(va, 1e-12))
    return (X + alpha * A).astype(np.float32)


def main():
    subs = sys.argv[1:] or SUBJECTS
    D = {}
    for sub in subs:
        try:
            D[sub] = load(sub)
            d = D[sub]
            # crc32, not hash(): Python randomises string hashing per process,
            # so hash() would draw a different artefact pattern on every run and
            # the curve would not be reproducible.
            sd = zlib.crc32(sub.encode())
            d["Af"] = artefact(d["Mf"], d["Xf"].shape[1], seed=sd)
            d["At"] = artefact(d["Mt"], d["Xt"].shape[1], seed=sd)
        except Exception as e:
            print("  [skip] %s: %s: %s" % (sub, type(e).__name__, e), flush=True)
    print("%d subjects\n" % len(D), flush=True)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("%-10s %12s %12s %10s %10s" %
          ("inject%", "cond_R2", "raw_R2", "acc", "secs"), flush=True)
    print("-" * 58, flush=True)
    for frac in LEVELS:
        t0 = time.time()
        C, R, A_ = [], [], []
        for sub, d in D.items():
            Xf = inject(d["Xf"], d["Af"], frac)
            Xt = inject(d["Xt"], d["At"], frac)
            Ff, Ft = _tangent(Xf), _tangent(Xt)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=0.1,
                                                   class_weight="balanced"))
            clf.fit(Ff, d["yf"])
            A_.append(balanced_accuracy(d["yt"], clf.predict(Ft)))
            vt = d["vt"]
            if vt.sum() > 40:
                C.append(FE.invariance_r2_conditional(Ft[vt], d["imu_t"][vt],
                                                      d["yt"][vt], d["st"][vt]))
                R.append(FE.invariance_r2_cv(Ft[vt], d["imu_t"][vt], d["st"][vt]))
        rec = {"inject_frac": frac,
               "cond_r2": float(np.nanmean(C)), "cond_sd": float(np.nanstd(C)),
               "raw_r2": float(np.nanmean(R)),
               "acc": float(np.nanmean(A_)), "n_sub": len(D)}
        out["f%.4f" % frac] = rec
        print("%-10.2f %+12.3f %+12.3f %10.3f %10.0f"
              % (frac * 100, rec["cond_r2"], rec["raw_r2"], rec["acc"],
                 time.time() - t0), flush=True)
        OUT.write_text(json.dumps(out, indent=1))

    # detection threshold: lowest injected level whose conditional R^2 clears the
    # clean-data baseline by more than the between-subject spread at that level
    base = out["f0.0000"]["cond_r2"]
    thr = None
    for frac in LEVELS[1:]:
        r = out["f%.4f" % frac]
        if r["cond_r2"] > base + max(r["cond_sd"], 0.02):
            thr = frac
            break
    print("\nclean-data conditional R2 : %+.3f" % base, flush=True)
    print("detection threshold       : %s"
          % ("%.2f%% injected artefact power" % (thr * 100) if thr
             else "NOT DETECTED at any level tested -- probe is insensitive"),
          flush=True)
    if thr:
        print("=> the reported -0.11 bounds real contamination below ~%.2f%% of "
              "signal power." % (thr * 100), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
