"""Pre-movement window analysis: decode the walk INTENT at the Stop->Walk transition
onset (movement still low) vs the stable Walk state (full movement). If the IMU-only
baseline collapses at the transition window while the EEG decode survives, the
walk/stop signal is genuinely neural, not just movement.

Simple, data-efficient classifiers (logistic regression on log-band-power / IMU
features), grouped CV within subject, pooled over sessions.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataio import DATA_ROOT, _load_raw, _imu_mag_series, _imu_window_feats, list_sessions

WIN, STEP = 1.0, 0.25            # 1 s windows
PREP_SPAN = 1.5                  # only the first 1.5 s after a transition onset
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]


def _segments(ev_tsv):
    E = pd.read_csv(ev_tsv, sep="\t")
    return [(float(r.onset), float(r.duration), str(r.trial_type)) for r in E.itertuples()]


def _bandpower(win):                                   # log-variance per channel
    return np.log(win.var(axis=1) + 1e-8)


def _collect(subject, scheme):
    """scheme='prep' : walk = first 1.5s of x8 (Stop->Walk), stop = stable x0.
       scheme='stable': walk = stable x81,               stop = stable x0."""
    Xs, Ms, ys, gs = [], [], [], []
    gid = 0
    for ses in list_sessions(subject):
        d = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg"
        edf = d / f"{subject}_ses-{ses:02d}_task-training_eeg.edf"
        ev = d / f"{subject}_ses-{ses:02d}_task-training_acq-rexstate_events.tsv"
        if not (edf.exists() and ev.exists()):
            continue
        raw = _load_raw(edf, 8.0, 30.0); sf = raw.info["sfreq"]; data = raw.get_data()
        head = _imu_mag_series(subject, ses, "training", "head")
        exo = _imu_mag_series(subject, ses, "training", "exo")
        wl, ws = int(WIN * sf), int(STEP * sf)
        for onset, dur, typ in _segments(ev):
            if scheme == "prep":
                if typ == "x8":   lab, s0, s1 = 1, onset, onset + min(PREP_SPAN, dur)
                elif typ == "x0" and dur > 3: lab, s0, s1 = 0, onset + 0.5, onset + dur
                else: continue
            else:  # stable
                if typ == "x81":  lab, s0, s1 = 1, onset + 0.5, onset + dur
                elif typ == "x0" and dur > 3: lab, s0, s1 = 0, onset + 0.5, onset + dur
                else: continue
            a0, a1 = int(s0 * sf), int(s1 * sf)
            for a in range(a0, a1 - wl + 1, ws):
                Xs.append(_bandpower(data[:, a:a + wl])); ys.append(lab); gs.append(gid)
                Ms.append(_imu_window_feats(head, a, a + wl, sf) + _imu_window_feats(exo, a, a + wl, sf))
            gid += 1
    return np.array(Xs), np.array(Ms), np.array(ys), np.array(gs)


def _cv_bacc(X, y, g):
    from sklearn.metrics import balanced_accuracy_score
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return float("nan")
    skf = StratifiedGroupKFold(n_splits=min(5, len(np.unique(g))))
    accs = []
    for tr, te in skf.split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))


if __name__ == "__main__":
    print(f"{'subj':8}| {'STABLE (full walk)':^22}| {'PREP (walk onset)':^22}")
    print(f"{'':8}| {'IMU':>6} {'EEG':>6} {'nW/nS':>8}| {'IMU':>6} {'EEG':>6} {'nW/nS':>8}")
    rows = {}
    for sub in SUBJECTS:
        r = {}
        for scheme in ("stable", "prep"):
            X, M, y, g = _collect(sub, scheme)
            nW, nS = int((y == 1).sum()), int((y == 0).sum())
            r[scheme] = (_cv_bacc(M, y, g), _cv_bacc(X, y, g), nW, nS)
        rows[sub] = r
        s, p = r["stable"], r["prep"]
        print(f"{sub:8}| {s[0]:6.3f} {s[1]:6.3f} {s[2]:4d}/{s[3]:<4d}| {p[0]:6.3f} {p[1]:6.3f} {p[2]:4d}/{p[3]:<4d}", flush=True)
    print("-" * 62)
    def mean(scheme, i): return np.nanmean([rows[s][scheme][i] for s in rows])
    print(f"{'MEAN':8}| {mean('stable',0):6.3f} {mean('stable',1):6.3f} {'':8}| "
          f"{mean('prep',0):6.3f} {mean('prep',1):6.3f}")
    print("\nWin if PREP: IMU baseline drops (less movement) while EEG stays above chance/comparable")
    print("=> the walk/stop signal is genuinely neural, decodable before movement builds up.")
