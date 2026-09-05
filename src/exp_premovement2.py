"""PATH B (rigorous) - is the pre-movement decode NEURAL or just movement leak?

Path B v1 showed preparation-vs-stop decodes ~0.71, but the prep window still moved
more than the stop control, so part of that is confound. Here we (1) use an EARLIER,
cleaner window that ends 1 s before onset, (2) attach the full 12-D head+exo IMU
descriptor to every window, and (3) decode prep-vs-stop three ways:
    raw            - features as-is
    IMU-residual   - regress the 12-D IMU out of the features first (movement-invariant)
    IMU-only       - decode from the IMU alone (the movement baseline)
If IMU-residual stays well above chance AND above IMU-only, there is a genuine
movement-invariant pre-movement neural signature of walk intent.
"""
import os, sys, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
import mne
mne.set_log_level("ERROR")
from dataio import DATA_ROOT, _rexstate_segments, _imu_mag_series, _imu_window_feats, list_sessions
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
SF = 100.0
MOTOR = ("C3", "Cz", "C4", "FC1", "FC2", "FCz", "CP1", "CP2", "CPz", "C1", "C2")


def _load(subject, ses):
    edf = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg" / f"{subject}_ses-{ses:02d}_task-training_eeg.edf"
    evt = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg" / f"{subject}_ses-{ses:02d}_task-training_acq-rexstate_events.tsv"
    if not (edf.exists() and evt.exists()):
        return None, None
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]
    raw.set_channel_types({c: "eog" for c in eog})
    raw.filter(0.1, 40.0, picks=["eeg", "eog"], method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    if eog:
        try:
            raw, _ = mne.preprocessing.regress_artifact(raw, picks="eeg", picks_artifact="eog", copy=True, verbose="ERROR")
        except Exception:
            pass
    raw.pick("eeg")
    return raw, _rexstate_segments(evt)


def _band(d, lo, hi):
    b, a = signal.butter(4, [lo / (SF / 2), hi / (SF / 2)], btype="band")
    return signal.filtfilt(b, a, d, axis=1)


def _bp(x, lo, hi):
    f, p = signal.welch(x, SF, nperseg=min(x.shape[1], int(SF)), axis=1)
    m = (f >= lo) & (f < hi)
    return p[:, m].sum(1)


def collect(subject):
    F, M, y, g = [], [], [], []
    for ses in list_sessions(subject):
        raw, seg = _load(subject, ses)
        if raw is None or not seg:
            continue
        ch = raw.ch_names
        mi = [i for i, c in enumerate(ch) if c in MOTOR] or list(range(len(ch)))
        data = raw.get_data()
        mrcp = _band(data, 0.3, 3.0)
        head = _imu_mag_series(subject, ses, "training", "head")
        exo = _imu_mag_series(subject, ses, "training", "exo")

        def add(label, a0, b0):
            a, b = int(a0 * SF), int(b0 * SF)
            if a < 0 or b > data.shape[1] or (b - a) < int(1.2 * SF):
                return
            wm, we = mrcp[:, a:b], data[:, a:b]
            t = np.linspace(-1, 1, b - a)
            fm, fs = wm.mean(1), (wm * t).mean(1)
            mu = np.log(_bp(_band(we, 8, 13), 8, 13) + 1e-12)
            be = np.log(_bp(_band(we, 13, 30), 13, 30) + 1e-12)
            row = np.array([fm.mean(), fs.mean(), mu.mean(), be.mean(),
                            fm[mi].mean(), fs[mi].mean(), mu[mi].mean(), be[mi].mean()])
            imu = _imu_window_feats(head, a, b, SF) + _imu_window_feats(exo, a, b, SF)  # 12-D
            F.append(row); M.append(imu); y.append(label); g.append(ses)

        for i in range(len(seg) - 1):
            (o0, d0, l0), (o1, d1, l1) = seg[i], seg[i + 1]
            if l0 == 0 and l1 == 1 and d0 >= 4.0 and d1 >= 3.0:
                add(1, o1 - 2.5, o1 - 1.0)          # EARLY preparation (ends 1 s before onset)
        for (o, d, l) in seg:
            if l == 0 and d >= 6.0:
                add(0, o + 2.0, o + 3.5)            # sustained stop control
    return (np.array(F), np.array(M), np.array(y), np.array(g)) if F else (None,) * 4


def decode(X, y, g):
    sess = sorted(set(g))
    if len(sess) < 4:
        return np.nan
    tr = g <= sess[2]; te = ~tr
    if tr.sum() < 8 or te.sum() < 8 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
    return balanced_accuracy(y[te], clf.predict(X[te])), tr, te


def run(subject):
    F, M, y, g = collect(subject)
    if F is None or (y == 1).sum() < 10 or (y == 0).sum() < 10:
        print(f"  [{subject}] too few windows"); return None
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    def bacc(X):
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        c.fit(X[tr], y[tr]); return balanced_accuracy(y[te], c.predict(X[te]))
    raw_acc = bacc(F)
    imu_acc = bacc(M)
    # movement-invariant: regress 12-D IMU out of the EEG features (train fit), then decode
    res = Ridge(alpha=1.0).fit(M[tr], F[tr])
    Fres = F - res.predict(M)
    inv_acc = bacc(Fres)
    imu_prep = float(M[y == 1][:, 1].mean()); imu_stop = float(M[y == 0][:, 1].mean())
    out = {"subject": subject, "n_prep": int((y == 1).sum()), "n_stop": int((y == 0).sum()),
           "raw": raw_acc, "imu_only": imu_acc, "invariant": inv_acc,
           "imu_prep": imu_prep, "imu_stop": imu_stop}
    print(f"  [{subject}] raw={raw_acc:.3f} | IMU-only={imu_acc:.3f} | "
          f"MOVEMENT-INVARIANT={inv_acc:.3f}  (prepIMU {imu_prep:.2f} vs stopIMU {imu_stop:.2f})", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (pre-movement, rigorous) ########", flush=True)
        r = run(sub)
        if r:
            res[sub] = r
            (RESULTS / "premovement_invariant.json").write_text(json.dumps(res, indent=2))
    if res:
        m = lambda k: float(np.nanmean([res[s][k] for s in res if res[s][k] == res[s][k]]))
        print("\n============ PRE-MOVEMENT, MOVEMENT-INVARIANT (mean) ============")
        print(f"  raw prep-vs-stop          : {m('raw'):.3f}")
        print(f"  IMU-only (movement)       : {m('imu_only'):.3f}")
        print(f"  MOVEMENT-INVARIANT decode : {m('invariant'):.3f}   <- the honest neural pre-movement signal")
        print(f"  prep IMU {m('imu_prep'):.2f} vs stop IMU {m('imu_stop'):.2f}")
        print("  invariant >> 0.50 and > (a chance-level IMU-residual) => genuine pre-movement neural signature.")
