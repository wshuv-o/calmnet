"""IDEA 1/4 - IMU-referenced artifact cleaning, THEN decode.

If the walk/stop confound is motion ARTIFACT (electrode/cable sway, neck EMG driven by
head motion), the IMU is a reference for it. We regress lagged head+exo accel/gyro out of
every EEG channel (adaptive noise cancellation / iCanClean-style), then decode band-power
walk/stop on the CLEANED signal. Comparison:
  raw band-power decode   -> confounded baseline (~0.84)
  cleaned band-power decode + how much IMU still predicts the cleaned features
If cleaning drops the decode toward chance AND removes IMU-predictability, the signal was
mostly artifact. If a decode SURVIVES cleaning while IMU can no longer predict it, that is
recovered movement-invariant neural signal.
"""
import os, sys, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
import mne, pandas as pd, re
mne.set_log_level("ERROR")
from dataio import DATA_ROOT, _rexstate_segments, list_sessions
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
SF = 100.0
LAGS = range(-4, 5)            # +/- 40 ms reference lags


def imu_ref(subject, ses, n_eeg):
    """Return (n_eeg, R) matrix of head+exo accel/gyro components, resampled + lagged."""
    mdir = DATA_ROOT / subject / f"ses-{ses:02d}" / "motion"
    cols = []
    for tracksys in ("head", "exo"):
        tsv = mdir / f"{subject}_ses-{ses:02d}_task-training_tracksys-{tracksys}_motion.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")
        c = [x for x in df.columns if re.search(r"acc|gyro", str(x), re.I)]
        if not c:
            continue
        v = df[c].astype(float).to_numpy()
        # resample to EEG length
        idx = np.linspace(0, len(v) - 1, n_eeg)
        v = np.stack([np.interp(idx, np.arange(len(v)), v[:, j]) for j in range(v.shape[1])], 1)
        v = (v - v.mean(0)) / (v.std(0) + 1e-9)
        cols.append(v)
    if not cols:
        return None
    base = np.concatenate(cols, 1)                       # (n_eeg, k)
    R = [np.roll(base, l, axis=0) for l in LAGS]         # lagged copies
    return np.concatenate(R, 1)                          # (n_eeg, k*len(LAGS))


def bp_feats(data):
    """log band-power (mu, beta) per channel over a window -> feature vector."""
    def bp(x, lo, hi):
        f, p = signal.welch(x, SF, nperseg=min(x.shape[1], int(SF)), axis=1)
        m = (f >= lo) & (f < hi)
        return np.log(p[:, m].sum(1) + 1e-12)
    return np.concatenate([bp(data, 8, 13), bp(data, 13, 30)])


def build(subject, clean):
    Xf, y, M, g = [], [], [], []
    for ses in list_sessions(subject):
        base = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg"
        edf = base / f"{subject}_ses-{ses:02d}_task-training_eeg.edf"
        evt = base / f"{subject}_ses-{ses:02d}_task-training_acq-rexstate_events.tsv"
        if not (edf.exists() and evt.exists()):
            continue
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        eog = [c for c in raw.ch_names if "EOG" in c.upper()]
        raw.set_channel_types({c: "eog" for c in eog})
        raw.filter(1.0, 45.0, picks=["eeg", "eog"], method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
        if eog:
            try:
                raw, _ = mne.preprocessing.regress_artifact(raw, picks="eeg", picks_artifact="eog", copy=True, verbose="ERROR")
            except Exception:
                pass
        raw.pick("eeg")
        data = raw.get_data()                            # (C, N)
        N = data.shape[1]
        if clean:
            R = imu_ref(subject, ses, N)
            if R is not None:
                # regress lagged IMU out of each channel (adaptive noise cancel)
                beta = np.linalg.lstsq(R, data.T, rcond=None)[0]     # (kL, C)
                data = (data.T - R @ beta).T                         # cleaned
        # motion magnitude for invariance check
        Rmag = imu_ref(subject, ses, N)
        mot = np.abs(Rmag[:, :Rmag.shape[1] // len(LAGS)]).mean(1) if Rmag is not None else np.zeros(N)
        seg = _rexstate_segments(evt)
        T = int(2.0 * SF)
        for (o, d, l) in seg:
            a0 = int(o * SF) + int(0.5 * SF)
            for a in range(a0, min(int((o + d) * SF), N) - T + 1, int(0.5 * SF)):
                w = data[:, a:a + T]
                w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-7)
                Xf.append(bp_feats(w)); y.append(l); g.append(ses)
                M.append(float(mot[a:a + T].mean()))
    return np.array(Xf), np.array(y), np.array(M), np.array(g)


def decode(X, y, g):
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    if tr.sum() < 10 or te.sum() < 10 or len(set(y[te])) < 2:
        return np.nan, None, None
    c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    c.fit(X[tr], y[tr])
    return balanced_accuracy(y[te], c.predict(X[te])), tr, te


def run(subject):
    Xr, y, M, g = build(subject, clean=False)
    Xc, y2, M2, g2 = build(subject, clean=True)
    if len(Xr) < 30:
        print(f"  [{subject}] too few"); return None
    raw_acc, tr, te = decode(Xr, y, g)
    cln_acc, _, _ = decode(Xc, y, g)
    # how much does IMU predict the features, raw vs cleaned?
    def mot_r2(X):
        rr = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); rr.fit(X[tr], M[tr])
        return r2_score(M[te], rr.predict(X[te]))
    out = {"subject": subject, "raw_acc": raw_acc, "cleaned_acc": cln_acc,
           "raw_motion_r2": mot_r2(Xr), "cleaned_motion_r2": mot_r2(Xc)}
    print(f"  [{subject}] raw={raw_acc:.3f} (IMU->feat R2={out['raw_motion_r2']:+.2f}) | "
          f"CLEANED={cln_acc:.3f} (IMU->feat R2={out['cleaned_motion_r2']:+.2f})", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (IMU-referenced cleaning) ########", flush=True)
        r = run(sub)
        if r:
            res[sub] = r
            (RESULTS / "clean_imu.json").write_text(json.dumps(res, indent=2))
    if res:
        m = lambda k: float(np.nanmean([res[s][k] for s in res if res[s][k] == res[s][k]]))
        print("\n============ IMU-REFERENCED CLEANING (mean) ============")
        print(f"  raw band-power decode     : {m('raw_acc'):.3f}  (IMU->feat R2 {m('raw_motion_r2'):+.2f})")
        print(f"  CLEANED band-power decode : {m('cleaned_acc'):.3f}  (IMU->feat R2 {m('cleaned_motion_r2'):+.2f})")
        print("  if cleaned decode stays high with IMU->feat R2 ~ 0 => neural signal recovered;")
        print("  if cleaned decode falls to ~chance => the decode was motion artifact.")
