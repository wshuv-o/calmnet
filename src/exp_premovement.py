"""PATH B - Escape the movement confound via the PRE-MOVEMENT window.

The confound is that walk = the exoskeleton is moving. So decode BEFORE movement:
the readiness potential / Bereitschaftspotential (slow 0.1-3 Hz shift) and mu/beta ERD
build ~1.5-2 s before gait onset (Kornhuber & Deecke 1965; single-trial gait-intention
decoding, Jiang et al. 2015). The main pipeline band-passes 8-30 Hz and so DISCARDS the
MRCP entirely.

Test: decode a PREPARATION window (before each Stop->Walk onset, IMU still quiet) vs a
sustained-STOP control window (also quiet). If EEG separates them while the IMU cannot,
the signal is neural, not movement -- the confound is escaped. We contrast this with a
MOVEMENT window (during walk, IMU high) which is the confounded baseline.
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
from dataio import DATA_ROOT, _rexstate_segments, _imu_mag_series, list_sessions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
SF = 100.0
MOTOR = ("C3", "Cz", "C4", "FC1", "FC2", "FCz", "CP1", "CP2", "Cpz", "CPz", "C1", "C2")


def _load(subject, ses, task="training"):
    edf = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg" / f"{subject}_ses-{ses:02d}_task-{task}_eeg.edf"
    evt = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg" / f"{subject}_ses-{ses:02d}_task-{task}_acq-rexstate_events.tsv"
    if not (edf.exists() and evt.exists()):
        return None
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
    return raw


def _band(data, lo, hi, sf=SF):
    b, a = signal.butter(4, [lo / (sf / 2), hi / (sf / 2)], btype="band")
    return signal.filtfilt(b, a, data, axis=1)


def _feats(win_mrcp, win_erd):
    """win_*: (C, T). MRCP: mean + slope; ERD: log-power. Aggregate all + motor."""
    C, T = win_mrcp.shape
    t = np.linspace(-1, 1, T)
    slope = (win_mrcp * t).mean(1)                       # ~ linear trend (readiness ramp)
    mean = win_mrcp.mean(1)
    mu = np.log(_bp(win_erd, 8, 13) + 1e-12)
    beta = np.log(_bp(win_erd, 13, 30) + 1e-12)
    return mean, slope, mu, beta


def _bp(x, lo, hi, sf=SF):
    f, p = signal.welch(x, sf, nperseg=min(x.shape[1], int(sf)), axis=1)
    m = (f >= lo) & (f < hi)
    return p[:, m].sum(1)


def collect(subject):
    sess = list_sessions(subject)
    rows = {"prep": [], "move": [], "stop": []}   # feature rows
    imu = {"prep": [], "move": [], "stop": []}    # motion magnitude per window
    grp = {"prep": [], "move": [], "stop": []}    # session id
    ch_names = None
    for ses in sess:
        raw = _load(subject, ses)
        if raw is None:
            continue
        ch_names = raw.ch_names
        motor_idx = [i for i, c in enumerate(ch_names) if c in MOTOR] or list(range(len(ch_names)))
        data = raw.get_data()
        mrcp = _band(data, 0.3, 3.0)
        erd = _band(data, 8.0, 30.0)
        evt = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg" / f"{subject}_ses-{ses:02d}_task-training_acq-rexstate_events.tsv"
        seg = _rexstate_segments(evt)          # (onset, dur, label) Walk=1 Stop=0
        head = _imu_mag_series(subject, ses, "training", "head")

        def imu_mag(a, b):
            if head is None:
                return 0.0
            acc = head[0]; msf = head[2]
            s = acc[int(a / SF * msf):int(b / SF * msf)]
            return float(np.mean(np.abs(s - acc.mean()))) if len(s) else 0.0

        def add(kind, a0, b0):
            a, b = int(a0 * SF), int(b0 * SF)
            if a < 0 or b > data.shape[1] or b - a < int(1.0 * SF):
                return
            fm, fs, mu, be = _feats(mrcp[:, a:b], erd[:, a:b])
            # aggregate: mean over all + mean over motor channels
            row = np.concatenate([[fm.mean(), fs.mean(), mu.mean(), be.mean()],
                                  [fm[motor_idx].mean(), fs[motor_idx].mean(),
                                   mu[motor_idx].mean(), be[motor_idx].mean()]])
            rows[kind].append(row); imu[kind].append(imu_mag(a, b)); grp[kind].append(ses)

        # Stop->Walk transitions
        for i in range(len(seg) - 1):
            (o0, d0, l0), (o1, d1, l1) = seg[i], seg[i + 1]
            if l0 == 0 and l1 == 1 and d0 >= 3.0 and d1 >= 3.0:
                add("prep", o1 - 1.75, o1 - 0.25)     # BEFORE onset: preparation (IMU quiet)
                add("move", o1 + 1.5, o1 + 3.0)       # DURING walk: movement (IMU high)
        # sustained STOP controls (deep inside stop segments, away from edges)
        for (o, d, l) in seg:
            if l == 0 and d >= 6.0:
                add("stop", o + 2.0, o + 3.5)
    return rows, imu, grp


def decode(pos_rows, pos_grp, neg_rows, neg_grp):
    X = np.array(pos_rows + neg_rows)
    y = np.array([1] * len(pos_rows) + [0] * len(neg_rows))
    g = np.array(pos_grp + neg_grp)
    tr = g <= sorted(set(g))[2] if len(set(g)) >= 3 else np.ones(len(g), bool)
    te = ~tr
    if tr.sum() < 8 or te.sum() < 8 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
    return balanced_accuracy(y[te], clf.predict(X[te]))


def run(subject):
    rows, imu, grp = collect(subject)
    npre, nmov, nstop = len(rows["prep"]), len(rows["move"]), len(rows["stop"])
    if min(npre, nstop) < 10:
        print(f"  [{subject}] too few transitions (prep={npre}, stop={nstop})"); return None
    prep_acc = decode(rows["prep"], grp["prep"], rows["stop"], grp["stop"])
    move_acc = decode(rows["move"], grp["move"], rows["stop"], grp["stop"])
    imu_prep = float(np.mean(imu["prep"])); imu_move = float(np.mean(imu["move"])); imu_stop = float(np.mean(imu["stop"]))
    out = {"subject": subject, "n_prep": npre, "n_move": nmov, "n_stop": nstop,
           "prep_vs_stop_acc": prep_acc, "move_vs_stop_acc": move_acc,
           "imu_prep": imu_prep, "imu_move": imu_move, "imu_stop": imu_stop}
    print(f"  [{subject}] PREP-vs-stop={prep_acc:.3f} (IMU {imu_prep:.3f} vs stop {imu_stop:.3f})  |  "
          f"MOVE-vs-stop={move_acc:.3f} (IMU {imu_move:.3f})", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (pre-movement) ########", flush=True)
        r = run(sub)
        if r:
            res[sub] = r
            (RESULTS / "premovement.json").write_text(json.dumps(res, indent=2))
    if res:
        g = lambda k: float(np.nanmean([res[s][k] for s in res if res[s][k] == res[s][k]]))
        print("\n============ PRE-MOVEMENT (mean over subjects) ============")
        print(f"  PREPARATION vs stop : {g('prep_vs_stop_acc'):.3f}   (IMU prep {g('imu_prep'):.3f} ~ stop {g('imu_stop'):.3f} => movement-free)")
        print(f"  MOVEMENT   vs stop : {g('move_vs_stop_acc'):.3f}   (IMU move {g('imu_move'):.3f} => confounded)")
        print("  If PREP>chance with prep-IMU ~ stop-IMU, the decode is neural, not movement.")
