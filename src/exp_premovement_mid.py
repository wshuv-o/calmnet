"""PATH B capstone (GPU) - nonlinear adversarial test of a movement-invariant
pre-movement neural signal.

Linear IMU-removal drove the pre-movement decode to chance; this is the definitive
nonlinear version. We build raw 2 s pre-movement EEG windows (0.3-30 Hz: MRCP + ERD)
labelled prep(=1, about to walk) vs sustained-stop(=0), attach the 12-D head+exo IMU,
and train the CALMNetMID adversarial net two ways:
    plain      (lam_adv=0)  - movement-confounded baseline
    invariant  (lam_adv=1)  - movement disentangled
If invariant accuracy > chance AND intent->IMU R^2 < 0, a genuine movement-invariant
pre-movement signature exists. Otherwise the confound is fundamental to this label.
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
from mid import train_mid, predict_mid, encode_all, motion_probe_r2
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
SF = 100.0
WIN = 2.0            # seconds -> T=200 (matches EEGNet config)


def _load(subject, ses):
    base = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg"
    edf = base / f"{subject}_ses-{ses:02d}_task-training_eeg.edf"
    evt = base / f"{subject}_ses-{ses:02d}_task-training_acq-rexstate_events.tsv"
    if not (edf.exists() and evt.exists()):
        return None, None
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]
    raw.set_channel_types({c: "eog" for c in eog})
    raw.filter(0.3, 30.0, picks=["eeg", "eog"], method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    if eog:
        try:
            raw, _ = mne.preprocessing.regress_artifact(raw, picks="eeg", picks_artifact="eog", copy=True, verbose="ERROR")
        except Exception:
            pass
    raw.pick("eeg")
    return raw, _rexstate_segments(evt)


def collect(subject):
    X, y, M, g = [], [], [], []
    T = int(WIN * SF)
    for ses in list_sessions(subject):
        raw, seg = _load(subject, ses)
        if raw is None or not seg:
            continue
        data = raw.get_data().astype(np.float32)
        head = _imu_mag_series(subject, ses, "training", "head")
        exo = _imu_mag_series(subject, ses, "training", "exo")

        def add(label, a0):
            a = int(a0 * SF); b = a + T
            if a < 0 or b > data.shape[1]:
                return
            w = data[:, a:b]
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-7)   # z per chan
            imu = _imu_window_feats(head, a, b, SF) + _imu_window_feats(exo, a, b, SF)
            X.append(w.astype(np.float32)); y.append(label); M.append(imu); g.append(ses)

        for i in range(len(seg) - 1):
            (o0, d0, l0), (o1, d1, l1) = seg[i], seg[i + 1]
            if l0 == 0 and l1 == 1 and d0 >= 4.0 and d1 >= 2.0:
                for st in (o1 - 3.0, o1 - 2.5):          # sliding prep windows, both end >=0.5s pre-onset
                    add(1, st)
        for (o, d, l) in seg:
            if l == 0 and d >= 7.0:
                for st in (o + 2.0, o + 4.0):
                    add(0, st)
    if not X:
        return (None,) * 4
    return np.stack(X), np.array(y), np.array(M, np.float32), np.array(g)


def run(subject):
    X, y, M, g = collect(subject)
    if X is None or (y == 1).sum() < 12 or (y == 0).sum() < 12:
        print(f"  [{subject}] too few windows"); return None
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    if tr.sum() < 12 or te.sum() < 12 or len(set(y[te])) < 2:
        print(f"  [{subject}] bad split"); return None
    # inner val split from train (by session)
    vtr = g < sess[2]; vv = g == sess[2]
    if len(set(y[vv])) < 2:
        vtr, vv = tr, tr
    res = {"subject": subject, "n_prep": int((y == 1).sum()), "n_stop": int((y == 0).sum())}
    for tag, lam in (("plain", 0.0), ("invariant", 1.0)):
        m = train_mid(X[vtr], y[vtr], M[vtr], X[vv], y[vv], epochs=120,
                      lam_adv=lam, lam_dec=lam, backbone="eegnet", seed=0)
        acc = balanced_accuracy(y[te], predict_mid(m, X[te])[1].argmax(1))
        zi_tr, _ = encode_all(m, X[tr]); zi_te, _ = encode_all(m, X[te])
        r2 = motion_probe_r2(zi_tr, M[tr], zi_te, M[te], nonlinear=True)
        res[f"{tag}_acc"] = acc; res[f"{tag}_r2"] = r2
        print(f"  [{subject}] {tag:9s} acc={acc:.3f}  intent->IMU R2={r2:+.3f}", flush=True)
    # IMU-only reference
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    c.fit(M[tr], y[tr]); res["imu_only"] = balanced_accuracy(y[te], c.predict(M[te]))
    print(f"  [{subject}] IMU-only={res['imu_only']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    out = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (pre-movement MID, GPU) ########", flush=True)
        r = run(sub)
        if r:
            out[sub] = r
            (RESULTS / "premovement_mid.json").write_text(json.dumps(out, indent=2))
    if out:
        m = lambda k: float(np.nanmean([out[s][k] for s in out if k in out[s] and out[s][k] == out[s][k]]))
        print("\n============ PRE-MOVEMENT MID (mean over subjects) ============")
        print(f"  plain (confounded)        acc={m('plain_acc'):.3f}  R2={m('plain_r2'):+.3f}")
        print(f"  MOVEMENT-INVARIANT (MID)  acc={m('invariant_acc'):.3f}  R2={m('invariant_r2'):+.3f}")
        print(f"  IMU-only (movement)       acc={m('imu_only'):.3f}")
        print("  invariant acc >> 0.50 with R2<0 => real movement-invariant pre-movement signal.")
