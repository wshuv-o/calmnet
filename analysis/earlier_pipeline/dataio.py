"""Cohort A (ds007788) windows for the earlier pipeline: 8-30 Hz zero-phase FIR, EOG regression."""
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import mne  # noqa: E402

mne.set_log_level("ERROR")

ROOT = Path(os.environ.get("CALMNET_DATA", Path(__file__).resolve().parents[2] / "datasets"))
DATA_ROOT = ROOT / "ds007788"
CACHE_DIR = ROOT / "cache"
MI_TASKS = ("training",)
STATE_TO_LABEL = {"x81": 1, "x0": 0}             # walk 1, stop 0


@dataclass
class EpochSet:
    X: np.ndarray
    y: np.ndarray
    session: np.ndarray
    day: np.ndarray
    task: np.ndarray
    motion: np.ndarray
    segment: np.ndarray = None
    imu_feats: np.ndarray = None
    ch_names: list = field(default_factory=list)
    sfreq: float = 100.0

    def __len__(self):
        return len(self.y)


def list_sessions(subject="sub-01"):
    return sorted(int(p.name.split("-")[1]) for p in (DATA_ROOT / subject).glob("ses-*")
                  if p.is_dir())


def _session_date(subject, ses):
    for js in (DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg").glob("*_eeg.json"):
        m = re.search(r"Recorded on: (\d{4}-\d{2}-\d{2})",
                      json.loads(js.read_text()).get("TaskDescription", ""))
        if m:
            return date(*map(int, m.group(1).split("-")))
    return None


def session_days(subject="sub-01"):
    sess = list_sessions(subject)
    dates = {s: _session_date(subject, s) for s in sess}
    ref = dates[sess[0]]
    return {s: (dates[s] - ref).days if dates[s] else -1 for s in sess}


def _load_raw(edf_path, l_freq, h_freq):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]
    raw.set_channel_types({c: "eog" for c in eog})
    raw.filter(l_freq, h_freq, picks=["eeg", "eog"], method="fir", phase="zero",
               fir_design="firwin", verbose="ERROR")
    if eog:
        try:
            raw, _ = mne.preprocessing.regress_artifact(raw, picks="eeg", picks_artifact="eog",
                                                        copy=True, verbose="ERROR")
        except Exception:
            pass
    raw.pick("eeg")
    return raw


def _segments(events_tsv):
    df = pd.read_csv(events_tsv, sep="\t")
    return [(float(r["onset"]), float(r["duration"]), STATE_TO_LABEL[str(r["trial_type"])])
            for _, r in df.iterrows()
            if str(r["trial_type"]) in STATE_TO_LABEL and float(r["duration"]) > 0]


def _imu(subject, ses, task, tracksys):
    mdir = DATA_ROOT / subject / f"ses-{ses:02d}" / "motion"
    stem = f"{subject}_ses-{ses:02d}_task-{task}_tracksys-{tracksys}_motion"
    tsv, js = mdir / (stem + ".tsv"), mdir / (stem + ".json")
    if not tsv.exists():
        return None
    try:
        data = pd.read_csv(tsv, sep="\t")

        def mag(pattern):
            cols = [c for c in data.columns if re.search(pattern, str(c), re.I)]
            if not cols:
                return None
            v = data[cols].astype(float).to_numpy()
            v = v - v.mean(axis=0, keepdims=True)
            return np.sqrt((v ** 2).sum(axis=1))
        acc, gyr = mag(r"acc"), mag(r"gyro")
        if acc is None:
            return None
        if gyr is None:
            gyr = np.zeros_like(acc)
        sf = float(json.loads(js.read_text()).get("SamplingFrequency", 100.0)) \
            if js.exists() else 100.0
        return acc, gyr, sf
    except Exception:
        return None


def _imu_feats(imu, a, b, sf):
    if imu is None:
        return [0.0] * 6
    acc, gyr, msf = imu
    ma, mb = int(a / sf * msf), int(b / sf * msf)
    out = []
    for series in (acc, gyr):
        s = series[ma:mb]
        out += [float(np.mean(s)), float(np.std(s)), float(np.max(s))] if len(s) else [0.0] * 3
    return out


def build_epochs(subject="sub-01", sessions=None, tasks=MI_TASKS, win=2.0, step=0.5,
                 margin=0.5, l_freq=8.0, h_freq=30.0, zscore=True, use_cache=True):
    """Windows of the stable walk / stop segments; zscore=True normalises each channel of each window."""
    sessions = sessions or list_sessions(subject)
    days = session_days(subject)
    tag = (f"{subject}_s{'-'.join(map(str, sessions))}_{'-'.join(tasks)}_w{win}_st{step}"
           f"_{l_freq}-{h_freq}_z{int(zscore)}_v2imu")
    cache = CACHE_DIR / f"{tag}.npz"
    if use_cache and cache.exists():
        with open(cache, "rb") as fh, np.load(fh, allow_pickle=True) as d:
            m = {k: d[k] for k in d.files}
        return EpochSet(m["X"], m["y"], m["session"], m["day"], m["task"], m["motion"],
                        m["segment"], m["imu_feats"], list(m["ch_names"]), float(m["sfreq"]))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X, y, ses_l, day_l, task_l, mot, sid, feats = [], [], [], [], [], [], [], []
    ch_names, sfreq, seg_base = None, 100.0, 0
    for ses in sessions:
        eeg_dir = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg"
        for task in tasks:
            edf = eeg_dir / f"{subject}_ses-{ses:02d}_task-{task}_eeg.edf"
            evt = eeg_dir / f"{subject}_ses-{ses:02d}_task-{task}_acq-rexstate_events.tsv"
            if not (edf.exists() and evt.exists()):
                continue
            seg = _segments(evt)
            if not seg:
                continue
            raw = _load_raw(edf, l_freq, h_freq)
            ch_names, sfreq = raw.ch_names, raw.info["sfreq"]
            data = raw.get_data()
            sf = round(raw.info["sfreq"])
            wlen, wstep, mrg = round(win * sf), round(step * sf), round(margin * sf)
            head, exo = _imu(subject, ses, task, "head"), _imu(subject, ses, task, "exo")
            n0 = len(y)
            for gi, (onset, dur, lab) in enumerate(seg):
                s0 = int(onset * sf) + mrg
                s1 = min(int((onset + dur) * sf), data.shape[1])
                for a in range(s0, s1 - wlen + 1, wstep):
                    w = data[:, a:a + wlen]
                    if w.shape[1] != wlen:
                        continue
                    hf, ef = _imu_feats(head, a, a + wlen, sf), _imu_feats(exo, a, a + wlen, sf)
                    X.append(w); y.append(lab); sid.append(seg_base + gi)
                    feats.append(hf + ef); mot.append(hf[1])
            seg_base += len(seg)
            n = len(y) - n0
            ses_l += [ses] * n; day_l += [days.get(ses, -1)] * n; task_l += [task] * n

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if zscore and len(X):
        X = ((X - X.mean(axis=2, keepdims=True))
             / (X.std(axis=2, keepdims=True) + 1e-7)).astype(np.float32)
    es = EpochSet(X, y, np.asarray(ses_l), np.asarray(day_l), np.asarray(task_l, dtype=object),
                  np.asarray(mot, dtype=np.float32), np.asarray(sid, dtype=np.int64),
                  np.asarray(feats, dtype=np.float32), ch_names or [], sfreq)
    np.savez_compressed(cache, X=es.X, y=es.y, session=es.session, day=es.day, task=es.task,
                        motion=es.motion, segment=es.segment, imu_feats=es.imu_feats,
                        ch_names=np.array(es.ch_names, dtype=object), sfreq=es.sfreq)
    return es
