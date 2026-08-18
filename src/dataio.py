"""BIDS/EDF loading, preprocessing, and Walk/Stop epoching for NeuroRex (ds007788).

Labels come from the `rexstate` events (exoskeleton feedback state):
    x81 -> Walk (1),  x0 -> Idle/Stop (0),  x5/x8 -> transitions (excluded).

We slice stable Walk/Stop segments into overlapping fixed-length windows,
band-pass to the mu/beta MI band, and z-score per channel per epoch.
A head-IMU motion score is attached to each epoch for the abstention gate.
"""
from __future__ import annotations
import json
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

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "ds007788"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# The `training` task is the clean, scripted open-loop MI calibration block: its
# rexstate follows a fixed schedule the subject imagines in sync with, giving an
# identical, uncorrupted Walk/Stop structure every session (99 walk / 267 stop).
# walk6min/stop6min rexstate reflects closed-loop OUTCOME, not intent, so their
# labels are confounded -- kept available but not used by default.
MI_TASKS = ("training",)
ALL_MI_TASKS = ("training", "walk6min", "stop6min")

STATE_TO_LABEL = {"x81": 1, "x0": 0}  # Walk=1, Stop=0
LABEL_NAMES = {0: "Stop", 1: "Walk"}


@dataclass
class EpochSet:
    """A stack of epochs with aligned metadata arrays."""
    X: np.ndarray          # (N, C, T) float32
    y: np.ndarray          # (N,) int
    session: np.ndarray    # (N,) int session number
    day: np.ndarray        # (N,) int days since first session
    task: np.ndarray       # (N,) object task name
    motion: np.ndarray     # (N,) float head-IMU motion score (higher = more movement)
    segment: np.ndarray = None   # (N,) int group id per contiguous state segment (no-leak splits)
    imu_feats: np.ndarray = None # (N, 12) head+exo accel/gyro magnitude mean/std/max
    ch_names: list[str] = field(default_factory=list)
    sfreq: float = 100.0

    IMU_FEAT_NAMES = ("head_acc_mean", "head_acc_std", "head_acc_max",
                      "head_gyr_mean", "head_gyr_std", "head_gyr_max",
                      "exo_acc_mean", "exo_acc_std", "exo_acc_max",
                      "exo_gyr_mean", "exo_gyr_std", "exo_gyr_max")

    def __len__(self):
        return len(self.y)

    def subset(self, mask) -> "EpochSet":
        return EpochSet(
            self.X[mask], self.y[mask], self.session[mask], self.day[mask],
            self.task[mask], self.motion[mask],
            None if self.segment is None else self.segment[mask],
            None if self.imu_feats is None else self.imu_feats[mask],
            self.ch_names, self.sfreq,
        )

    def by_sessions(self, sessions) -> "EpochSet":
        sessions = set(int(s) for s in sessions)
        mask = np.array([s in sessions for s in self.session])
        return self.subset(mask)


def list_sessions(subject: str = "sub-01") -> list[int]:
    sub_dir = DATA_ROOT / subject
    return sorted(int(p.name.split("-")[1]) for p in sub_dir.glob("ses-*") if p.is_dir())


def _session_date(subject: str, ses: int) -> date | None:
    """Read the recording date from a task json sidecar."""
    ses_dir = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg"
    for js in ses_dir.glob("*_eeg.json"):
        meta = json.loads(js.read_text())
        m = re.search(r"Recorded on: (\d{4}-\d{2}-\d{2})", meta.get("TaskDescription", ""))
        if m:
            y, mo, d = map(int, m.group(1).split("-"))
            return date(y, mo, d)
    return None


def session_days(subject: str = "sub-01") -> dict[int, int]:
    """Map session number -> days elapsed since the first session."""
    sess = list_sessions(subject)
    dates = {s: _session_date(subject, s) for s in sess}
    ref = dates[sess[0]]
    return {s: (dates[s] - ref).days if dates[s] else -1 for s in sess}


def _load_raw(edf_path: Path, l_freq: float, h_freq: float,
              eog_regress: bool = True) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]
    raw.set_channel_types({c: "eog" for c in eog})
    # band-pass EEG + EOG to the MI band, then linearly regress EOG out of EEG
    raw.filter(l_freq, h_freq, picks=["eeg", "eog"], method="fir", phase="zero",
               fir_design="firwin", verbose="ERROR")
    if eog_regress and eog:
        try:
            raw, _ = mne.preprocessing.regress_artifact(
                raw, picks="eeg", picks_artifact="eog", copy=True, verbose="ERROR")
        except Exception:
            pass
    raw.pick("eeg")
    return raw


def _rexstate_segments(events_tsv: Path) -> list[tuple[float, float, int]]:
    """Return (onset, duration, label) for stable Walk/Stop states."""
    df = pd.read_csv(events_tsv, sep="\t")
    out = []
    for _, r in df.iterrows():
        lab = STATE_TO_LABEL.get(str(r["trial_type"]))
        if lab is not None and float(r["duration"]) > 0:
            out.append((float(r["onset"]), float(r["duration"]), lab))
    return out


def _imu_mag_series(subject: str, ses: int, task: str, tracksys: str):
    """Return (accel-magnitude, gyro-magnitude, sfreq) for one IMU (head|exo)."""
    mdir = DATA_ROOT / subject / f"ses-{ses:02d}" / "motion"
    tsv = mdir / f"{subject}_ses-{ses:02d}_task-{task}_tracksys-{tracksys}_motion.tsv"
    js = mdir / f"{subject}_ses-{ses:02d}_task-{task}_tracksys-{tracksys}_motion.json"
    if not tsv.exists():
        return None
    try:
        data = pd.read_csv(tsv, sep="\t")
        def mag(pattern):
            cols = [c for c in data.columns if re.search(pattern, str(c), re.I)]
            if not cols:
                return None
            v = data[cols].astype(float).to_numpy()
            v = v - v.mean(axis=0, keepdims=True)      # drop gravity / DC offset
            return np.sqrt((v ** 2).sum(axis=1))
        acc, gyr = mag(r"acc"), mag(r"gyro")
        if acc is None:
            return None
        if gyr is None:
            gyr = np.zeros_like(acc)
        sf = 100.0
        if js.exists():
            sf = float(json.loads(js.read_text()).get("SamplingFrequency", 100.0))
        return acc, gyr, sf
    except Exception:
        return None


def _imu_window_feats(imu, a, b, sf):
    """[mean, std, max] of accel-mag and gyro-mag over EEG-sample window [a,b)."""
    if imu is None:
        return [0.0] * 6
    acc, gyr, msf = imu
    ma, mb = int(a / sf * msf), int(b / sf * msf)
    out = []
    for series in (acc, gyr):
        s = series[ma:mb]
        out += [float(np.mean(s)), float(np.std(s)), float(np.max(s))] if len(s) else [0.0, 0.0, 0.0]
    return out


def _windows_from_task(subject, ses, task, raw, seg, win, step, margin, seg_base):
    """Slice one task's stable segments into windows; attach IMU features + segment ids."""
    data = raw.get_data()                     # (C, N)
    sf = round(raw.info["sfreq"])             # some EDFs report 99.999.../100.0001
    wlen, wstep = round(win * sf), round(step * sf)
    mrg = round(margin * sf)
    head = _imu_mag_series(subject, ses, task, "head")
    exo = _imu_mag_series(subject, ses, task, "exo")

    X, y, mot, sid, feats = [], [], [], [], []
    for gi, (onset, dur, lab) in enumerate(seg):
        s0 = int(onset * sf) + mrg
        s1 = min(int((onset + dur) * sf), data.shape[1])   # clamp to recording length
        for a in range(s0, s1 - wlen + 1, wstep):
            w = data[:, a:a + wlen]
            if w.shape[1] != wlen:                       # guard against edge truncation
                continue
            X.append(w)
            y.append(lab)
            sid.append(seg_base + gi)
            hf = _imu_window_feats(head, a, a + wlen, sf)   # head: 6 feats
            ef = _imu_window_feats(exo, a, a + wlen, sf)    # exo: 6 feats
            feats.append(hf + ef)
            mot.append(hf[1])                               # head accel std (legacy scalar)
    return X, y, mot, sid, feats, seg_base + len(seg)


def build_epochs(subject="sub-01", sessions=None, tasks=MI_TASKS,
                 win=2.0, step=0.5, margin=0.5, l_freq=8.0, h_freq=30.0,
                 zscore=True, use_cache=True) -> EpochSet:
    """Load, preprocess and epoch the requested sessions/tasks into an EpochSet."""
    sessions = sessions or list_sessions(subject)
    days = session_days(subject)
    tag = f"{subject}_s{'-'.join(map(str, sessions))}_{'-'.join(tasks)}_w{win}_st{step}_{l_freq}-{h_freq}_z{int(zscore)}_v2imu"
    cache = CACHE_DIR / f"{tag}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        return EpochSet(d["X"], d["y"], d["session"], d["day"], d["task"],
                        d["motion"], d["segment"], d["imu_feats"],
                        list(d["ch_names"]), float(d["sfreq"]))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    allX, ally, allses, allday, alltask, allmot, allsid, allfeat = [], [], [], [], [], [], [], []
    ch_names, sfreq = None, 100.0
    seg_base = 0

    for ses in sessions:
        eeg_dir = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg"
        for task in tasks:
            edf = eeg_dir / f"{subject}_ses-{ses:02d}_task-{task}_eeg.edf"
            evt = eeg_dir / f"{subject}_ses-{ses:02d}_task-{task}_acq-rexstate_events.tsv"
            if not (edf.exists() and evt.exists()):
                continue
            seg = _rexstate_segments(evt)
            if not seg:
                continue
            raw = _load_raw(edf, l_freq, h_freq)
            ch_names, sfreq = raw.ch_names, raw.info["sfreq"]
            X, y, mot, sid, feats, seg_base = _windows_from_task(
                subject, ses, task, raw, seg, win, step, margin, seg_base)
            allX += X; ally += y; allmot += mot; allsid += sid; allfeat += feats
            allses += [ses] * len(y); allday += [days.get(ses, -1)] * len(y)
            alltask += [task] * len(y)
            print(f"  ses-{ses:02d} {task:9s}: {len(y):4d} epochs "
                  f"(walk={sum(y)}, stop={len(y)-sum(y)})", flush=True)

    X = np.asarray(allX, dtype=np.float32)
    y = np.asarray(ally, dtype=np.int64)
    if zscore and len(X):
        mu = X.mean(axis=2, keepdims=True)
        sd = X.std(axis=2, keepdims=True) + 1e-7
        X = ((X - mu) / sd).astype(np.float32)

    es = EpochSet(X, y, np.asarray(allses), np.asarray(allday),
                  np.asarray(alltask, dtype=object), np.asarray(allmot, dtype=np.float32),
                  segment=np.asarray(allsid, dtype=np.int64),
                  imu_feats=np.asarray(allfeat, dtype=np.float32),
                  ch_names=ch_names or [], sfreq=sfreq)
    np.savez_compressed(cache, X=es.X, y=es.y, session=es.session, day=es.day,
                        task=es.task, motion=es.motion, segment=es.segment,
                        imu_feats=es.imu_feats,
                        ch_names=np.array(es.ch_names, dtype=object), sfreq=es.sfreq)
    return es


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("Session days:", session_days())
    es = build_epochs(use_cache=False)
    print(f"\nTotal epochs: {len(es)}  | shape {es.X.shape}  | sfreq {es.sfreq}")
    print(f"Channels ({len(es.ch_names)}): {es.ch_names}")
    for s in list_sessions():
        m = es.session == s
        print(f"ses-{s:02d} day{es.day[m][0] if m.any() else '?':>3}: "
              f"{m.sum():4d} epochs  walk={es.y[m].sum():4d} stop={(es.y[m]==0).sum():4d}")
