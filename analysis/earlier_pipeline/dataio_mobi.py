"""Cohort B (MoBI treadmill) windows for the earlier pipeline."""
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")

_ROOT = Path(os.environ.get("CALMNET_DATA", Path(__file__).resolve().parents[2] / "datasets"))
ROOT = _ROOT / "mobi_treadmill"
CACHE = _ROOT / "cache_mobi"
SFREQ = 100.0
CHUNK_S = 20.0                                  # grouping unit for the held-out split


@dataclass
class MobiEpochs:
    X: np.ndarray
    y: np.ndarray
    trial: np.ndarray
    segment: np.ndarray
    motion: np.ndarray                          # (N, 12) goniometer std / range

    def __len__(self):
        return len(self.y)

    def by_trials(self, trials):
        m = np.isin(self.trial, list(trials))
        return MobiEpochs(self.X[m], self.y[m], self.trial[m], self.segment[m], self.motion[m])


def subjects():
    if not ROOT.is_dir():
        return []
    return sorted({p.name.split("-")[0] for p in ROOT.glob("SL*-T*") if p.is_dir()})


def _walk_bounds(trial_dir):
    on = off = None
    cond = trial_dir / "conductor.txt"
    if cond.exists():
        ev = []
        for l in cond.read_text(errors="replace").splitlines()[2:]:
            q = l.split()
            if len(q) >= 2:
                try:
                    ev.append((float(q[0]), int(float(q[1]))))
                except ValueError:
                    pass
        t8 = [t for t, c in ev if c == 8]
        t10 = [t for t, c in ev if c == 10]
        if t8:
            on = min(t8)
        cand = [t for t in t10 if on is None or t > on + 60]
        if cand:
            off = max(cand)
    return on, off


def build_subject(sub, win=2.0, step=0.5, l_freq=8.0, h_freq=30.0, margin=2.0,
                  use_cache=True, zscore=True):
    tag = f"{sub}_w{win}_s{step}_{l_freq}-{h_freq}_c{int(CHUNK_S)}" + ("" if zscore else "_z0")
    cf = CACHE / f"{tag}.npz"
    if use_cache and cf.exists():
        d = np.load(cf, allow_pickle=True)
        return MobiEpochs(d["X"], d["y"], d["trial"], d["segment"], d["motion"])

    sos = butter(4, [l_freq, h_freq], btype="band", fs=SFREQ, output="sos")
    wlen, wstep, mrg = int(win * SFREQ), int(step * SFREQ), int(margin * SFREQ)
    chunk = int(CHUNK_S * SFREQ)
    X, y, tr, seg, mot = [], [], [], [], []
    segid = 0
    for t_idx in (1, 2, 3):
        d = ROOT / f"{sub}-T{t_idx:02d}"
        if not (d / "eeg.txt").exists():
            continue
        E = np.loadtxt(d / "eeg.txt", skiprows=1, dtype=np.float32)
        J = np.loadtxt(d / "joints.txt", skiprows=2, dtype=np.float32)
        te, eeg, G = E[:, 0], E[:, 1:].T, J[:, 1:7]     # measured goniometers only
        n = min(eeg.shape[1], len(G))
        eeg, G, te = eeg[:, :n], G[:n], te[:n]
        on, off = _walk_bounds(d)
        if on is None or off is None:
            continue
        eeg = sosfiltfilt(sos, eeg, axis=-1).astype(np.float32)
        for b0, b1, lab in ((0.0, on, 0), (on, off, 1), (off, te[-1], 0)):
            a0, a1 = int(b0 * SFREQ) + mrg, int(b1 * SFREQ) - mrg
            if a1 - a0 < wlen:
                continue
            for a in range(a0, a1 - wlen + 1, wstep):
                w = eeg[:, a:a + wlen]
                if w.shape[1] != wlen:
                    continue
                c0 = (a - a0) // chunk
                if (a - a0 + wlen - 1) // chunk != c0:
                    continue
                gw = G[a:a + wlen]
                X.append(w); y.append(lab); tr.append(t_idx); seg.append(segid + int(c0))
                mot.append([v for j in range(gw.shape[1])
                            for v in (float(gw[:, j].std()), float(gw[:, j].max() - gw[:, j].min()))])
            segid += int((a1 - a0) // chunk) + 1
    if not X:
        return None
    X = np.asarray(X, dtype=np.float32)
    if zscore:
        X = ((X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-7)).astype(np.float32)
    else:
        X = (X / (float(X.std()) + 1e-30)).astype(np.float32)
    es = MobiEpochs(X, np.asarray(y, np.int64), np.asarray(tr, np.int64),
                    np.asarray(seg, np.int64), np.asarray(mot, np.float32))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cf, X=es.X, y=es.y, trial=es.trial, segment=es.segment, motion=es.motion)
    return es
