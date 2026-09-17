"""Third cohort: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB).

Loaded to match paper/PREREGISTRATION_cohort3.md exactly. That file was
committed before any model touched this cohort, and this loader must not
deviate from it: motor execution runs only, rest (T0) against movement
(T1 or T2), 8-30 Hz zero-phase fourth-order Butterworth, 100 Hz, 2 s windows
at 0.5 s step, and a window labelled only if it lies wholly inside one
annotation.

Nothing is cached to disk. The whole cohort is a few hundred megabytes in
memory, and the drive is the binding constraint on this machine.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = (Path(__file__).resolve().parent.parent / "data" / "eegbci"
        / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0")

FIT_RUNS = (3, 5, 7, 9)
TEST_RUNS = (11, 13)
SFREQ = 100.0
WIN, STEP = 2.0, 0.5
BAND = (8.0, 30.0)
REST, MOVE = 0, 1


def subjects():
    """Every subject folder present, S001-S020. None excluded on results."""
    out = []
    for i in range(1, 21):
        d = ROOT / ("S%03d" % i)
        if all((d / ("S%03dR%02d.edf" % (i, r))).exists()
               for r in FIT_RUNS + TEST_RUNS):
            out.append("S%03d" % i)
    return out


def _run(sub: str, run: int):
    """Windows, labels and block ids for one run, in time order."""
    f = ROOT / sub / ("%sR%02d.edf" % (sub, run))
    raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.resample(SFREQ, verbose="ERROR")
    sos = butter(4, BAND, btype="band", fs=SFREQ, output="sos")
    data = sosfiltfilt(sos, raw.get_data(), axis=-1).astype(np.float32)

    wlen, wstep = int(WIN * SFREQ), int(STEP * SFREQ)
    X, y, blk = [], [], []
    for bi, a in enumerate(raw.annotations):
        desc = str(a["description"])
        if desc not in ("T0", "T1", "T2"):
            continue
        lab = REST if desc == "T0" else MOVE
        s0 = int(round(a["onset"] * SFREQ))
        s1 = int(round((a["onset"] + a["duration"]) * SFREQ))
        # Only windows lying wholly inside this annotation (pre-registered).
        for st in range(s0, s1 - wlen + 1, wstep):
            if st + wlen > data.shape[1]:
                break
            X.append(data[:, st:st + wlen])
            y.append(lab)
            blk.append(bi)
    return (np.stack(X) if X else np.empty((0, data.shape[0], wlen), np.float32),
            np.asarray(y, np.int64), np.asarray(blk, np.int64))


def build_subject(sub: str):
    """Fit and test sets, with run, task and globally increasing block ids."""
    parts = {}
    base = 0
    for split, runs in (("fit", FIT_RUNS), ("test", TEST_RUNS)):
        Xs, ys, rs, gs = [], [], [], []
        for r in runs:
            X, y, blk = _run(sub, r)
            if len(y) == 0:
                continue
            Xs.append(X)
            ys.append(y)
            rs.append(np.full(len(y), r, np.int64))
            gs.append(blk + base)          # non-decreasing within each run
            base += int(blk.max()) + 1
        parts[split] = (np.concatenate(Xs), np.concatenate(ys),
                        np.concatenate(rs), np.concatenate(gs))
    return parts


def block_length_windows(sub: str) -> float:
    """Median windows per class block across the fit runs.

    Diagnostic only, computed after the prediction was registered, to confirm
    the protocol value used there. It does not feed any decision.
    """
    lens = []
    for r in FIT_RUNS:
        _, y, blk = _run(sub, r)
        for b in np.unique(blk):
            lens.append(int((blk == b).sum()))
    return float(np.median(lens)) if lens else float("nan")
