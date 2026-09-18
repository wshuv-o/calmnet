"""Fifth cohort: DECODED (EUROBENCH), lower-limb exoskeleton, in pipeline shape.

Why this cohort
---------------
It is the closest public match to cohort A that exists. Participants wear a
lower-limb exoskeleton and alternate between standing relaxed with the device
stopped and walking while performing kinesthetic motor imagery, which is the
same stop-against-walk-intent contrast cohort A decodes. Recordings are
separated by months rather than weeks, so the session shift the alignment layer
exists to correct is larger here than anywhere else in the project.

Protocol, from CONDITIONS.txt and the JSON sidecars. Condition 3 is 16 trials
of 75 s, each holding 15 s standing relaxed, 24 s walking under motor imagery,
22 s walking under regressive subtraction, and 14 s standing relaxed. Event
codes are in the task column of each CSV:

    402  relax, standing, exoskeleton stopped   -> class 0 (Stop)
    404  motor imagery, exoskeleton moving      -> class 1 (Walk)
    406  regressive subtraction, still moving   -> dropped

The counting task is dropped rather than folded into either class: it is a
distraction condition, not a movement-intent state, and merging it would make
the label mean something different here than in the other cohorts.

Preprocessing matches cohorts B and C: fourth-order Butterworth 8-30 Hz,
resampled 200 to 100 Hz, 4 s windows at 0.5 s step. Windows never cross a block
boundary, so every window carries one label.

Split. Only one participant recorded on two dates; the rest have a single
session of 16 runs. An across-session split is therefore not available on this
cohort, and the split is temporal across runs instead: runs 1-8 fit, runs 9-16
test, which is the convention cohort B already uses. This tests less drift than
cohorts A, C and D and that limitation is stated rather than hidden, since the
drift claim is what the project is about.
"""
from __future__ import annotations

import io
import os
import re
import zipfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, filtfilt

ZIP = os.environ.get("DECODED_ZIP", r"E:\calmnet_data\decoded\dataset.zip")
PROTOCOL = os.environ.get("DECODED_PROTOCOL", "EXPERIENCE")
SF_IN, SF = 200.0, 100.0
BAND = (8.0, 30.0)
CODES = {402: 0, 404: 1}           # relax/standing -> Stop, motor imagery -> Walk
_ZC = {}


def _zf():
    if "z" not in _ZC:
        _ZC["z"] = zipfile.ZipFile(ZIP)
    return _ZC["z"]


def subjects():
    """Participants with enough condition-3 runs to split into fit and test."""
    pat = re.compile(r"%s/(subject_\d+)/" % PROTOCOL)
    names = set()
    for n in _zf().namelist():
        m = pat.match(n)
        if m:
            names.add(m.group(1))
    return sorted(s for s in names if len(_runs(s)) >= 8)


def _dates(sub):
    z = _zf()
    return sorted({m.group(1) for n in z.namelist()
                   if (m := re.search(r"%s/%s/CSV/(\d{8})/" % (PROTOCOL, sub), n))})


def _runs(sub):
    """Condition-3 run files, whether or not a date folder sits in the path.

    Most participants store runs directly under CSV/; a few have a date
    subfolder. Both layouts appear in the same archive.
    """
    pat = re.compile(r"%s/%s/CSV/(?:\d{8}/)?%s_cond_03_run_(\d+)_EEG\.csv$"
                     % (PROTOCOL, sub, sub))
    hits = [(int(m.group(1)), n) for n in _zf().namelist()
            if (m := pat.match(n))]
    return [n for _, n in sorted(hits)]


def _read_run(name):
    """One run: EEG array (C, T) at SF, and the per-sample task code."""
    raw = _zf().read(name).decode("utf-8", "replace")
    lines = raw.strip().split("\n")
    head = lines[0].strip().split(";")
    cols = {c: i for i, c in enumerate(head)}
    eeg_names = [c for c in head
                 if c not in ("time", "task", "HR", "HL", "VU", "VD")]
    if "task" not in cols or not eeg_names:
        return None, None, None
    idx = [cols[c] for c in eeg_names]
    ti = cols["task"]
    data = np.empty((len(lines) - 1, len(idx) + 1), np.float64)
    keep = 0
    for ln in lines[1:]:
        p = ln.split(";")
        if len(p) <= ti:
            continue
        try:
            data[keep, :-1] = [float(p[i]) for i in idx]
            data[keep, -1] = float(p[ti] or 0)
        except ValueError:
            continue
        keep += 1
    if keep < int(5 * SF_IN):
        return None, None, None
    data = data[:keep]
    X, task = data[:, :-1].T, data[:, -1]

    b, a = butter(4, [BAND[0] / (SF_IN / 2), BAND[1] / (SF_IN / 2)],
                  btype="band")
    X = filtfilt(b, a, X, axis=-1)
    step = int(round(SF_IN / SF))
    return X[:, ::step].astype(np.float32), task[::step].astype(int), eeg_names


def _windows(X, task, win=4.0, step=0.5):
    """Sliding windows inside each contiguous single-code block.

    Windows never span a boundary, so no window mixes relax with motor imagery
    and no label is a majority vote.
    """
    n = int(round(win * SF))
    hop = max(1, int(round(step * SF)))
    xs, ys, segs = [], [], []
    change = np.flatnonzero(np.diff(task) != 0) + 1
    bounds = np.concatenate([[0], change, [len(task)]])
    for si in range(len(bounds) - 1):
        a, b = bounds[si], bounds[si + 1]
        code = int(task[a])
        if code not in CODES or (b - a) < n:
            continue
        for s in range(a, b - n + 1, hop):
            xs.append(X[:, s:s + n])
            ys.append(CODES[code])
            segs.append(si)
    if not xs:
        return None, None, None
    return np.stack(xs), np.array(ys, int), np.array(segs, int)


class ES:
    pass


def build_subject(sub, win=4.0, step=0.5, zscore=False):
    X, y, sess, seg, task, chn = [], [], [], [], [], None
    runs = _runs(sub)
    base = 0
    for ri, run in enumerate(runs):
        sig, tk, names = _read_run(run)
        if sig is None:
            continue
        chn = chn or names
        xs, ys, sg = _windows(sig, tk, win, step)
        if xs is None:
            continue
        X.append(xs)
        y.append(ys)
        # One "session" per run, so streams() treats each run as its own
        # contiguous recording and no window context spans two runs.
        sess.append(np.full(len(ys), ri, int))
        seg.append(sg + base)
        task.append(np.full(len(ys), "exo", object))
        base = int(sg.max()) + 1 + base
    if not X:
        raise RuntimeError("no usable runs for %s" % sub)
    es = ES()
    es.X = np.concatenate(X)
    if zscore:
        es.X = ((es.X - es.X.mean(-1, keepdims=True))
                / (es.X.std(-1, keepdims=True) + 1e-8))
    es.y = np.concatenate(y)
    es.session = np.concatenate(sess)
    es.trial = es.session
    es.segment = np.concatenate(seg)
    es.task = np.concatenate(task)
    es.sfreq, es.ch_names = SF, chn or []
    es.motion = np.zeros((len(es.y), 1), np.float32)
    es.n_runs = int(es.session.max()) + 1

    def by_sessions(which):
        m = np.isin(es.session, which)
        o = ES()
        for a in ("X", "y", "session", "trial", "task", "segment", "motion"):
            setattr(o, a, getattr(es, a)[m])
        o.sfreq, o.ch_names = es.sfreq, es.ch_names
        return o

    es.by_sessions = by_sessions
    return es


if __name__ == "__main__":
    ss = subjects()
    print("usable subjects: %d  %s" % (len(ss), ss))
    es = build_subject(ss[0])
    print("%s  X %s  classes %s  runs %d  channels %d"
          % (ss[0], es.X.shape, np.bincount(es.y), es.n_runs,
             len(es.ch_names)))
