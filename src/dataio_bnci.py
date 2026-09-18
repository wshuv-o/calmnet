"""Fourth cohort: BNCI2014-001 (BCI Competition IV-2a), in the pipeline's shape.

Why this dataset
----------------
Nine participants recorded across two sessions on different days, which is the
session-to-session setting this project is about; 22 EEG channels against our
60 and 64, so the tangent branch is tested on a third montage size; and trial
order is randomised, so a single-class block is one trial rather than the 309
windows of cohort B.

Preprocessing follows the existing cohorts: band-pass 8-30 Hz, resample to
100 Hz, one window per trial over the 4 s motor-imagery interval, giving
22 x 400 to match cohort A's window length exactly. Left hand against right
hand, the standard binary contrast on this dataset.

Split convention matches the other cohorts: fit on the first session, test on
the second. No trial from the test session is seen during fitting.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np

SFREQ = 100.0
BAND = (8.0, 30.0)
CLASSES = ("left_hand", "right_hand")


def subjects():
    return [str(s) for s in range(1, 10)]


def _prep(raw, picks):
    """Band-pass then resample, in that order, as the other loaders do."""
    raw = raw.copy().pick(picks)
    raw.filter(BAND[0], BAND[1], method="iir",
               iir_params=dict(order=4, ftype="butter"), verbose=False)
    raw.resample(SFREQ, verbose=False)
    return raw


def build_subject(sub, win=4.0, step=0.5, zscore=False):
    """One EpochSet-shaped object: X (n, C, T), y, session, task, segment.

    Returned attribute names match dataio_mobi so that streams() and the rest
    of the harness need no special case.
    """
    import mne
    from moabb.datasets import BNCI2014_001

    ds = BNCI2014_001()
    data = ds.get_data(subjects=[int(sub)])[int(sub)]

    X, y, sess, seg = [], [], [], []
    for si, (sname, runs) in enumerate(sorted(data.items())):
        for rname, raw in sorted(runs.items()):
            picks = mne.pick_types(raw.info, eeg=True, eog=False)
            r = _prep(raw, picks)
            ev, eid = mne.events_from_annotations(r, verbose=False)
            keep = {k: v for k, v in eid.items() if k in CLASSES}
            if not keep:
                continue
            ep = mne.Epochs(r, ev, event_id=keep, tmin=0.0,
                            tmax=win - 1.0 / SFREQ, baseline=None,
                            preload=True, verbose=False, on_missing="ignore")
            if len(ep) == 0:
                continue
            xs = ep.get_data(copy=True).astype(np.float32)
            lab = ep.events[:, -1]
            inv = {v: k for k, v in keep.items()}
            ys = np.array([0 if inv[v] == CLASSES[0] else 1 for v in lab], int)
            X.append(xs)
            y.append(ys)
            sess.append(np.full(len(ys), si, int))
            seg.append(np.arange(len(ys)) + 1000 * len(seg))
    if not X:
        raise RuntimeError("no trials for subject %s" % sub)

    X = np.concatenate(X)
    n = int(round(win * SFREQ))
    X = X[:, :, :n]
    if zscore:
        X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)

    class ES:
        pass

    es = ES()
    es.X = X
    es.y = np.concatenate(y)
    es.session = np.concatenate(sess)
    es.trial = es.session
    es.task = np.array(["mi"] * len(es.y), object)
    es.segment = np.concatenate(seg)
    es.sfreq = SFREQ
    es.ch_names = []
    es.motion = np.zeros((len(es.y), 1), np.float32)

    def by_sessions(which):
        m = np.isin(es.session, which)
        o = ES()
        for a in ("X", "y", "session", "trial", "task", "segment", "motion"):
            setattr(o, a, getattr(es, a)[m])
        o.sfreq, o.ch_names = SFREQ, es.ch_names
        return o

    es.by_sessions = by_sessions
    return es


if __name__ == "__main__":
    es = build_subject("1")
    print("subject 1: X", es.X.shape, "classes", np.bincount(es.y),
          "sessions", sorted(set(es.session.tolist())))
