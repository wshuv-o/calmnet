"""How much of cohort A sits in blocks longer than the estimator memory.

Same block definition as tools/tau_blk_rule.py (contiguous single-class runs of
windows, counted within a recording, pooled over participants), but the stream
grouping is inlined so that torch is never imported.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Shuvo\Downloads\bci-review\calm-net\src")
from dataio import build_epochs  # noqa: E402

MU = 160


def runs_within_recordings(y, session, task, segment):
    """contiguous single-class runs, never crossing a recording"""
    key = np.array(["%s|%s" % (s, t) for s, t in zip(session, task)])
    out = []
    for rec in np.unique(key):
        m = key == rec
        ys, seg = np.asarray(y)[m], np.asarray(segment)[m]
        order = np.argsort(seg, kind="stable")
        ys = ys[order]
        if not len(ys):
            continue
        cut = np.flatnonzero(np.diff(ys) != 0)
        b = np.concatenate([[-1], cut, [len(ys) - 1]])
        out += [int(v) for v in np.diff(b)]
    return out


allr = []
for i in range(1, 8):
    es = build_epochs(subject="sub-0%d" % i, win=4.0, step=0.5, zscore=False)
    allr += runs_within_recordings(es.y.astype(int), es.session, es.task, es.segment)

a = np.array(allr)
tot = a.sum()
long = a[a > MU]
print("cohort A: %d blocks, median %.0f, mean %.1f, longest %d, %d windows total"
      % (len(a), np.median(a), a.mean(), a.max(), tot))
print("blocks longer than mu=%d: %d (%.1f %% of blocks), holding %.1f %% of all windows"
      % (MU, len(long), 100 * len(long) / len(a), 100 * long.sum() / tot))
for q in (50, 75, 90, 95, 99):
    print("  %2dth percentile: %.0f windows" % (q, np.percentile(a, q)))
