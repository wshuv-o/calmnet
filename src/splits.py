"""Leakage-free splitting helpers (group by contiguous state segment)."""
from __future__ import annotations
import numpy as np


def grouped_split(segment, y, frac=0.15, seed=0):
    """Split indices into (major, minor) so that whole segments stay together and
    both classes appear in the minor split. `frac` targets the minor size."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    minor = []
    for c in np.unique(y):
        segs = np.unique(segment[y == c])
        rng.shuffle(segs)
        k = max(1, int(round(len(segs) * frac)))
        pick = set(segs[:k].tolist())
        minor += [i for i in idx if segment[i] in pick and y[i] == c]
    minor = np.array(sorted(set(minor)))
    mask = np.ones(len(y), bool); mask[minor] = False
    return idx[mask], minor


def grouped_kfold(segment, y, k=4, seed=0):
    """Yield (train_idx, test_idx) folds where each fold holds out whole segments,
    stratified so every class is represented in each test fold."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    class_folds = {}
    for c in np.unique(y):
        segs = np.unique(segment[y == c]); rng.shuffle(segs)
        class_folds[c] = np.array_split(segs, k)
    for f in range(k):
        test_segs = set()
        for c in np.unique(y):
            test_segs.update(class_folds[c][f].tolist())
        test_mask = np.array([segment[i] in test_segs for i in idx])
        if test_mask.any() and (~test_mask).any():
            yield idx[~test_mask], idx[test_mask]
