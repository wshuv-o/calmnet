import numpy as np


def grouped_split(segment, y, frac=0.3, seed=0):
    """Hold out ~frac of each class's segments, keeping segments whole."""
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
    mask = np.ones(len(y), bool)
    mask[minor] = False
    return idx[mask], minor
