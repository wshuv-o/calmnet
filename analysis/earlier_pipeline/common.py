"""Helpers shared by the earlier-pipeline experiments (unchanged from the runs)."""
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def balanced_accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.any():
            accs.append((y_pred[m] == c).mean())
    return float(np.mean(accs))


def grouped_split(segment, y, frac=0.15, seed=0):
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


def imu_valid_mask(imu_feats, session):
    valid = np.ones(len(imu_feats), bool)
    for s in np.unique(session):
        m = session == s
        if not imu_feats[m].any():
            valid[m] = False
    return valid


def normalise(scheme, Xf, Xc, Xt):
    """none: one global scalar; perwindow: per channel per window; global: per channel from Xf."""
    if scheme == "none":
        s = float(Xf.std()) + 1e-30
        return Xf / s, Xc / s, Xt / s
    if scheme == "perwindow":
        f = lambda X: ((X - X.mean(axis=2, keepdims=True))
                       / (X.std(axis=2, keepdims=True) + 1e-7)).astype(np.float32)
        return f(Xf), f(Xc), f(Xt)
    if scheme == "global":
        mu = Xf.mean(axis=(0, 2), keepdims=True)
        sd = Xf.std(axis=(0, 2), keepdims=True) + 1e-30
        f = lambda X: ((X - mu) / sd).astype(np.float32)
        return f(Xf), f(Xc), f(Xt)
    raise ValueError(scheme)


def bandpass(X, lo, hi, sfreq=100.0, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def log_var(X):
    return np.log(X.var(axis=-1) + 1e-8).astype(np.float32)


def covariances(X, eps=1e-5):
    Xc = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", Xc, Xc) / X.shape[-1]
    scale = np.einsum("nii->n", C)[:, None, None] / X.shape[1]
    return C + eps * np.maximum(scale, 1e-30) * np.eye(X.shape[1], dtype=np.float32)


def euclidean_align(C):
    M = C.mean(0)
    w, V = np.linalg.eigh(M)
    Minv = V @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-10, None))) @ V.T
    return np.einsum("ij,njk,kl->nil", Minv, C, Minv).astype(np.float32)


def tangent(C):
    w, V = np.linalg.eigh(C)
    L = np.einsum("nck,nk,ndk->ncd", V, np.log(np.clip(w, 1e-10, None)), V)
    n = C.shape[1]
    m = np.ones((n, n), np.float32)
    m[np.triu_indices(n, 1)] = np.sqrt(2.0)
    iu = np.triu_indices(n)
    return (L * m)[:, iu[0], iu[1]].astype(np.float32)


def invariance_r2_conditional(F, M, y, groups, n_splits=3, n_components=64):
    """Movement decodable from F beyond the class label."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    M = np.asarray(M, dtype=np.float64).copy()
    for c in np.unique(y):
        m = y == c
        if m.sum() > 1:
            M[m] -= M[m].mean(0)
    groups, F = np.asarray(groups), np.asarray(F, dtype=np.float64)
    good = np.isfinite(F).all(axis=1) & np.isfinite(M).all(axis=1)
    if good.sum() < 30:
        return float("nan")
    F, M, groups = F[good], M[good], groups[good]
    n_g = len(np.unique(groups))
    if len(F) < 30 or n_g < 2:
        return float("nan")
    Mz = (M - M.mean(0)) / (M.std(0) + 1e-6)
    k = int(min(n_components, F.shape[1], len(F) // 4))
    est = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                        RidgeCV(alphas=np.logspace(-2, 4, 13)))
    pred = cross_val_predict(est, F, Mz, cv=GroupKFold(n_splits=min(n_splits, n_g)),
                             groups=groups)
    return float(r2_score(Mz, pred, multioutput="variance_weighted"))
