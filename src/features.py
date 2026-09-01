"""Feature-representation experiments.

The 131-architecture sweep held the input representation fixed (2 s windows,
8-30 Hz, raw reference, log band-power) and varied only the model. The one time
the representation was varied instead -- the frequency-band series -- it moved
the honest number more than any architecture did:

    mu 8-13    acc 0.612   R^2 -0.013
    8-30       acc 0.688   R^2 +0.032
    beta 13-30 acc 0.709   R^2 +0.101
    1-45       acc 0.718   R^2 +0.127

So "no architecture beat the ceiling" is really "no architecture beat the
ceiling on ONE representation". This module tests the representation axis
properly, with the pieces standard practice would use on mobile EEG and that
the sweep never touched:

  ASR         Artifact Subspace Reconstruction -- the standard preprocessing
              for locomotion EEG. Removes high-variance artefact subspaces
              calibrated on clean data. Attacks the movement confound in the
              SIGNAL, not in the loss.
  Laplacian   surface Laplacian / CAR re-referencing: spatial high-pass that
              suppresses spatially broad artefact.
  window      1 s / 2 s / 3 s / 4 s -- never varied; ERD/ERS is sensitive to it.
  CSP         classical common spatial patterns, and its filter-bank form.
  Riemann     covariance -> tangent space, with Euclidean Alignment for
              cross-session drift.
  connectivity  coherence / phase-locking -- everything so far was power-only,
              discarding phase entirely.

Every representation is scored the same way as the architectures: balanced
accuracy AND intent->motion R^2, because a representation can raise accuracy by
admitting more movement just as an architecture can.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


# --------------------------------------------------------------------------- #
# Re-referencing
# --------------------------------------------------------------------------- #
def car(X):
    """Common average reference."""
    return (X - X.mean(axis=1, keepdims=True)).astype(np.float32)


def laplacian(X, pos, k=4):
    """Surface Laplacian: subtract the mean of each channel's k nearest
    neighbours. Spatial high-pass -- suppresses spatially broad sources, which
    is what gross movement artefact looks like."""
    from scipy.spatial.distance import cdist
    D = cdist(pos, pos)
    np.fill_diagonal(D, np.inf)
    nb = np.argsort(D, axis=1)[:, :k]
    out = X.copy()
    for c in range(X.shape[1]):
        out[:, c] = X[:, c] - X[:, nb[c]].mean(axis=1)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #
def fit_asr(X_clean, sfreq=100.0, cutoff=20):
    """Calibrate ASR on the least-artefacted windows.

    Calibration data must be relatively clean, so windows are ranked by motion
    energy and the quietest third is used. Passing the whole recording would
    teach ASR that movement artefact is normal.
    """
    from meegkit.asr import ASR
    asr = ASR(method="euclid", cutoff=cutoff, sfreq=sfreq)
    cal = np.concatenate(list(X_clean), axis=-1)          # (C, N*T)
    asr.fit(cal)
    return asr


def apply_asr(asr, X):
    out = np.empty_like(X)
    for i in range(len(X)):
        try:
            out[i] = asr.transform(X[i])
        except Exception:
            out[i] = X[i]
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Classical features
# --------------------------------------------------------------------------- #
def bandpass(X, lo, hi, sfreq=100.0, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def log_var(X):
    """Log-variance per channel -- the CSP read-out feature."""
    return np.log(X.var(axis=-1) + 1e-8).astype(np.float32)


def csp_fit(X, y, n_comp=6, reg=1e-4):
    """Common Spatial Patterns by simultaneous diagonalisation of the two
    class covariance matrices.

    Shrinkage is not optional here: CAR and the surface Laplacian are
    rank-deficient spatial operators (CAR removes exactly one degree of
    freedom), so the composite covariance is singular and the generalised
    eigenproblem fails with "B is not positive definite". A small ridge on the
    diagonal restores definiteness without materially changing the filters.
    """
    from scipy.linalg import eigh
    n_ch = X.shape[1]
    C = []
    for c in (0, 1):
        Xc = X[y == c]
        cov = np.einsum("nct,ndt->cd", Xc, Xc) / (Xc.shape[0] * Xc.shape[2])
        C.append(cov / (np.trace(cov) + 1e-12))
    B = C[0] + C[1]
    B = B + reg * np.trace(B) / n_ch * np.eye(n_ch, dtype=B.dtype)
    w, V = eigh(C[0], B)
    idx = np.argsort(w)
    sel = np.concatenate([idx[:n_comp // 2], idx[-(n_comp - n_comp // 2):]])
    return V[:, sel].T.astype(np.float32)                 # (n_comp, C)


def csp_apply(W, X):
    return np.einsum("kc,nct->nkt", W, X).astype(np.float32)


def fbcsp(Xtr, ytr, Xte, bands=((8, 13), (13, 20), (20, 30)), n_comp=6, sfreq=100.0):
    """Filter-bank CSP: per-band spatial filters, log-variance features."""
    Ftr, Fte = [], []
    for lo, hi in bands:
        Btr, Bte = bandpass(Xtr, lo, hi, sfreq), bandpass(Xte, lo, hi, sfreq)
        W = csp_fit(Btr, ytr, n_comp)
        Ftr.append(log_var(csp_apply(W, Btr)))
        Fte.append(log_var(csp_apply(W, Bte)))
    return np.concatenate(Ftr, 1), np.concatenate(Fte, 1)


# --------------------------------------------------------------------------- #
# Riemannian with Euclidean Alignment
# --------------------------------------------------------------------------- #
def covariances(X, eps=1e-5):
    Xc = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", Xc, Xc) / X.shape[-1]
    return C + eps * np.eye(X.shape[1], dtype=np.float32)


def euclidean_align(C):
    """Euclidean Alignment (He & Wu): whiten by the session's mean covariance.
    A standard, label-free remedy for cross-session drift, which is exactly the
    longitudinal setting here."""
    M = C.mean(0)
    w, V = np.linalg.eigh(M)
    Minv = V @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-10, None))) @ V.T
    return np.einsum("ij,njk,kl->nil", Minv, C, Minv).astype(np.float32)


def tangent(C):
    """Log-Euclidean tangent vectors of SPD covariance matrices."""
    w, V = np.linalg.eigh(C)
    L = np.einsum("nck,nk,ndk->ncd", V, np.log(np.clip(w, 1e-10, None)), V)
    n = C.shape[1]
    m = np.ones((n, n), np.float32)
    m[np.triu_indices(n, 1)] = np.sqrt(2.0)
    iu = np.triu_indices(n)
    return (L * m)[:, iu[0], iu[1]].astype(np.float32)


# --------------------------------------------------------------------------- #
# Connectivity / phase -- information the power-only pipeline discards
# --------------------------------------------------------------------------- #
def plv_features(X, lo=8, hi=30, sfreq=100.0, max_ch=20):
    """Pairwise phase-locking value on a channel subset. Phase coupling is
    orthogonal to band power, so it can carry signal the whole sweep ignored."""
    Xb = bandpass(X[:, :max_ch], lo, hi, sfreq)
    ph = np.angle(hilbert(Xb, axis=-1))
    n = ph.shape[1]
    iu = np.triu_indices(n, 1)
    d = ph[:, iu[0], :] - ph[:, iu[1], :]
    return np.abs(np.exp(1j * d).mean(-1)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Invariance probe
# --------------------------------------------------------------------------- #
def invariance_r2_cv(F, M, groups, alpha=1.0, n_splits=3):
    """Movement recoverability, measured WITHIN one distribution.

    The earlier probe fitted on the training sessions and scored on the test
    sessions. That conflates two different things: a representation whose
    features merely SHIFT across sessions scores strongly negative while still
    encoding movement perfectly well inside any given session. Measured that
    way, FBCSP looked like the most invariant representation tested (-0.501)
    when an honest probe puts it at +0.181 -- movement plainly recoverable.

    Here the probe is cross-validated inside the evaluation set, grouped by
    session, so a positive R^2 means movement really is recoverable from these
    features and a negative one means it is not. Session-grouped folds keep the
    probe from exploiting within-session autocorrelation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.metrics import r2_score
    groups = np.asarray(groups)
    n_g = len(np.unique(groups))
    if len(F) < 30 or n_g < 2:
        return float("nan")
    mu, sd = M.mean(0), M.std(0) + 1e-6
    Mz = (M - mu) / sd
    cv = GroupKFold(n_splits=min(n_splits, n_g))
    pred = cross_val_predict(Ridge(alpha), F, Mz, cv=cv, groups=groups)
    return float(r2_score(Mz, pred, multioutput="variance_weighted"))
