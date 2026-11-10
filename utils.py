import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalise_global(Xf, Xc, Xt):
    """Per-channel mean and SD from the fitting split, applied to all splits."""
    mu = Xf.mean(axis=(0, 2), keepdims=True)
    sd = Xf.std(axis=(0, 2), keepdims=True) + 1e-30
    f = lambda X: ((X - mu) / sd).astype(np.float32)
    return f(Xf), f(Xc), f(Xt)


def weighted_ce(logits, y, weight):
    """Class-weighted cross-entropy, written as in the original runs (same rounding)."""
    ce = F.cross_entropy(logits, y, weight=weight, reduction="none")
    return 0.5 * (ce.sum() / (len(y) + 1e-6)) + 0.5 * F.cross_entropy(logits, y, weight=weight)


def balanced_accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


def expected_calibration_error(probs, labels, n_bins=15):
    conf, correct = probs.max(axis=1), (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits, labels, max_iter=200):
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def softmax_np(logits, T=1.0):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def streams(session, task, segment):
    """Indices of each recording, in time order."""
    out = []
    key = np.array(["%s|%s" % (s, t) for s, t in zip(session, task)])
    for k in dict.fromkeys(key.tolist()):
        idx = np.flatnonzero(key == k)
        assert np.all(np.diff(np.asarray(segment)[idx]) >= 0), "stream %s out of order" % k
        out.append(idx)
    return out


def transition_matrix(y, strm, n_cls=2, prior=1.0):
    A = np.full((n_cls, n_cls), prior, float)
    for idx in strm:
        ys = np.asarray(y)[idx]
        for a, b in zip(ys[:-1], ys[1:]):
            A[int(a), int(b)] += 1.0
    return A / A.sum(1, keepdims=True)


def forward_filter(P, A, pi, eps=1e-12):
    """Causal two-state forward recursion over calibrated posteriors."""
    E = np.exp(np.log(np.clip(P, eps, None)) - np.log(np.clip(pi, eps, None)))
    a = pi * E[0]
    a = a / (a.sum() + eps)
    out = [a]
    for t in range(1, len(P)):
        a = (a @ A) * E[t]
        a = a / (a.sum() + eps)
        out.append(a)
    return np.stack(out)


def onset_metrics(y_true, y_pred, strm, step=0.5):
    """False onsets per minute of the negative class, median delay, missed-onset fraction."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    false_onsets, stop_windows, lat = 0, 0, []
    for idx in strm:
        yt, yp = y_true[idx], y_pred[idx]
        stop_windows += int((yt == 0).sum())
        rise = np.flatnonzero((yp[1:] == 1) & (yp[:-1] == 0)) + 1
        false_onsets += int(sum(1 for r in rise if yt[r] == 0))
        for r in np.flatnonzero((yt[1:] == 1) & (yt[:-1] == 0)) + 1:
            stop_after = np.flatnonzero(yt[r:] == 0)
            end = r + int(stop_after[0]) if len(stop_after) else len(yt)
            hit = np.flatnonzero(yp[r:end] == 1)
            lat.append(float(hit[0]) * step if len(hit) else float("nan"))
    stop_min = max(stop_windows * step / 60.0, 1e-9)
    return {"false_onsets_per_min": float(false_onsets / stop_min),
            "latency_s": float(np.nanmedian(lat)) if lat else float("nan"),
            "missed_onsets": float(np.mean([np.isnan(v) for v in lat])) if lat else float("nan"),
            "n_true_onsets": int(len(lat))}


def conditional_r2(F_, M, y, groups, n_splits=3, n_components=64):
    """Movement decodable from F_ beyond the class label (movement centred within class)."""
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
    groups, F_ = np.asarray(groups), np.asarray(F_, dtype=np.float64)
    good = np.isfinite(F_).all(axis=1) & np.isfinite(M).all(axis=1)
    if good.sum() < 30:
        return float("nan")
    F_, M, groups = F_[good], M[good], groups[good]
    n_g = len(np.unique(groups))
    if n_g < 2:
        return float("nan")
    Mz = (M - M.mean(0)) / (M.std(0) + 1e-6)
    k = int(min(n_components, F_.shape[1], len(F_) // 4))
    est = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                        RidgeCV(alphas=np.logspace(-2, 4, 13)))
    pred = cross_val_predict(est, F_, Mz, cv=GroupKFold(n_splits=min(n_splits, n_g)),
                             groups=groups)
    return float(r2_score(Mz, pred, multioutput="variance_weighted"))
