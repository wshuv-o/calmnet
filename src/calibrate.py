"""Calibration: temperature scaling, Expected Calibration Error, reliability
curves, and split-conformal prediction sets."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter=200) -> float:
    """Fit a single scalar T>0 minimising NLL of softmax(logits/T) on a held-out split."""
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


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=15):
    """Top-label ECE with equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def reliability_curve(probs, labels, n_bins=10):
    """Return (bin_conf, bin_acc, bin_count) for a reliability diagram."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bc, ba, cnt = [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            bc.append(conf[m].mean()); ba.append(correct[m].mean()); cnt.append(int(m.sum()))
        else:
            bc.append((bins[i] + bins[i + 1]) / 2); ba.append(np.nan); cnt.append(0)
    return np.array(bc), np.array(ba), np.array(cnt)


# --------------------------------------------------------------------------- #
# Split-conformal prediction (LAC / softmax score)
# --------------------------------------------------------------------------- #
def conformal_qhat(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha=0.1) -> float:
    """Threshold q̂ on the nonconformity score s=1-p_true for target coverage 1-alpha."""
    n = len(cal_labels)
    scores = 1.0 - cal_probs[np.arange(n), cal_labels]
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def conformal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean (N, K) membership: class k in set iff 1-p_k <= q̂."""
    return (1.0 - probs) <= qhat


def adaptive_conformal(probs: np.ndarray, labels: np.ndarray, q0: float,
                       alpha=0.1, eta=0.02):
    """Gibbs-Candes adaptive conformal inference: online-update the threshold q_t
    on the stream so long-run set coverage tracks 1-alpha under drift.

    Returns (realized_coverage, mean_set_size, q_trace). Feedback-driven: after each
    window the realized (mis)coverage nudges q for the next one.
    """
    n = len(labels)
    scores = 1.0 - probs[np.arange(n), labels]      # true-class nonconformity
    q = q0
    covered = np.empty(n); setsize = np.empty(n); trace = np.empty(n)
    for t in range(n):
        covered[t] = scores[t] <= q                 # true label in set?
        setsize[t] = int(np.sum((1.0 - probs[t]) <= q))
        trace[t] = q
        q = q + eta * ((1 - alpha) - covered[t])    # raise q if we under-covered
        q = float(np.clip(q, 0.0, 1.0))
    return float(covered.mean()), float(setsize.mean()), trace


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 2)) * 3
    labels = (logits[:, 1] + rng.normal(scale=2, size=500) > logits[:, 0]).astype(int)
    p = softmax_np(logits)
    T = fit_temperature(logits, labels)
    pc = softmax_np(logits, T)
    print(f"T={T:.3f}  ECE {expected_calibration_error(p,labels):.4f} -> "
          f"{expected_calibration_error(pc,labels):.4f}  Brier {brier_score(pc,labels):.4f}")
    qh = conformal_qhat(pc[:250], labels[:250])
    sets = conformal_sets(pc[250:], qh)
    print(f"qhat={qh:.3f}  mean set size {sets.sum(1).mean():.2f}  "
          f"singleton frac {(sets.sum(1)==1).mean():.2f}")
