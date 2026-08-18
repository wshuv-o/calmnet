"""Abstention / selective classification: risk-coverage curves, selective risk
at fixed coverage, and confidence-vs-correctness AUROC.

The safe default action on abstain is Stop (label 0): a wrongly-executed Walk
command is the dangerous failure, so when unsure we hold."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


SAFE_ACTION = 0  # Stop / hold


def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.any():
            accs.append((y_pred[m] == c).mean())
    return float(np.mean(accs))


def confidence_auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of confidence separating correct from incorrect predictions."""
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, conf))


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray):
    """Sweep the confidence threshold; return (coverage, selective_risk) arrays.

    Points are ordered by decreasing threshold, i.e. increasing coverage.
    Selective risk = error rate among the accepted (non-abstained) samples."""
    order = np.argsort(-conf)                       # most confident first
    correct = correct[order].astype(float)
    n = len(correct)
    cum_correct = np.cumsum(correct)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k
    return coverage, risk


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(conf, correct)
    return float(np.trapezoid(risk, cov))


def selective_risk_at_coverage(conf, correct, coverage=0.8):
    """Error rate among the most-confident `coverage` fraction of samples."""
    cov, risk = risk_coverage_curve(conf, correct)
    idx = np.searchsorted(cov, coverage)
    idx = min(idx, len(risk) - 1)
    return float(risk[idx])


def abstain_decision(probs, conf, tau, conformal_set=None, motion_flag=None):
    """Return (action, executed_mask).

    Execute the argmax command only when confidence >= tau, the conformal set is
    a singleton (if provided), and no heavy head motion is flagged; otherwise
    abstain to the safe Stop action."""
    pred = probs.argmax(axis=1)
    accept = conf >= tau
    if conformal_set is not None:
        accept &= (conformal_set.sum(axis=1) == 1)
    if motion_flag is not None:
        accept &= ~motion_flag
    action = np.where(accept, pred, SAFE_ACTION)
    return action, accept


def executed_command_accuracy(y_true, probs, conf, tau, **kw):
    """Fraction of *executed* (non-abstained) commands that are correct, plus coverage."""
    _, accept = abstain_decision(probs, conf, tau, **kw)
    if accept.sum() == 0:
        return float("nan"), 0.0
    pred = probs.argmax(axis=1)
    acc = (pred[accept] == y_true[accept]).mean()
    return float(acc), float(accept.mean())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 400)
    correct = (rng.uniform(size=400) < conf).astype(int)
    print(f"AUROC {confidence_auroc(conf, correct):.3f}  AURC {aurc(conf, correct):.3f}  "
          f"risk@80% {selective_risk_at_coverage(conf, correct, 0.8):.3f}")
