"""Shared training / inference loop for the Walk/Stop decoders."""
from __future__ import annotations
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import build_model, enable_mc_dropout
from abstain import balanced_accuracy

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(X):
    t = torch.as_tensor(X, dtype=torch.float32)
    return t.unsqueeze(1) if t.ndim == 3 else t          # (N,1,C,T)


def _loader(X, y, batch=64, shuffle=True):
    ds = TensorDataset(_to_tensor(X), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def class_weights(y, n_classes=2):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_model(name, Xtr, ytr, Xval=None, yval=None, *, epochs=100, lr=1e-3,
                weight_decay=1e-3, batch=64, patience=20, seed=0, verbose=False,
                model_kw=None):
    """Train a model with class-weighted CE and early stopping on val balanced-acc."""
    set_seed(seed)
    model = build_model(name, **(model_kw or {})).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights(ytr)
    tr = _loader(Xtr, ytr, batch, shuffle=True)

    has_val = Xval is not None and len(Xval) > 0
    best_state, best_score, best_epoch = None, -1.0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w)
            loss.backward()
            opt.step()

        if has_val:
            _, pred, _ = predict(model, Xval)
            score = balanced_accuracy(yval, pred)
        else:
            score = -float(loss.item())
        if score > best_score:
            best_score, best_state, best_epoch = score, {k: v.detach().clone()
                                                          for k, v in model.state_dict().items()}, ep
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    ep{ep:3d} loss {loss.item():.3f} val_bacc {score:.3f}")
        if has_val and ep - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, batch=256):
    """Return (logits, pred, probs) with dropout OFF."""
    model.eval()
    logits = []
    for i in range(0, len(X), batch):
        xb = _to_tensor(X[i:i + batch]).to(DEVICE)
        logits.append(model(xb).cpu().numpy())
    logits = np.concatenate(logits)
    probs = _softmax(logits)
    return logits, probs.argmax(1), probs


@torch.no_grad()
def predict_mc(model, X, n_samples=30, batch=256):
    """MC-dropout: mean probs and predictive entropy for epistemic uncertainty."""
    enable_mc_dropout(model)
    acc = np.zeros((len(X), 2))
    for _ in range(n_samples):
        p = []
        for i in range(0, len(X), batch):
            xb = _to_tensor(X[i:i + batch]).to(DEVICE)
            p.append(_softmax(model(xb).cpu().numpy()))
        acc += np.concatenate(p)
    probs = acc / n_samples
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)
    return probs, entropy


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
