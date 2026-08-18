"""Motion-Invariant Disentanglement (MID).

A CALMNet feature backbone is split into an *intent* subspace and an *artefact*
subspace. The artefact subspace is trained to predict the head-motion signal; the
intent subspace is trained -- through a gradient-reversal layer -- to be UNABLE to
predict it. The walk/stop classifier reads only the intent subspace, so the
decision is constrained to movement-invariant neural features.

We validate the central claim by probing how well each subspace predicts motion
(lower intent->motion R^2 = more invariant) and by comparing walk/stop accuracy
and the movement baseline with MID on vs off.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from models import CALMNet, EEGNet
from train import set_seed, DEVICE, class_weights
from abstain import balanced_accuracy


# --------------------------------------------------------------------------- #
# Gradient reversal
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


# --------------------------------------------------------------------------- #
# MID model
# --------------------------------------------------------------------------- #
class CALMNetMID(nn.Module):
    """MID v2: disentangle any backbone's features against a K-dim IMU vector."""
    def __init__(self, n_chan=60, n_time=200, n_classes=2, k_imu=12, p_drop=0.5,
                 backbone="calmnet", adj=None):
        super().__init__()
        if backbone == "graph":
            from graphnet import GraphEEGNet
            self.backbone = GraphEEGNet(n_chan, n_time, n_classes=n_classes, adj=adj, p_drop=p_drop)
        else:
            bb = EEGNet if backbone == "eegnet" else CALMNet
            self.backbone = bb(n_chan, n_time, n_classes=n_classes, p_drop=p_drop)
        d = self.backbone.feat_dim
        self.d_int = d // 2
        self.classify = nn.Linear(self.d_int, n_classes)                 # intent -> class
        self.art_head = nn.Sequential(nn.Linear(d - self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, k_imu))              # artefact -> IMU vec
        self.adv_head = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, 32), nn.ELU(),
                                      nn.Linear(32, k_imu))              # intent -> IMU (adversary)

    def split(self, x):
        z = self.backbone.features(x)
        return z[:, :self.d_int], z[:, self.d_int:]

    def forward(self, x, grl_lambda=1.0):
        z_int, z_art = self.split(x)
        logits = self.classify(z_int)
        art_pred = self.art_head(z_art)
        adv_pred = self.adv_head(grad_reverse(z_int, grl_lambda))
        return logits, art_pred, adv_pred, z_int

    @torch.no_grad()
    def encode(self, x):
        z_int, z_art = self.split(x)
        return z_int, z_art


def _to_t(X):
    t = torch.as_tensor(X, dtype=torch.float32)
    return t.unsqueeze(1) if t.ndim == 3 else t


def _decorr_penalty(z, m):
    """Squared Frobenius norm of the cross-covariance between z and m (batch).
    Drives linear independence of the intent code from the IMU vector."""
    zc = z - z.mean(0, keepdim=True)
    mc = m - m.mean(0, keepdim=True)
    cov = (zc.t() @ mc) / max(z.shape[0] - 1, 1)        # (d_int, k)
    return (cov ** 2).sum()


def train_mid(Xtr, ytr, Mtr, Xval, yval, *, epochs=100, lr=1e-3, wd=1e-3, batch=64,
              lam_adv=1.0, lam_art=1.0, lam_dec=1.0, patience=20, seed=0, backbone="calmnet",
              adj=None):
    """Train CALMNetMID v2. Mtr = (N, K) multimodal IMU features (standardised inside)."""
    set_seed(seed)
    Mtr = np.atleast_2d(Mtr).astype(np.float32)
    k = Mtr.shape[1]
    model = CALMNetMID(n_chan=Xtr.shape[1], k_imu=k, backbone=backbone, adj=adj).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = class_weights(ytr)
    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-6
    M_std = (Mtr - mu) / sd

    ds = TensorDataset(_to_t(Xtr), torch.as_tensor(ytr),
                       torch.as_tensor(M_std, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    best, best_score, best_ep = None, -1, 0
    for ep in range(epochs):
        p = ep / max(epochs - 1, 1)
        grl = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0          # DANN ramp
        model.train()
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            logits, art, adv, z_int = model(xb, grl_lambda=grl * lam_adv)
            loss = (F.cross_entropy(logits, yb, weight=w)
                    + lam_art * F.mse_loss(art, mb)
                    + F.mse_loss(adv, mb)                    # adversary learns; encoder (GRL) unlearns
                    + lam_dec * grl * _decorr_penalty(z_int, mb))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lg, _, _, _ = model(_to_t(Xval).to(DEVICE), grl_lambda=0.0)
            score = balanced_accuracy(yval, lg.argmax(1).cpu().numpy())
        if score > best_score:
            best_score, best, best_ep = score, {kk: v.detach().clone()
                                                for kk, v in model.state_dict().items()}, ep
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best)
    return model


@torch.no_grad()
def predict_mid(model, X, batch=256):
    model.eval()
    lg = []
    for i in range(0, len(X), batch):
        out = model(_to_t(X[i:i + batch]).to(DEVICE), grl_lambda=0.0)[0]
        lg.append(out.cpu().numpy())
    lg = np.concatenate(lg)
    e = np.exp(lg - lg.max(1, keepdims=True))
    return lg, e / e.sum(1, keepdims=True)


@torch.no_grad()
def encode_all(model, X, batch=256):
    zi, za = [], []
    for i in range(0, len(X), batch):
        a, b = model.encode(_to_t(X[i:i + batch]).to(DEVICE))
        zi.append(a.cpu().numpy()); za.append(b.cpu().numpy())
    return np.concatenate(zi), np.concatenate(za)


def motion_probe_r2(z_tr, M_tr, z_te, M_te, nonlinear=False):
    """Recover the IMU vector from a subspace (lower avg R^2 = more invariant)."""
    if nonlinear:
        est = MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=0)
    else:
        est = Ridge(alpha=1.0)
    est.fit(z_tr, M_tr)
    return float(r2_score(M_te, est.predict(z_te), multioutput="variance_weighted"))


# --------------------------------------------------------------------------- #
# HSIC (nonlinear independence) + disentanglement-aware selection, band-power model
# --------------------------------------------------------------------------- #
def _rbf(x):
    d2 = torch.cdist(x, x) ** 2
    sig2 = d2.detach().median().clamp(min=1e-6)
    return torch.exp(-d2 / (2 * sig2 + 1e-8))


def hsic(z, m):
    """Biased empirical HSIC with RBF kernels; 0 iff z, m independent."""
    B = z.shape[0]
    if B < 4:
        return z.new_zeros(())
    K, L = _rbf(z), _rbf(m)
    H = torch.eye(B, device=z.device) - 1.0 / B
    return (K @ H @ L @ H).diagonal().sum() / (B - 1) ** 2


def train_mid_hsic(Xtr, ytr, Mtr, Xval, yval, Mval, *, epochs=120, lr=1e-3, wd=1e-3, batch=64,
                   lam_adv=1.0, lam_art=1.0, lam_dec=1.0, lam_hsic=4.0, lam_sel=1.0, seed=0,
                   backbone="calmnet"):
    """MID with HSIC penalty + disentanglement-aware selection: keep the epoch
    maximising  val_bacc - lam_sel * max(0, intent->IMU linear R^2 on val)."""
    set_seed(seed)
    Mtr = np.atleast_2d(Mtr).astype(np.float32); Mval = np.atleast_2d(Mval).astype(np.float32)
    model = CALMNetMID(n_chan=Xtr.shape[1], k_imu=Mtr.shape[1], backbone=backbone).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = class_weights(ytr)
    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-6
    Mf, Mv = (Mtr - mu) / sd, (Mval - mu) / sd
    ds = TensorDataset(_to_t(Xtr), torch.as_tensor(ytr), torch.as_tensor(Mf, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    nv = len(Xval); vf = np.arange(nv) % 10 < 7                     # 70/30 split of val for the probe
    best, best_score = None, -1e9
    for ep in range(epochs):
        grl = 2.0 / (1.0 + np.exp(-10 * ep / max(epochs - 1, 1))) - 1.0
        model.train()
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            logits, art, adv, z_int = model(xb, grl_lambda=grl * lam_adv)
            loss = (F.cross_entropy(logits, yb, weight=w) + lam_art * F.mse_loss(art, mb)
                    + F.mse_loss(adv, mb) + grl * (lam_dec * _decorr_penalty(z_int, mb)
                                                   + lam_hsic * hsic(z_int, mb)))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(_to_t(Xval).to(DEVICE), grl_lambda=0.0)[0]
            bacc = balanced_accuracy(yval, lg.argmax(1).cpu().numpy())
        zi = encode_all(model, Xval)[0]                            # cheap: only val encoded
        r2 = r2_score(Mv[~vf], Ridge(alpha=1.0).fit(zi[vf], Mv[vf]).predict(zi[~vf]),
                      multioutput="variance_weighted")
        score = bacc - lam_sel * max(0.0, r2)
        if score > best_score:
            best_score = score
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model
