"""Full CALM-Net encoder: multi-band spatial covariance -> SPD tangent space ->
cross-frequency coupling attention (XFCA), with MID heads and source-free CORAL
alignment. This realises the architecture described in the paper (vs the compact
band-power instantiation used earlier).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from train import set_seed, DEVICE, class_weights
from abstain import balanced_accuracy
from mid import grad_reverse, _decorr_penalty

BANDS = [(8, 13), (13, 20), (20, 30)]          # mu, low-beta, high-beta
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


# --------------------------------------------------------------------------- #
# Multi-band SPD tangent-space features (precomputed, cached per subject)
# --------------------------------------------------------------------------- #
def _bandpass(X, lo, hi, sf=100.0, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=sf, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def _tangent(Xb, eps=1e-5):
    """Log-Euclidean tangent vector of the per-epoch spatial covariance.
    Xb: (N, C, T) -> (N, C(C+1)/2)."""
    N, C, T = Xb.shape
    Xb = Xb - Xb.mean(-1, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", Xb, Xb) / T + eps * np.eye(C, dtype=np.float32)
    ev, V = np.linalg.eigh(cov)                                   # ev (N,C), V (N,C,C)
    logcov = np.einsum("nck,nk,ndk->ncd", V, np.log(np.clip(ev, 1e-6, None)), V)
    w = np.ones((C, C), dtype=np.float32)
    w[np.triu_indices(C, 1)] = np.sqrt(2.0)                       # norm-preserving off-diag
    iu = np.triu_indices(C)
    return (logcov * w)[:, iu[0], iu[1]].astype(np.float32)


def tangent_features(es, subject, use_cache=True):
    """Return (N, n_bands, tri_dim) multi-band tangent features."""
    path = CACHE / f"tangent_{subject}.npz"
    if use_cache and path.exists():
        return np.load(path)["V"]
    feats = [_tangent(_bandpass(es.X, lo, hi)) for lo, hi in BANDS]
    V = np.stack(feats, axis=1).astype(np.float32)               # (N, nb, tri)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, V=V)
    return V


def coral_standardise(V, mu, sd):
    return ((V - mu) / sd).astype(np.float32)


def session_coral(V_session):
    """Source-free diagonal CORAL: align a test session to the standard (train)
    distribution using only its own second-order statistics."""
    mu = V_session.mean(0, keepdims=True)
    sd = V_session.std(0, keepdims=True) + 1e-6
    return ((V_session - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# XFCA + MID model over the band tokens
# --------------------------------------------------------------------------- #
class BandXFCAMID(nn.Module):
    def __init__(self, tri_dim, n_bands=3, d=48, heads=4, n_classes=2, k_imu=12, p_drop=0.5):
        super().__init__()
        self.n_bands = n_bands
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(tri_dim, d), nn.LayerNorm(d),
                                                 nn.ELU(), nn.Dropout(p_drop)) for _ in range(n_bands)])
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=p_drop)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p_drop)
        self.d_int = d // 2
        self.classify = nn.Linear(self.d_int, n_classes)
        self.art_head = nn.Sequential(nn.Linear(d - self.d_int, 32), nn.ELU(), nn.Linear(32, k_imu))
        self.adv_head = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, 32), nn.ELU(), nn.Linear(32, k_imu))

    def features(self, v):                                        # v: (B, nb, tri)
        toks = torch.stack([self.proj[b](v[:, b]) for b in range(self.n_bands)], dim=1)
        att, _ = self.attn(toks, toks, toks)                     # cross-frequency coupling
        return self.norm(self.drop(att.mean(1)))

    def split(self, v):
        z = self.features(v)
        return z[:, :self.d_int], z[:, self.d_int:]

    def forward(self, v, grl_lambda=1.0):
        z_int, z_art = self.split(v)
        return (self.classify(z_int), self.art_head(z_art),
                self.adv_head(grad_reverse(z_int, grl_lambda)), z_int)

    @torch.no_grad()
    def encode(self, v):
        z_int, z_art = self.split(v)
        return z_int, z_art


def _t(V):
    return torch.as_tensor(V, dtype=torch.float32)


def train_riemann(Vtr, ytr, Mtr, Vval, yval, *, epochs=120, lr=1e-3, wd=1e-2, batch=64,
                  lam_adv=1.0, lam_art=1.0, lam_dec=1.0, patience=25, seed=0):
    set_seed(seed)
    Mtr = np.atleast_2d(Mtr).astype(np.float32)
    model = BandXFCAMID(Vtr.shape[2], n_bands=Vtr.shape[1], k_imu=Mtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = class_weights(ytr)
    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-6
    M_std = (Mtr - mu) / sd
    ds = TensorDataset(_t(Vtr), torch.as_tensor(ytr), torch.as_tensor(M_std, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    best, best_score, best_ep = None, -1, 0
    for ep in range(epochs):
        grl = 2.0 / (1.0 + np.exp(-10 * ep / max(epochs - 1, 1))) - 1.0
        model.train()
        for vb, yb, mb in dl:
            vb, yb, mb = vb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            logits, art, adv, z_int = model(vb, grl_lambda=grl * lam_adv)
            loss = (F.cross_entropy(logits, yb, weight=w)
                    + lam_art * F.mse_loss(art, mb) + F.mse_loss(adv, mb)
                    + lam_dec * grl * _decorr_penalty(z_int, mb))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(_t(Vval).to(DEVICE), grl_lambda=0.0)[0]
            sc = balanced_accuracy(yval, lg.argmax(1).cpu().numpy())
        if sc > best_score:
            best_score, best, best_ep = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best)
    return model


@torch.no_grad()
def predict_riemann(model, V, batch=256):
    model.eval(); lg = []
    for i in range(0, len(V), batch):
        lg.append(model(_t(V[i:i+batch]).to(DEVICE), grl_lambda=0.0)[0].cpu().numpy())
    lg = np.concatenate(lg)
    e = np.exp(lg - lg.max(1, keepdims=True))
    return lg, e / e.sum(1, keepdims=True)


@torch.no_grad()
def encode_riemann(model, V, batch=256):
    zi = []
    for i in range(0, len(V), batch):
        a, _ = model.encode(_t(V[i:i+batch]).to(DEVICE)); zi.append(a.cpu().numpy())
    return np.concatenate(zi)


# --------------------------------------------------------------------------- #
# Improved disentanglement: HSIC (nonlinear independence) + selection fix
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


def train_riemann_v2(Vfit, yfit, Mfit, Vval, yval, Mval, *, epochs=120, lr=1e-3, wd=1e-2,
                     batch=64, lam_adv=1.0, lam_art=1.0, lam_dec=1.0, lam_hsic=4.0,
                     lam_sel=1.0, seed=0):
    """Riemannian MID with an added HSIC penalty and disentanglement-aware selection:
    pick the epoch maximising  val_bacc - lam_sel * max(0, intent->IMU linear R^2)."""
    set_seed(seed)
    Mfit = np.atleast_2d(Mfit).astype(np.float32); Mval = np.atleast_2d(Mval).astype(np.float32)
    model = BandXFCAMID(Vfit.shape[2], n_bands=Vfit.shape[1], k_imu=Mfit.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = class_weights(yfit)
    mu, sd = Mfit.mean(0), Mfit.std(0) + 1e-6
    Mf_std, Mv_std = (Mfit - mu) / sd, (Mval - mu) / sd
    ds = TensorDataset(_t(Vfit), torch.as_tensor(yfit), torch.as_tensor(Mf_std, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    best, best_score = None, -1e9
    for ep in range(epochs):
        grl = 2.0 / (1.0 + np.exp(-10 * ep / max(epochs - 1, 1))) - 1.0
        model.train()
        for vb, yb, mb in dl:
            vb, yb, mb = vb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            logits, art, adv, z_int = model(vb, grl_lambda=grl * lam_adv)
            loss = (F.cross_entropy(logits, yb, weight=w) + lam_art * F.mse_loss(art, mb)
                    + F.mse_loss(adv, mb) + grl * (lam_dec * _decorr_penalty(z_int, mb)
                                                   + lam_hsic * hsic(z_int, mb)))
            loss.backward(); opt.step()
        # disentanglement-aware selection
        model.eval()
        with torch.no_grad():
            lg = model(_t(Vval).to(DEVICE), grl_lambda=0.0)[0]
            bacc = balanced_accuracy(yval, lg.argmax(1).cpu().numpy())
        zi_f, zi_v = encode_riemann(model, Vfit), encode_riemann(model, Vval)
        r2 = r2_score(Mv_std, Ridge(alpha=1.0).fit(zi_f, Mf_std).predict(zi_v),
                      multioutput="variance_weighted")
        score = bacc - lam_sel * max(0.0, r2)
        if score > best_score:
            best_score = score
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model
