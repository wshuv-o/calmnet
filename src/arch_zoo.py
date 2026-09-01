"""Configurable architecture zoo for CALM-Net variants.

One flexible band-power backbone with independent toggles (filter bank, spatial
expansion, squeeze-excitation, residual path, pooling/readout, normalisation,
classifier head), wrapped in the standard MID head set so that every variant is
measured the same way: walk/stop accuracy AND how well the 12-D IMU vector can
still be recovered from the intent subspace.

Accuracy alone is not a valid score on this dataset -- the label is partly a
movement label, so a stronger encoder can raise accuracy purely by leaking
movement. A variant only counts as an improvement if accuracy rises while
intent->IMU R^2 stays <= 0.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mid import grad_reverse, _decorr_penalty, hsic
from train import set_seed, DEVICE
from abstain import balanced_accuracy


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class SqueezeExcite(nn.Module):
    """Channel-wise recalibration over the feature-map axis."""

    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(ch // r, 2)), nn.ELU(),
                                nn.Linear(max(ch // r, 2), ch), nn.Sigmoid())

    def forward(self, x):                       # (B, C, 1, T)
        w = self.fc(x.mean(dim=(2, 3)))         # (B, C)
        return x * w[:, :, None, None]


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tok):                     # (B, Tok, C)
        w = torch.softmax(self.score(tok), dim=1)
        return (w * tok).sum(1)


class MHAPool(nn.Module):
    """Self-attention across time tokens, then mean. Lets distant sub-windows
    interact instead of being scored independently."""

    def __init__(self, dim, heads=4, p_drop=0.3):
        super().__init__()
        heads = max(1, min(heads, dim // 8))
        while dim % heads:
            heads -= 1
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=p_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tok):
        a, _ = self.attn(tok, tok, tok)
        return self.norm(a + tok).mean(1)


class FlexNet(nn.Module):
    """Band-power backbone with independent architectural toggles.

    multi-scale temporal filter bank -> grouped spatial conv (CSP-like)
    -> [SE] -> square -> avg-pool -> log  -> [readout] -> features
    """

    def __init__(self, n_chan=60, n_time=200, n_classes=2,
                 kernels=(13, 25, 51), F_=8, D=2, pool=25, stride=5,
                 p_drop=0.5, se=False, readout="attn", norm="bn",
                 residual=False, heads=4):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(1, F_, (1, k), padding=(0, k // 2), bias=False) for k in kernels])
        Fc = F_ * len(kernels)
        self.bn_t = nn.BatchNorm2d(Fc)
        self.spatial = nn.Conv2d(Fc, Fc * D, (n_chan, 1), groups=Fc, bias=False)
        self.bn_s = nn.BatchNorm2d(Fc * D) if norm == "bn" else nn.GroupNorm(
            max(1, (Fc * D) // 8), Fc * D)
        self.se = SqueezeExcite(Fc * D) if se else None
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(p_drop)
        self.residual = residual

        dim = Fc * D
        self.readout_kind = readout
        from arch_zoo2 import make_readout
        self.readout = make_readout(readout, dim, heads, p_drop)
        self.norm_out = nn.LayerNorm(dim)
        self.feat_dim = dim

    def _tokens(self, x):
        x = torch.cat([b(x) for b in self.branches], dim=1)
        x = self.bn_t(x)
        x = self.spatial(x)
        x = self.bn_s(x)
        if self.se is not None:
            x = self.se(x)
        p = x ** 2
        p = self.pool(p)
        p = torch.log(torch.clamp(p, min=1e-6))
        if self.residual:
            # log-power of the un-normalised branch, same shape, as a skip path
            p = p + torch.log(torch.clamp(self.pool(x.detach() ** 2), min=1e-6)) * 0.1
        p = self.drop(p)
        return p.squeeze(2).transpose(1, 2)      # (B, Tok, dim)

    def features(self, x):
        tok = self._tokens(x)
        z = tok.mean(1) if self.readout is None else self.readout(tok)
        return self.norm_out(z)

    def forward(self, x):
        raise NotImplementedError("use GenericMID")


# --------------------------------------------------------------------------- #
# MID wrapper around any backbone exposing .features() and .feat_dim
# --------------------------------------------------------------------------- #
class GenericMID(nn.Module):
    def __init__(self, backbone, n_classes=2, k_imu=12, head="linear", int_frac=0.5):
        super().__init__()
        self.backbone = backbone
        d = backbone.feat_dim
        self.d_int = max(2, int(round(d * int_frac)))
        d_art = d - self.d_int
        if head == "mlp":
            self.classify = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                          nn.Dropout(0.3), nn.Linear(32, n_classes))
        else:
            self.classify = nn.Linear(self.d_int, n_classes)
        self.art_head = nn.Sequential(nn.Linear(d_art, 32), nn.ELU(), nn.Linear(32, k_imu))
        self.adv_head = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, 32), nn.ELU(), nn.Linear(32, k_imu))

    def split(self, x):
        z = self.backbone.features(x)
        return z[:, :self.d_int], z[:, self.d_int:]

    def forward(self, x, grl=1.0):
        zi, za = self.split(x)
        return self.classify(zi), self.art_head(za), self.adv_head(grad_reverse(zi, grl)), zi

    @torch.no_grad()
    def encode(self, x):
        return self.split(x)


def _t(a, dtype=torch.float32):
    t = torch.as_tensor(a, dtype=dtype)
    return t.unsqueeze(1) if (dtype == torch.float32 and t.ndim == 3) else t


def build_variant(cfg, n_chan=60, n_time=200, k_imu=12):
    kind = cfg.get("backbone", "flex")
    if isinstance(kind, str) and kind.startswith("bd:"):
        from braindecode_zoo import build_bd_variant
        return build_bd_variant(kind[3:], n_chan=n_chan, n_time=n_time, k_imu=k_imu,
                                head=cfg.get("head", "linear"),
                                int_frac=cfg.get("int_frac", 0.5))
    if kind != "flex":
        from arch_zoo2 import ShallowNet, DeepNet, TCNNet
        if kind == "shallow":
            bb = ShallowNet(n_chan, n_time, F_=cfg.get("F", 40), p_drop=cfg.get("p_drop", 0.5))
        elif kind == "deep":
            bb = DeepNet(n_chan, n_time, F_=cfg.get("F", 25), p_drop=cfg.get("p_drop", 0.5),
                         blocks=cfg.get("blocks", 3))
        elif kind == "tcn":
            bb = TCNNet(n_chan, n_time, F_=cfg.get("F", 32), levels=cfg.get("levels", 4),
                        p_drop=cfg.get("p_drop", 0.4))
        elif kind == "eegnet":
            from models import EEGNet
            bb = EEGNet(n_chan, n_time, p_drop=cfg.get("p_drop", 0.5))
        else:
            raise ValueError(f"unknown backbone {kind}")
        return GenericMID(bb, k_imu=k_imu, head=cfg.get("head", "linear"),
                          int_frac=cfg.get("int_frac", 0.5))
    bb = FlexNet(n_chan=n_chan, n_time=n_time,
                 kernels=cfg.get("kernels", (13, 25, 51)),
                 F_=cfg.get("F", 8), D=cfg.get("D", 2),
                 pool=cfg.get("pool", 25), stride=cfg.get("stride", 5),
                 p_drop=cfg.get("p_drop", 0.5), se=cfg.get("se", False),
                 readout=cfg.get("readout", "attn"), norm=cfg.get("norm", "bn"),
                 residual=cfg.get("residual", False), heads=cfg.get("heads", 4))
    return GenericMID(bb, k_imu=k_imu, head=cfg.get("head", "linear"),
                      int_frac=cfg.get("int_frac", 0.5))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_variant(cfg, Xtr, ytr, Mtr, Xval, yval, *, epochs=80, patience=20, seed=0):
    """Train one variant. Honours cfg toggles: lr, wd, batch, lam_adv, lam_art,
    lam_dec, lam_hsic, label_smooth, mixup, and disentanglement-aware selection."""
    set_seed(seed)
    Mtr = np.atleast_2d(Mtr).astype(np.float32)
    model = build_variant(cfg, n_chan=Xtr.shape[1], n_time=Xtr.shape[2],
                          k_imu=Mtr.shape[1]).to(DEVICE)
    lr = cfg.get("lr", 1e-3); wd = cfg.get("wd", 1e-3); batch = cfg.get("batch", 64)
    from arch_zoo2 import make_optimizer, augment_batch, PENALTIES
    opt = make_optimizer(cfg.get('optim','adamw'), model.parameters(), lr, wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cfg.get("cosine", False) else None)

    cnt = np.bincount(ytr, minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)),
                     dtype=torch.float32, device=DEVICE)

    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-6
    Mst = ((Mtr - mu) / sd).astype(np.float32)
    dl = DataLoader(TensorDataset(_t(Xtr), _t(ytr, torch.long), _t(Mst)),
                    batch_size=batch, shuffle=True)

    lam_adv = cfg.get("lam_adv", 1.0); lam_art = cfg.get("lam_art", 1.0)
    lam_dec = cfg.get("lam_dec", 1.0); lam_hsic = cfg.get("lam_hsic", 0.0)
    ls = cfg.get("label_smooth", 0.0); mixup = cfg.get("mixup", 0.0)

    best, best_score, best_ep = None, -1e9, 0
    for ep in range(epochs):
        grl = 2.0 / (1.0 + np.exp(-10 * ep / max(epochs - 1, 1))) - 1.0
        model.train()
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            xb = augment_batch(xb, cfg.get("augment"), cfg.get("aug_strength", 0.1))
            opt.zero_grad()
            if mixup > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(mixup, mixup))
                perm = torch.randperm(xb.size(0), device=DEVICE)
                xb = lam * xb + (1 - lam) * xb[perm]
                logits, art, adv, zi = model(xb, grl * lam_adv)
                cls = (lam * F.cross_entropy(logits, yb, weight=w, label_smoothing=ls)
                       + (1 - lam) * F.cross_entropy(logits, yb[perm], weight=w,
                                                     label_smoothing=ls))
            else:
                logits, art, adv, zi = model(xb, grl * lam_adv)
                cls = F.cross_entropy(logits, yb, weight=w, label_smoothing=ls)

            loss = cls + lam_art * F.mse_loss(art, mb) + F.mse_loss(adv, mb)
            loss = loss + grl * lam_dec * _decorr_penalty(zi, mb)
            if lam_hsic > 0:
                loss = loss + grl * lam_hsic * hsic(zi, mb)
            pen = cfg.get("penalty")
            if pen in PENALTIES and cfg.get("lam_pen", 0.0) > 0:
                loss = loss + grl * cfg["lam_pen"] * PENALTIES[pen](zi, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
            opt.step()
        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            lg = model(_t(Xval).to(DEVICE), 0.0)[0]
        bacc = balanced_accuracy(yval, lg.argmax(1).cpu().numpy())
        score = bacc
        if cfg.get("disent_select", False):
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score
            zi_v = encode_variant(model, Xval)
            Mv = ((np.atleast_2d(cfg["_Mval"]) - mu) / sd).astype(np.float32)
            h = np.arange(len(zi_v)) % 10 < 7
            try:
                r2v = r2_score(Mv[~h], Ridge(alpha=1.0).fit(zi_v[h], Mv[h]).predict(zi_v[~h]),
                               multioutput="variance_weighted")
            except Exception:
                r2v = 0.0
            score = bacc - cfg.get("lam_sel", 1.0) * max(0.0, r2v)

        if score > best_score:
            best_score, best_ep = score, ep
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best)
    return model, (mu, sd)


@torch.no_grad()
def predict_variant(model, X, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(_t(X[i:i + batch]).to(DEVICE), 0.0)[0].cpu().numpy())
    lg = np.concatenate(out)
    e = np.exp(lg - lg.max(1, keepdims=True))
    return lg, e / e.sum(1, keepdims=True)


@torch.no_grad()
def encode_variant(model, X, batch=512):
    model.eval()
    zs = []
    for i in range(0, len(X), batch):
        zs.append(model.encode(_t(X[i:i + batch]).to(DEVICE))[0].cpu().numpy())
    return np.concatenate(zs)


def n_params(cfg, n_chan=60, n_time=200):
    m = build_variant(cfg, n_chan=n_chan, n_time=n_time)
    return sum(p.numel() for p in m.parameters())
