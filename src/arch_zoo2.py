"""Extensions to the architecture zoo: alternative backbones, readouts,
augmentations, independence penalties and optimisers.

Adds axes the first 24-variant sweep did not cover, so that "we searched the
architecture space" means something wider than knob-twiddling on one backbone:

  backbones   ShallowConvNet / DeepConvNet style, EEGNet, temporal-CNN
  readouts    GRU, max, std, log-var, first+last token
  augment     channel dropout, time masking, additive noise, time shift, scaling
  penalties   MMD, orthogonality, CORAL, in addition to HSIC / cross-covariance
  optimisers  SGD-momentum, RMSprop, AdamW with warmup

Everything is wrapped in the same GenericMID head set so the invariance probe
remains comparable across every variant.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #
class GRUReadout(nn.Module):
    """Sequence model over the time tokens -- exploits temporal persistence of
    the walk/stop state rather than treating tokens as an unordered bag."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, dim)

    def forward(self, tok):
        h, _ = self.gru(tok)
        return self.proj(h.mean(1))


class StatPool(nn.Module):
    """Concatenate mean and std across tokens, then project back to dim.
    Band-power variance over the window is itself informative."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, tok):
        return self.proj(torch.cat([tok.mean(1), tok.std(1)], dim=-1))


class MaxPoolReadout(nn.Module):
    def forward(self, tok):
        return tok.max(dim=1).values


class EdgePool(nn.Module):
    """First and last token -- sensitive to onset/offset structure."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, tok):
        return self.proj(torch.cat([tok[:, 0], tok[:, -1]], dim=-1))


def make_readout(kind, dim, heads=4, p_drop=0.3):
    from arch_zoo import AttnPool, MHAPool
    if kind == "attn":
        return AttnPool(dim)
    if kind == "mha":
        return MHAPool(dim, heads, p_drop)
    if kind == "gru":
        return GRUReadout(dim)
    if kind == "stat":
        return StatPool(dim)
    if kind == "max":
        return MaxPoolReadout()
    if kind == "edge":
        return EdgePool(dim)
    return None                                  # plain mean


# --------------------------------------------------------------------------- #
# Alternative backbones (all expose .features() and .feat_dim)
# --------------------------------------------------------------------------- #
class ShallowNet(nn.Module):
    """ShallowConvNet-style: single wide temporal conv -> spatial -> square ->
    mean-pool -> log. The classical FBCSP-flavoured MI baseline."""

    def __init__(self, n_chan=60, n_time=200, F_=40, k=25, pool=45, stride=10,
                 p_drop=0.5, **kw):
        super().__init__()
        self.temporal = nn.Conv2d(1, F_, (1, k), bias=False)
        self.spatial = nn.Conv2d(F_, F_, (n_chan, 1), bias=False)
        self.bn = nn.BatchNorm2d(F_)
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(p_drop)
        self.norm_out = nn.LayerNorm(F_)
        self.feat_dim = F_

    def features(self, x):
        x = self.spatial(self.temporal(x))
        x = self.bn(x)
        x = torch.log(torch.clamp(self.pool(x ** 2), min=1e-6))
        x = self.drop(x)
        return self.norm_out(x.squeeze(2).transpose(1, 2).mean(1))


class DeepNet(nn.Module):
    """DeepConvNet-style stacked temporal blocks."""

    def __init__(self, n_chan=60, n_time=200, F_=25, p_drop=0.5, blocks=3, **kw):
        super().__init__()
        self.temporal = nn.Conv2d(1, F_, (1, 10), bias=False)
        self.spatial = nn.Conv2d(F_, F_, (n_chan, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(F_)
        # track the time dimension so a deep stack cannot pool it below 1
        t = n_time - 9                                   # after the temporal conv
        layers, ch = [], F_
        for _ in range(blocks):
            if t - 9 < 2:                                # no room for another block
                break
            t -= 9
            layers += [nn.Conv2d(ch, ch * 2, (1, 10), bias=False),
                       nn.BatchNorm2d(ch * 2), nn.ELU()]
            ch *= 2
            if t // 3 >= 2:
                layers += [nn.MaxPool2d((1, 3), (1, 3))]
                t //= 3
            layers += [nn.Dropout(p_drop)]
        self.body = nn.Sequential(*layers)
        self.norm_out = nn.LayerNorm(ch)
        self.feat_dim = ch

    def features(self, x):
        x = F.elu(self.bn0(self.spatial(self.temporal(x))))
        x = self.body(x)
        return self.norm_out(x.squeeze(2).mean(-1))


class TCNNet(nn.Module):
    """Dilated temporal conv stack on top of a spatial projection."""

    def __init__(self, n_chan=60, n_time=200, F_=32, levels=4, p_drop=0.4, **kw):
        super().__init__()
        self.spatial = nn.Conv2d(1, F_, (n_chan, 1), bias=False)
        self.bn = nn.BatchNorm2d(F_)
        blocks = []
        for i in range(levels):
            d = 2 ** i
            blocks.append(nn.Sequential(
                nn.Conv1d(F_, F_, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(F_),
                nn.ELU(), nn.Dropout(p_drop)))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(F_)
        self.feat_dim = F_

    def features(self, x):
        x = self.bn(self.spatial(x)).squeeze(2)      # (B, F, T)
        for b in self.blocks:
            x = x + b(x)[..., :x.shape[-1]]
        return self.norm_out(x.mean(-1))


# --------------------------------------------------------------------------- #
# Augmentations (applied to (B,1,C,T) batches during training only)
# --------------------------------------------------------------------------- #
def augment_batch(x, kind, strength=0.1):
    if kind == "none" or not kind:
        return x
    B, _, C, T = x.shape
    if kind == "chan_drop":
        m = (torch.rand(B, 1, C, 1, device=x.device) > strength).float()
        return x * m
    if kind == "time_mask":
        w = max(1, int(T * strength))
        s = torch.randint(0, max(1, T - w), (1,)).item()
        y = x.clone(); y[..., s:s + w] = 0
        return y
    if kind == "noise":
        return x + torch.randn_like(x) * strength
    if kind == "shift":
        s = int(np.random.uniform(-strength, strength) * T)
        return torch.roll(x, shifts=s, dims=-1)
    if kind == "scale":
        f = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2 * strength
        return x * f
    return x


# --------------------------------------------------------------------------- #
# Independence penalties between the intent code and the IMU vector
# --------------------------------------------------------------------------- #
def mmd_penalty(z, m):
    """RBF-kernel MMD between the joint and the product of marginals, approximated
    by comparing z against a permuted-m pairing."""
    perm = torch.randperm(m.shape[0], device=m.device)
    a = torch.cat([z, m], 1)
    b = torch.cat([z, m[perm]], 1)

    def k(u, v):
        d = torch.cdist(u, v) ** 2
        s = d.detach().median().clamp(min=1e-6)
        return torch.exp(-d / (2 * s + 1e-8))

    return k(a, a).mean() - 2 * k(a, b).mean() + k(b, b).mean()


def ortho_penalty(z, m):
    """Force the intent code orthogonal to the IMU vector after centring."""
    zc = z - z.mean(0, keepdim=True)
    mc = m - m.mean(0, keepdim=True)
    zc = zc / (zc.norm(dim=0, keepdim=True) + 1e-6)
    mc = mc / (mc.norm(dim=0, keepdim=True) + 1e-6)
    return ((zc.t() @ mc) ** 2).mean()


def coral_penalty(z, m):
    """Match second-order statistics: penalise the difference between the
    covariance of z and that of a movement-permuted reference."""
    def cov(u):
        uc = u - u.mean(0, keepdim=True)
        return (uc.t() @ uc) / max(u.shape[0] - 1, 1)
    d = min(z.shape[1], m.shape[1])
    return ((cov(z[:, :d]) - cov(m[:, :d])) ** 2).mean()


PENALTIES = {"mmd": mmd_penalty, "ortho": ortho_penalty, "coral": coral_penalty}


def make_optimizer(kind, params, lr, wd):
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr * 10, momentum=0.9, weight_decay=wd)
    if kind == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
