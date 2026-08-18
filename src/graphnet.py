"""Spatio-temporal graph decoder over the real scalp electrode geometry.

Every baseline here (EEGNet, ShallowConvNet, our band-power net) mixes channels with a
DENSE spatial convolution: the 60 electrodes are an unordered list and the scalp layout
is discarded. This model instead builds a graph from the dataset's own 3D electrode
coordinates (electrodes.tsv) and propagates over it, so spatial mixing is constrained
by actual scalp adjacency, then reads out with an attention over electrodes (which also
says WHICH electrodes drive the decode).

  temporal filter bank -> graph diffusion (scalp geometry) -> electrode attention
  -> log band-power -> temporal attention -> logits
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataio import DATA_ROOT
from models import _AttentionPool


def electrode_adjacency(ch_names, subject="sub-01", ses=1, k=8):
    """Normalised adjacency (C,C) from the dataset's 3D electrode coordinates.
    kNN graph on the scalp with Gaussian edge weights, self-loops, GCN normalisation."""
    tsv = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg" / f"{subject}_ses-{ses:02d}_electrodes.tsv"
    df = pd.read_csv(tsv, sep="\t")
    pos = {str(r["name"]): np.array([r["x"], r["y"], r["z"]], float) for _, r in df.iterrows()}
    missing = [c for c in ch_names if c not in pos]
    if missing:
        raise KeyError(f"electrodes.tsv missing {len(missing)} channels e.g. {missing[:3]}")
    P = np.stack([pos[c] for c in ch_names])                      # (C,3)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)    # pairwise distance
    sigma = np.median(D[D > 0])
    W = np.exp(-(D ** 2) / (2 * sigma ** 2))
    # keep only k nearest neighbours per node
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    mask = np.zeros_like(W); np.put_along_axis(mask, idx, 1.0, axis=1)
    W = W * mask
    W = np.maximum(W, W.T)                                        # symmetric
    W = W + np.eye(len(ch_names))                                 # self-loops
    d = W.sum(1)
    A = W / np.sqrt(np.outer(d, d))                               # D^-1/2 W D^-1/2
    return torch.tensor(A, dtype=torch.float32)


class GraphEEGNet(nn.Module):
    def __init__(self, n_chan=60, n_time=200, n_classes=2, adj=None, F_=8,
                 kernels=(13, 25, 51), gsteps=2, pool=25, stride=5, p_drop=0.5, n_heads=4):
        super().__init__()
        if adj is None:
            adj = torch.eye(n_chan)
        self.register_buffer("A", adj)
        self.branches = nn.ModuleList([
            nn.Conv2d(1, F_, (1, k), padding=(0, k // 2), bias=False) for k in kernels])
        Fc = F_ * len(kernels)
        self.bn_t = nn.BatchNorm2d(Fc)
        self.gtrans = nn.ModuleList([nn.Conv2d(Fc, Fc, (1, 1), bias=False) for _ in range(gsteps)])
        self.bn_g = nn.BatchNorm2d(Fc)
        # v3: SIGNED spatial readout (graph-regularised CSP).
        # v1/v2 used a softmax over electrodes -> a convex combination (non-negative,
        # sums to 1) -> it can only AVERAGE, never CONTRAST. Motor imagery needs signed
        # spatial filters (C3 minus surround), so softmax was structurally wrong; the
        # learned weights collapsed to ~uniform (0.018 ~ 1/60). Here the graph diffusion
        # supplies the geometry prior and an unconstrained depthwise conv does the
        # contrast, one bank of n_heads signed filters per temporal feature.
        self.n_heads = n_heads
        Fk = Fc * n_heads
        self.spatial = nn.Conv2d(Fc, Fk, (n_chan, 1), groups=Fc, bias=False)
        self.bn_s = nn.BatchNorm2d(Fk)
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(p_drop)
        self.attn = _AttentionPool(Fk)
        self.norm = nn.LayerNorm(Fk)
        self.feat_dim = Fk
        self.classify = nn.Linear(Fk, n_classes)

    def features(self, x):                                         # (B,1,C,T)
        x = torch.cat([b(x) for b in self.branches], dim=1)        # (B,Fc,C,T)
        x = self.bn_t(x)
        for gt in self.gtrans:                                     # graph diffusion on the scalp
            x = torch.einsum("cd,bfdt->bfct", self.A, x)
            x = F.elu(gt(x))
        x = self.bn_g(x)
        x = self.spatial(x)                                        # signed spatial filters -> (B,Fk,1,T)
        x = self.bn_s(x)
        x = torch.log(torch.clamp(self.pool(x ** 2), min=1e-6))    # log band-power
        x = self.drop(x)
        tokens = x.squeeze(2).transpose(1, 2)                      # (B,Tok,Fc*K)
        return self.norm(self.attn(tokens))

    def forward(self, x):
        return self.classify(self.features(x))

    def top_electrodes(self, ch_names, n=8):
        """Electrodes the signed spatial filters rely on most (mean |weight|)."""
        w = self.spatial.weight.detach().cpu().abs().mean(dim=(0, 1, 3)).numpy()
        w = w / (w.sum() + 1e-9)
        order = np.argsort(-w)[:n]
        return [(ch_names[i], float(w[i])) for i in order]
