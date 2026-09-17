"""ATCNet+ : the best published decoder here, with modules added and measured.

ATCNet (Altaheri et al. 2022) is the strongest architecture on this task once
the training set is large enough -- 0.881 at 4881 fit windows against 0.828 at
905, the largest data-driven gain of any model tested. It is used here UNMODIFIED
as the backbone; every addition is a switchable module so each one can be made to
earn its place or be dropped.

Each module exists because a specific measurement demanded it, and each targets a
specific one of the five reported outcomes.

  rate   -- ATCNet's temporal hyperparameters are specified in SAMPLES and tuned
            at 250 Hz. At our 100 Hz the same counts stretch every receptive
            field 2.5x and push the input below braindecode's minimum, which
            silently rewrites the model: n_windows 5 -> 3 (its windowed attention
            ensemble, the "AT" in ATCNet), tcn_kernel 4 -> 2. Rate-matching
            restores the intended durations -- 256 ms and 64 ms filters, ~200 ms
            frames -- and drops the minimum below our input so nothing degrades.
            TARGETS: accuracy.

  ctx    -- ATCNet's attention spans sliding windows INSIDE one 4 s epoch. The
            structure in this task is at 12-32 s: self-transition 0.98/0.96 at a
            0.5 s step, and a fixed label-space dwell prior cut spurious
            activations ~3x on both cohorts across 4 seeds. PowerAttnNet failed
            partly because its attention was at the within-epoch timescale --
            deleting it was worth +0.024. This adds a CAUSAL transformer across
            consecutive epochs (K at stride S spans win + (K-1)*step*S seconds),
            so the model learns the persistence the fixed prior imposes.
            TARGETS: false activations/min, latency, accuracy.

  gate   -- a SelectiveNet-style abstention head trained under a coverage
            constraint, so declining to act is learned jointly with the decision
            rather than thresholded onto a finished classifier afterwards.
            TARGETS: accuracy@coverage, calibration.

DELIBERATELY NOT INCLUDED, because each was measured and failed:

  * a fused amplitude side-channel (-0.005 across 7 backbones; its shuffle
    control matched it exactly, so the gain was capacity, not information)
  * adversarial / decorrelation / HSIC invariance losses (-0.029 / -0.196 /
    -0.012), and the corrected probe says there is no excess movement
    information to remove (-0.11, contamination bounded below 0.5% of signal
    power)
  * an MRCP pathway (single-trial accuracy 0.604 vs 0.775 for ERD; combining
    them scored WORSE than ERD alone, and detection was slower, not earlier)
  * within-epoch attention beyond what ATCNet already has (measured harmful)

The backbone is byte-for-byte the published model: braindecode exposes
`final_layer`, and replacing it with Identity turns the network into a feature
extractor without touching anything upstream.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class ATCNetPlus(nn.Module):
    """ATCNet backbone + optional cross-epoch context + optional selective gate.

    Input is (B, K, C, T): K consecutive epochs at a fixed stride. The label
    predicted is the LAST epoch's and the transformer mask is causal, so only the
    present and past are used -- the only variant a worn device could run. With
    K=1 and ctx off, this is exactly the published ATCNet plus a linear head.
    """

    def __init__(self, backbone, feat_dim, n_outputs=2, use_ctx=True,
                 use_gate=True, d_model=128, n_layers=2, n_heads=4, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.use_ctx, self.use_gate = use_ctx, use_gate
        self.proj = nn.Sequential(nn.LayerNorm(feat_dim),
                                  nn.Linear(feat_dim, d_model), nn.GELU())
        if use_ctx:
            self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4, dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.ctx = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, n_outputs)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model // 4, 1)) if use_gate else None

    def forward(self, x):
        if x.dim() == 3:                       # (B, C, T) -> single epoch
            x = x.unsqueeze(1)
        B, K = x.shape[:2]
        f = self.backbone(x.reshape(B * K, *x.shape[2:])).flatten(1)
        e = self.proj(f).reshape(B, K, -1)
        if self.use_ctx and K > 1:
            mask = torch.triu(torch.ones(K, K, device=x.device, dtype=torch.bool),
                              diagonal=1)
            e = self.ctx(self.pos(e), mask=mask)
        h = self.norm(e[:, -1])
        out = {"logits": self.classify(h), "emb": h}
        out["gate"] = (torch.sigmoid(self.gate(h)).squeeze(-1) if self.use_gate
                       else torch.ones(B, device=x.device))
        return out


def build_atcnet_plus(n_chan, n_time, n_outputs=2, rate=True, use_ctx=True,
                      use_gate=True, sfreq=100.0, **kw):
    """Assemble ATCNet+ with the requested modules switched on."""
    from atcnet_rate import build_atcnet
    net = build_atcnet("rate" if rate else "default", n_chan, n_time, n_outputs,
                       sfreq)
    if isinstance(net.final_layer, nn.ModuleList):
        net.final_layer = nn.ModuleList([nn.Identity() for _ in net.final_layer])
    else:
        net.final_layer = nn.Identity()
    net.eval()
    with torch.no_grad():
        feat_dim = net(torch.zeros(2, n_chan, n_time)).flatten(1).shape[1]
    net.train()
    return ATCNetPlus(net, feat_dim, n_outputs, use_ctx, use_gate, **kw)


def build_bd_plus(name, n_chan, n_time, n_outputs=2, sfreq=100.0, **kw):
    """Any braindecode decoder under the identical wrapper stock ATCNet gets.

    final_layer -> Identity, then the same LayerNorm-Linear-GELU projection and
    linear classifier as the `base` arm (context and gate off), so a published
    model and stock ATCNet differ only in the backbone.
    """
    from braindecode_zoo import BDBackbone
    bb = BDBackbone(name, n_chan=n_chan, n_time=n_time, sfreq=sfreq)
    return ATCNetPlus(bb.net, bb.feat_dim, n_outputs, use_ctx=False,
                      use_gate=False, **kw)


def selective_loss(out, y, weight=None, target_coverage=0.9, lam=32.0):
    """CE on the covered set + quadratic coverage penalty + full-coverage aux.

    The auxiliary term keeps the backbone learning from every sample; without it
    the gate can shut early and starve the encoder of gradient.
    """
    g = out["gate"].clamp(1e-6, 1.0)
    ce = F.cross_entropy(out["logits"], y, weight=weight, reduction="none")
    cov = g.mean()
    sel = (g * ce).sum() / (g.sum() + 1e-6)
    pen = lam * torch.clamp(target_coverage - cov, min=0.0) ** 2
    aux = F.cross_entropy(out["logits"], y, weight=weight)
    return 0.5 * (sel + pen) + 0.5 * aux, cov.detach()


if __name__ == "__main__":
    for rate in (False, True):
        for ctx in (False, True):
            net = build_atcnet_plus(60, 400, rate=rate, use_ctx=ctx, use_gate=True)
            n = sum(p.numel() for p in net.parameters())
            nb = sum(p.numel() for p in net.backbone.parameters())
            K = 12 if ctx else 1
            o = net(torch.randn(2, K, 60, 400))
            print("rate=%-5s ctx=%-5s backbone=%7d +modules=%7d  nw=%d  out=%s"
                  % (rate, ctx, nb, n - nb, net.backbone.n_windows,
                     tuple(o["logits"].shape)))
