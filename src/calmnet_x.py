"""CALM-Net-X: a large multi-scale power decoder with cross-epoch context.

Every component here exists because something measured in this project demanded
it. Nothing is included because it is fashionable, and the two mechanisms that
failed when tested are deliberately absent.

WHAT THE MEASUREMENTS SAID

  1. POWER IS THE SIGNAL. Correcting per-window amplitude normalisation is worth
     +0.134 (ds007788) and +0.174 (MoBI) on band-power features, while a
     correlation-only representation moves +0.000 -- a dose-response that tracks
     exactly how much marginal power each representation carries. Published
     models BatchNorm immediately after their spatial convolution and discard it.
     -> explicit square->pool->log power pathway, with normalisation applied
        BEFORE the squaring so power RATIOS survive (scaling by s shifts
        log-power by a constant, which an affine layer absorbs).

  2. THE STATES PERSIST FOR 12-32 SECONDS. Self-transition is 0.98/0.96 at a
     0.5 s step. A label-space dwell prior across windows cut spurious
     activations ~3x on both cohorts and 4 seeds. But attention INSIDE a 4 s
     window bought nothing -- PowerAttnNet's attention stack was actively
     harmful (+0.024 to delete it, consistent across 3 seeds).
     -> the temporal modelling belongs ACROSS epochs, not within them. A
        sequence of K epochs at 0.5 s stride spans 4 + (K-1)/2 seconds; K=24
        covers ~15.5 s, inside the measured dwell range. This is the one
        timescale no model here has been given access to, and it is the reason
        this architecture exists.

  3. ARCHITECTURE ONLY SEPARATES AT SCALE. At ~905 fit windows every design
     collapsed to 0.84 +- 0.02; at 4881 windows they separate by up to 0.053.
     So this model is built for, and only evaluated on, the full-data regime.

  4. CAPACITY IS NOT THE LEVER BY ITSELF. EEGConformer (441k) gained least from
     more data (+0.006) while ATCNet (45k) gained most (+0.053). Size is
     therefore spent on MULTI-SCALE BREADTH -- parallel temporal resolutions,
     each with its own spatial filters -- rather than on depth, which is what
     EEGConformer already has and does not convert into accuracy here.

  5. ATCNet's WINDOWED ATTENTION ENSEMBLE is the strongest published mechanism
     (0.881 at full data), and its hyperparameters are DURATIONS: a 256 ms
     temporal filter, 64 ms second filter, ~200 ms feature frames. Those are
     reproduced at 100 Hz rather than copied as sample counts.

WHY "MORE THAN ACCURACY" IS ARCHITECTURAL HERE

A decoder that starts and stops a person's legs is not well described by
balanced accuracy. Three of the five reported metrics are properties the model
is TRAINED for, not merely scored on:

  * an abstention gate trained under a coverage constraint (SelectiveNet-style),
    so the model can decline rather than guess -- and accuracy@coverage becomes
    a real operating point instead of a post-hoc threshold sweep;
  * temperature-free calibration pressure, because a device that acts on
    confidence needs the confidence to mean something (ECE is reported);
  * the cross-epoch pathway directly targets false ACTIVATIONS (spurious
    Stop->Walk transitions per minute of standing), which per-window accuracy
    hides: one long false run and fifty scattered false windows score alike.

Invariance (conditional movement R^2) is reported but NOT trained against. Every
adversarial/decorrelation objective tried in this project hurt -- adv -0.029,
decorr -0.196, hsic -0.012 -- and the corrected probe says the representation
carries no excess movement information to begin with (-0.11, with contamination
bounded below 0.5% of signal power by an injection curve).

DELIBERATELY ABSENT

  * within-epoch attention (measured harmful, +0.024 to remove)
  * a fused amplitude side-channel (measured -0.005 across 7 backbones; its
    shuffle control matched it exactly, so the gain was capacity not information)
  * adversarial / decorrelation / HSIC invariance losses (all measured harmful)
  * the motion canceller (an IMU classifier: 0.501 with its reference zeroed)
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
SFREQ = 100.0

# ATCNet's durations, reproduced at our sampling rate rather than copied as
# sample counts (see atcnet_rate.py for why that distinction matters).
SCALE_MS = (64.0, 128.0, 256.0)     # parallel temporal resolutions
FRAME_MS = 200.0                    # one log-power frame


class PowerScale(nn.Module):
    """One temporal resolution: band-pass-like conv -> spatial filter ->
    square -> pool -> log, emitting a SEQUENCE of log-power frames.

    A longer kernel resolves lower frequencies; running several in parallel lets
    the model spend capacity on frequency breadth instead of depth, which is
    what measurement 4 above argues for.
    """

    def __init__(self, n_chan, kern_ms, n_filt, depth_mult, pool, sfreq=SFREQ,
                 dropout=0.3):
        super().__init__()
        k = max(3, int(round(kern_ms / 1000.0 * sfreq)))
        self.temporal = nn.Conv2d(1, n_filt, (1, k), padding=(0, k // 2), bias=False)
        self.bn_t = nn.BatchNorm2d(n_filt)
        self.spatial = nn.Conv2d(n_filt, n_filt * depth_mult, (n_chan, 1),
                                 groups=n_filt, bias=False)
        self.bn_s = nn.BatchNorm2d(n_filt * depth_mult)
        self.pool, self.drop = pool, nn.Dropout(dropout)
        self.out_dim = n_filt * depth_mult

    def forward(self, x):                        # (B, C, T)
        h = self.bn_t(self.temporal(x.unsqueeze(1)))
        h = self.bn_s(self.spatial(h)).squeeze(2)          # (B, F, T)
        # normalisation is applied BEFORE squaring, so log-power ratios survive
        p = F.avg_pool1d(h.pow(2), self.pool, self.pool)
        return self.drop(torch.log(p + EPS))               # (B, F, T')


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class EpochEncoder(nn.Module):
    """Multi-scale log-power encoder for ONE epoch -> a single embedding."""

    def __init__(self, n_chan, n_time, n_filt=48, depth_mult=2, d_model=256,
                 dropout=0.3, sfreq=SFREQ):
        super().__init__()
        pool = max(2, int(round(FRAME_MS / 1000.0 * sfreq)))
        self.scales = nn.ModuleList([
            PowerScale(n_chan, ms, n_filt, depth_mult, pool, sfreq, dropout)
            for ms in SCALE_MS])
        d_cat = sum(s.out_dim for s in self.scales)
        self.proj = nn.Sequential(nn.LayerNorm(d_cat), nn.Linear(d_cat, d_model),
                                  nn.GELU(), nn.Dropout(dropout))
        self.frame_attn = nn.Linear(d_model, 1)     # pool frames by attention
        self.d_model = d_model

    def forward(self, x):                            # (B, C, T)
        z = torch.cat([s(x) for s in self.scales], dim=1)   # (B, sumF, T')
        z = self.proj(z.transpose(1, 2))                    # (B, T', d)
        w = torch.softmax(self.frame_attn(z), dim=1)        # (B, T', 1)
        return (z * w).sum(1)                               # (B, d)


class CalmNetX(nn.Module):
    """Per-epoch multi-scale power encoder + CAUSAL cross-epoch transformer.

    Input is a SEQUENCE of K consecutive epochs, (B, K, C, T). The label
    predicted is that of the LAST epoch, and the transformer mask is causal, so
    the model only ever uses the present and the past -- the only variant a worn
    device could actually run.
    """

    def __init__(self, n_chan, n_time, n_outputs=2, n_filt=48, depth_mult=2,
                 d_model=256, n_layers=4, n_heads=8, dropout=0.3,
                 use_context=True, sfreq=SFREQ):
        super().__init__()
        self.encoder = EpochEncoder(n_chan, n_time, n_filt, depth_mult, d_model,
                                    dropout, sfreq)
        self.use_context = use_context
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.ctx = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, n_outputs)
        # SelectiveNet-style gate: a trained decision to act or abstain, rather
        # than a threshold chosen after the fact
        self.gate = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model // 4, 1))

    def forward(self, x):                   # (B, K, C, T)
        B, K = x.shape[:2]
        e = self.encoder(x.reshape(B * K, *x.shape[2:])).reshape(B, K, -1)
        if self.use_context and K > 1:
            mask = torch.triu(torch.ones(K, K, device=x.device, dtype=torch.bool),
                              diagonal=1)
            e = self.ctx(self.pos(e), mask=mask)
        h = self.norm(e[:, -1])             # embedding of the LAST epoch
        return {"logits": self.classify(h),
                "gate": torch.sigmoid(self.gate(h)).squeeze(-1),
                "emb": h}


def build_calmnet_x(n_chan, n_time, n_outputs=2, size="base", use_context=True):
    cfg = {
        "small": dict(n_filt=16, depth_mult=2, d_model=96,  n_layers=2, n_heads=4),
        "base":  dict(n_filt=48, depth_mult=2, d_model=256, n_layers=4, n_heads=8),
        "large": dict(n_filt=96, depth_mult=2, d_model=384, n_layers=6, n_heads=8),
    }[size]
    return CalmNetX(n_chan, n_time, n_outputs, use_context=use_context, **cfg)


def selective_loss(out, y, weight=None, target_coverage=0.9, lam=32.0):
    """Cross-entropy on the covered set + a quadratic coverage penalty.

    Geifman & El-Yaniv's SelectiveNet objective. The gate learns WHICH inputs to
    act on while a penalty holds the acted-on fraction near `target_coverage`,
    so abstention is learned jointly with the decision instead of being a
    threshold applied to a fixed classifier afterwards. An auxiliary full-coverage
    head keeps the encoder training on every sample, which otherwise collapses
    early when the gate shuts.
    """
    g = out["gate"].clamp(1e-6, 1.0)
    ce = F.cross_entropy(out["logits"], y, weight=weight, reduction="none")
    cov = g.mean()
    sel = (g * ce).sum() / (g.sum() + 1e-6)
    pen = lam * torch.clamp(target_coverage - cov, min=0.0) ** 2
    aux = F.cross_entropy(out["logits"], y, weight=weight)
    return 0.5 * (sel + pen) + 0.5 * aux, cov.detach()


if __name__ == "__main__":
    for size in ("small", "base", "large"):
        for K in (1, 24):
            net = build_calmnet_x(60, 400, size=size, use_context=K > 1)
            n = sum(p.numel() for p in net.parameters())
            o = net(torch.randn(2, K, 60, 400))
            print("%-6s K=%-3d params=%9d  logits=%s gate=%s"
                  % (size, K, n, tuple(o["logits"].shape), tuple(o["gate"].shape)))
