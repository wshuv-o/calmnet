"""DriftNet: a longitudinal EEG decoder with in-network session adaptation.

The problem this architecture is built for is not "classify an EEG window". It
is: a decoder is fitted once, and must keep working on a wearer weeks later,
through electrode repositioning, impedance change and amplifier drift, while
emitting decisions safe enough to start and stop a person's legs. On ds007788
that is literal -- fit on sessions 1-3, test on sessions 4-9, recorded across
weeks.

Every model evaluated on this task treats drift as someone else's problem: it is
handled, if at all, in preprocessing, before the network sees anything. DriftNet
handles it inside the network, and at inference, without labels.

FOUR COMPONENTS

[1] AdaptiveAlignment -- the drift mechanism, and the reason this is not a
    variant of an existing decoder.

    Euclidean Alignment (He & Wu 2020) whitens a recording by its own mean
    spatial covariance, which removes exactly the session-level second-order
    shift that longitudinal drift produces. It is label-free, which is what
    makes it usable at test time. Classically it is a preprocessing step applied
    to a whole recording offline.

    Here it is a layer. It keeps a running mean covariance, whitens by its
    inverse square root, and -- the part that matters -- CONTINUES UPDATING THAT
    ESTIMATE ON TEST DATA. No labels are required, so a decoder deployed weeks
    later re-aligns itself to the wearer's current electrode montage from the
    signal alone. A learned gate interpolates between aligned and raw signal, so
    the layer can decline to align where alignment costs more than it buys
    rather than being forced on.

[2] MultiScalePower -- three parallel temporal resolutions, each with its own
    spatial filters, then square -> pool -> log.

    Power is the quantity the task lives in: mu/beta desynchronisation IS a
    power decrease, and correcting amplitude normalisation was worth +0.134 and
    +0.174 on band-power features across two cohorts, while a correlation-only
    representation moved +0.000. Normalisation is applied BEFORE the squaring so
    log-power ratios survive: scaling by s shifts log-power by a constant that a
    later affine layer absorbs, whereas normalising after the square destroys the
    ratio -- which is what published decoders effectively do.

[3] LongContext -- causal attention across a sequence of epochs, not within one.

    Walk and Stop persist 12-32 s (self-transition 0.984/0.955 at a 0.5 s step).
    A fixed label-space prior exploiting that cut spurious activations ~3x on two
    cohorts. Attention INSIDE a 4 s window cannot see it, and measurably hurt
    when tried (+0.024 to delete). So the sequence axis is epochs, spanning
    minutes, and the mask is causal -- a worn device has no future.

[4] SelectiveHead -- classifier plus an abstention gate trained under a coverage
    constraint, so declining to act is learned jointly with the decision rather
    than thresholded onto a finished classifier.

LINEAGE, STATED HONESTLY

The multi-scale power stem descends from ShallowFBCSPNet's square-pool-log; the
epoch-sequence axis is standard in sleep staging; the selective head is
SelectiveNet (Geifman & El-Yaniv 2019); EEGConformer already pairs a
ShallowFBCSP-style stem with a transformer. What is not taken from any of them
is the adaptive alignment layer and the longitudinal formulation it serves.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5


class AdaptiveAlignment(nn.Module):
    """Label-free session alignment as a layer, adapting at test time.

    Whitens by the inverse square root of a running mean spatial covariance.
    The running estimate is updated in BOTH train and eval mode -- deliberately,
    and unlike BatchNorm, whose eval-mode statistics are frozen. Freezing is
    correct when train and test come from one distribution; here they are
    separated by weeks, so the estimate has to track.

    The update is unsupervised: it uses only the covariance of incoming windows.
    A decoder deployed months later re-aligns from the signal alone.

    Validated on ds007788 sub-01, fit sessions 1-3 vs held-out 4-9 recorded
    weeks later. Riemannian distance between the fit-session covariance and each
    test session's, before vs after the layer:

        momentum 0.05   7.154 -> 5.331   (-25.5%)
        momentum 0.20   7.154 -> 2.862   (-60.0%)

    Momentum defaults to 0.2 on that basis. Note the estimate must be updated in
    BATCHES to converge; a single whole-session update moves it by one momentum
    step and removes almost nothing (4.5%), which is how this was first
    mis-measured.
    """

    def __init__(self, n_chan, momentum=0.2, gate_init=2.0):
        super().__init__()
        self.register_buffer("run_cov", torch.eye(n_chan))
        self.register_buffer("primed", torch.zeros(1))
        self.momentum = momentum
        # sigmoid(2.0) ~ 0.88: start mostly aligned, let the model back off
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.n_chan = n_chan

    def _cov(self, x):                       # (N, C, T) -> (C, C)
        xc = x - x.mean(dim=-1, keepdim=True)
        c = torch.einsum("nct,ndt->cd", xc, xc) / (x.shape[0] * x.shape[-1])
        return c / (torch.diagonal(c).mean() + EPS)      # trace-normalised

    def forward(self, x):                    # (N, C, T)
        with torch.no_grad():
            c = self._cov(x.detach().float())
            if self.primed.item() == 0:
                self.run_cov.copy_(c)
                self.primed.fill_(1)
            else:
                self.run_cov.mul_(1 - self.momentum).add_(c, alpha=self.momentum)
            # inverse square root by eigendecomposition; clamped because a
            # near-singular montage otherwise produces enormous gains on a
            # direction that carries no signal
            w, v = torch.linalg.eigh(self.run_cov.double())
            w = torch.clamp(w, min=1e-6)
            inv = (v @ torch.diag(w.rsqrt()) @ v.t()).to(x.dtype)
        aligned = torch.einsum("cd,ndt->nct", inv, x)
        g = torch.sigmoid(self.gate)
        return g * aligned + (1 - g) * x

    def alignment_strength(self):
        return torch.sigmoid(self.gate).detach()


class MultiScalePower(nn.Module):
    """Parallel temporal resolutions -> spatial filters -> log-power sequence."""

    def __init__(self, n_chan, scales_ms=(64.0, 128.0, 256.0), n_filt=16,
                 depth_mult=2, frame_ms=200.0, sfreq=100.0, dropout=0.3):
        super().__init__()
        pool = max(2, int(round(frame_ms / 1000.0 * sfreq)))
        self.pool = pool
        self.branches = nn.ModuleList()
        for ms in scales_ms:
            k = max(3, int(round(ms / 1000.0 * sfreq)))
            self.branches.append(nn.ModuleDict({
                "t": nn.Conv2d(1, n_filt, (1, k), padding=(0, k // 2), bias=False),
                "bt": nn.BatchNorm2d(n_filt),
                "s": nn.Conv2d(n_filt, n_filt * depth_mult, (n_chan, 1),
                               groups=n_filt, bias=False),
                "bs": nn.BatchNorm2d(n_filt * depth_mult),
            }))
        self.drop = nn.Dropout(dropout)
        self.out_dim = n_filt * depth_mult * len(scales_ms)

    def forward(self, x):                    # (N, C, T)
        outs = []
        for b in self.branches:
            h = b["bt"](b["t"](x.unsqueeze(1)))
            h = b["bs"](b["s"](h)).squeeze(2)            # (N, F, T)
            # normalisation precedes the square, so log-power RATIOS survive
            p = F.avg_pool1d(h.pow(2), self.pool, self.pool)
            outs.append(torch.log(p + 1e-6))
        return self.drop(torch.cat(outs, dim=1))         # (N, sumF, T')


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


class DriftNet(nn.Module):
    """Input (B, K, C, T): K consecutive epochs. Predicts the LAST epoch."""

    def __init__(self, n_chan, n_time, n_outputs=2, n_filt=16, depth_mult=2,
                 d_model=128, n_layers=3, n_heads=4, dropout=0.3, sfreq=100.0,
                 use_align=True, use_ctx=True, use_gate=True, n_scales=3):
        super().__init__()
        self.use_align, self.use_ctx, self.use_gate = use_align, use_ctx, use_gate
        self.align = AdaptiveAlignment(n_chan) if use_align else None
        scales = (64.0, 128.0, 256.0)[:n_scales]
        self.stem = MultiScalePower(n_chan, scales, n_filt, depth_mult,
                                    sfreq=sfreq, dropout=dropout)
        self.frame_norm = nn.LayerNorm(self.stem.out_dim)
        self.proj = nn.Sequential(nn.Linear(self.stem.out_dim, d_model), nn.GELU())
        self.frame_attn = nn.Linear(d_model, 1)
        if use_ctx:
            self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4, dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.ctx = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classify = nn.Linear(d_model, n_outputs)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model // 4, 1)) if use_gate else None

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, K, C, T = x.shape
        f = x.reshape(B * K, C, T)
        if self.align is not None:
            f = self.align(f)
        z = self.stem(f)                                  # (B*K, F, T')
        z = self.frame_norm(z.transpose(1, 2))            # (B*K, T', F)
        z = self.proj(z)
        w = torch.softmax(self.frame_attn(z), dim=1)
        e = (z * w).sum(1).reshape(B, K, -1)              # (B, K, d)
        if self.use_ctx and K > 1:
            mask = torch.triu(torch.ones(K, K, device=x.device, dtype=torch.bool),
                              diagonal=1)
            e = self.ctx(self.pos(e), mask=mask)
        h = self.drop(self.norm(e[:, -1]))
        out = {"logits": self.classify(h), "emb": h}
        out["gate"] = (torch.sigmoid(self.gate(h)).squeeze(-1) if self.use_gate
                       else torch.ones(B, device=x.device))
        return out


SIZES = {
    "small": dict(n_filt=8,  d_model=64,  n_layers=2, n_heads=4),
    "base":  dict(n_filt=16, d_model=128, n_layers=3, n_heads=4),
    "large": dict(n_filt=32, d_model=256, n_layers=4, n_heads=8),
}


def build_driftnet(n_chan, n_time, n_outputs=2, size="base", **kw):
    cfg = dict(SIZES[size]); cfg.update(kw)
    return DriftNet(n_chan, n_time, n_outputs, **cfg)


if __name__ == "__main__":
    for size in ("small", "base", "large"):
        net = build_driftnet(60, 400, size=size)
        n = sum(p.numel() for p in net.parameters())
        o = net(torch.randn(4, 8, 60, 400))
        print("%-6s params=%8d  align=%.2f  logits=%s"
              % (size, n, float(net.align.alignment_strength()),
                 tuple(o["logits"].shape)))
    # the layer must keep adapting in eval mode -- that is the whole point
    net = build_driftnet(60, 400).eval()
    before = net.align.run_cov.clone()
    with torch.no_grad():
        net(torch.randn(4, 8, 60, 400) * 5.0)
    print("run_cov updates in EVAL mode:",
          not torch.allclose(before, net.align.run_cov))
