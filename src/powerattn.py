"""PowerAttnNet: ShallowFBCSPNet's power pathway x ATCNet's temporal modelling.

A merge, not a bolt-on. Each parent holds exactly the piece the other lacks, and
both pieces are the ones this task was measured to need.

WHAT EACH PARENT DOES

ShallowFBCSPNet is FBCSP written as a network: temporal convolution (a learned
band-pass), spatial convolution (a learned CSP filter), then `square -> mean-pool
-> log`. That last chain is an explicit LOG-POWER estimator, and log-power is
exactly the quantity Walk-vs-Stop lives in, because mu/beta ERD is a power
decrease. Its limitation is the pooling: it averages power across the entire
window and emits one vector, so WHEN the power changed inside the window is
discarded.

ATCNet does the opposite. Multi-head self-attention plus a temporal convolutional
network over sliding sub-windows gives it genuine temporal modelling. But it
BatchNorms immediately after its spatial convolution, which rescales every
feature map to unit variance and destroys absolute power -- the very quantity
ShallowFBCSPNet is built to measure.

WHY THIS PARTICULAR MERGE, ON THIS DATA

Two measurements from this project motivate it, and neither is speculative:

  1. Power is the signal. A feature set built only from marginal channel power
     gains +0.134 balanced accuracy when amplitude normalisation is corrected,
     while correlation-based features move -0.016. Seven published models were
     insensitive to input normalisation at 0.82-0.84 because their internal
     BatchNorm had already discarded the amplitude.

  2. Temporal structure is real. Walk and Stop are sustained states with
     self-transition 0.98/0.96 at a 0.5 s step -- 12-32 s dwells. A label-space
     dwell prior cut spurious activations ~3x across both cohorts and 4 seeds.

So: keep the power estimator, but do NOT collapse it. Emit log-power as a
SEQUENCE and let attention and dilated convolutions model how it evolves.
Neither parent does this -- ShallowFBCSPNet has the power and throws away the
time axis; ATCNet has the time axis and throws away the power.

A previous attempt to hand the same information to published backbones as a
fused side-channel changed mean accuracy by -0.005 across seven architectures
(4 up, 3 down). That is why this is built as a pathway rather than an appendage:
a vector concatenated at the classifier cannot be attended over, and the
temporal evolution of power was the part that was missing.

CONTROLS BUILT IN

  mode="full"      the merge
  mode="nopower"   identical graph, but the square-log power estimator replaced
                   by a plain linear activation -- isolates the power pathway
  mode="noattn"    identical graph, attention and TCN replaced by mean pooling
                   over time -- collapses to a ShallowFBCSPNet-shaped model and
                   isolates the temporal pathway

If `full` does not beat BOTH ablations, the merge is not doing what it claims,
whatever its accuracy.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class LogPowerFront(nn.Module):
    """ShallowFBCSPNet's front end, emitting a SEQUENCE instead of one vector.

    temporal conv -> spatial conv -> square -> moving-average pool -> log

    The pooling stride sets the output rate: with pool=75 / stride=15 at 100 Hz
    a 4 s window yields ~24 log-power frames, which is enough sequence for
    attention to have something to attend to while keeping each frame's variance
    estimate stable (0.75 s of samples per frame).
    """

    def __init__(self, n_chan, n_filt=40, k_time=25, pool=75, stride=15):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filt, (1, k_time), padding=(0, k_time // 2))
        self.spatial = nn.Conv2d(n_filt, n_filt, (n_chan, 1), groups=1, bias=False)
        self.bn = nn.BatchNorm2d(n_filt)
        self.pool, self.stride = pool, stride

    def forward(self, x):                      # (B, C, T)
        h = self.temporal(x.unsqueeze(1))      # (B, F, C, T)
        h = self.spatial(h)                    # (B, F, 1, T)
        h = self.bn(h).squeeze(2)              # (B, F, T)
        # BatchNorm sits BEFORE the squaring, so it conditions the filter
        # outputs without destroying the power ratios the log then measures:
        # scaling a signal by s scales its log-power by a constant offset, which
        # a downstream affine layer absorbs, whereas normalising AFTER the
        # squaring (what the published models effectively do) removes the ratio.
        p = h.pow(2)
        p = F.avg_pool1d(p, kernel_size=self.pool, stride=self.stride)
        return torch.log(p + EPS)              # (B, F, T')


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


class TemporalBlock(nn.Module):
    """Dilated causal convolution block (ATCNet / locuslab TCN)."""

    def __init__(self, c_in, c_out, k=3, dilation=1, dropout=0.3):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation))
        self.pad, self.drop = pad, nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        h = self.drop(F.elu(self.conv1(x)[..., : -self.pad or None]))
        h = self.drop(F.elu(self.conv2(h)[..., : -self.pad or None]))
        return F.elu(h + (x if self.down is None else self.down(x)))


class PowerAttnNet(nn.Module):
    def __init__(self, n_chan, n_time, n_outputs=2, n_filt=40, d_model=40,
                 n_heads=4, n_tcn=2, dropout=0.3, mode="full"):
        super().__init__()
        self.mode = mode
        self.front = LogPowerFront(n_chan, n_filt=n_filt)
        self.norm = nn.LayerNorm(n_filt)
        self.pos = PositionalEncoding(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.tcn = nn.Sequential(*[
            TemporalBlock(d_model, d_model, dilation=2 ** i, dropout=dropout)
            for i in range(n_tcn)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        h = self.front.temporal(x.unsqueeze(1))
        h = self.front.spatial(h)
        h = self.front.bn(h).squeeze(2)
        if self.mode == "nopower":
            # control: same graph, no squaring/log -- a plain pooled activation,
            # so the pathway carries amplitude linearly rather than as log-power
            z = F.avg_pool1d(F.elu(h), self.front.pool, self.front.stride)
        else:
            z = torch.log(F.avg_pool1d(h.pow(2), self.front.pool,
                                       self.front.stride) + EPS)
        z = self.norm(z.transpose(1, 2))               # (B, T', F)

        if self.mode == "noattn":
            # control: collapse time immediately -- ShallowFBCSPNet-shaped
            return self.head(self.drop(z.mean(dim=1)))

        a, _ = self.attn(self.pos(z), self.pos(z), z, need_weights=False)
        z = self.attn_norm(z + self.drop(a))
        z = self.tcn(z.transpose(1, 2))                # (B, F, T')
        return self.head(self.drop(z[..., -1]))


# Capacity presets. At ~900 fit windows per subject this project measured a
# 1.3k-parameter network at 0.873 while 441k-parameter published models sat at
# 0.82-0.84, so capacity is not free here and the merge has to be testable at a
# size matched to the data rather than only at its natural size.
SIZES = {
    "tiny":  dict(n_filt=8,  d_model=8,  n_heads=2, n_tcn=1),
    "small": dict(n_filt=16, d_model=16, n_heads=2, n_tcn=1),
    "base":  dict(n_filt=40, d_model=40, n_heads=4, n_tcn=2),
}


def build_powerattn(n_chan, n_time, n_outputs=2, mode="full", size="base"):
    return PowerAttnNet(n_chan, n_time, n_outputs, mode=mode, **SIZES[size])
