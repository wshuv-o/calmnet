"""An amplitude side-channel for published EEG architectures.

THE DEFECT

Seven published architectures (3k-441k parameters) were benchmarked on this task
under three input normalisation schemes. Every one of them landed at 0.82-0.84
regardless of scheme:

    model            perwindow  global   none
    EEGNeX             0.834     0.832   0.835
    ATCNet             0.821     0.828   0.843
    EEGConformer       0.828     0.829   0.805
    ShallowFBCSPNet    0.824     0.821   0.818
    BDTCN              0.820     0.822   0.820
    Deep4Net           0.760     0.821   0.821
    EEGITNet           0.833     0.791   0.795

That insensitivity is not robustness, it is information loss. Each of these
networks applies BatchNorm immediately after its spatial convolution, which
rescales every feature map to zero mean and unit variance across the batch. The
absolute scale of the spatially-filtered signal is discarded there, permanently.

For most EEG benchmarks that is harmless or helpful. For THIS task it removes the
label: Walk versus Stop is separated by mu/beta event-related desynchronisation,
which is a POWER DECREASE. Power is variance, and variance is exactly what the
normalisation throws away. Measured directly -- a feature set built only from
marginal channel power gains +0.134 balanced accuracy when per-window
normalisation is removed, while correlation-based features move by -0.016,
because correlation survives normalisation and marginal power does not.

THE MODULE

`AmplitudeSideChannel` computes log-variance per channel over a few contiguous
time sub-windows, straight from the input, and fuses it with the backbone's
penultimate features before classification. It is deliberately the smallest
thing that can carry the missing quantity:

  * log-variance, not variance, because band power is log-normally distributed
    and the log makes the multiplicative ERD effect additive;
  * per sub-window rather than per epoch, so a power change part-way through a
    window is visible instead of averaged out;
  * computed on the raw input, BEFORE the backbone's own normalisation, which
    is the whole point -- once BatchNorm has run, the quantity is gone;
  * ~2-8k parameters, so any gain cannot be attributed to added capacity, and
    that is checked explicitly by a `shuffle` control that keeps the parameters
    and destroys only the correspondence between the amplitude vector and its
    trial.

The backbone is untouched and keeps its normalised pathway, which is what makes
it robust to amplitude drift across sessions. The side-channel restores what
that pathway costs. The claim under test is that this helps EVERY backbone, not
one -- a modification that improves a single architecture is a tuning result.
"""
from __future__ import annotations
import torch
import torch.nn as nn

EPS = 1e-8


def log_power(x, n_seg=4):
    """(B, C, T) -> (B, C*n_seg) log-variance per channel per sub-window.

    Centred within each sub-window so this measures POWER, not mean offset --
    a DC shift is an artefact signature, not an ERD one, and including it would
    hand the classifier a different confound than the one being restored.
    """
    B, C, T = x.shape
    seg = T // n_seg
    if seg < 2:
        return torch.log(x.var(dim=-1) + EPS)
    xs = x[..., :seg * n_seg].reshape(B, C, n_seg, seg)
    xs = xs - xs.mean(dim=-1, keepdim=True)
    return torch.log(xs.var(dim=-1) + EPS).reshape(B, C * n_seg)


class AmplitudeSideChannel(nn.Module):
    """Wrap any feature-extractor backbone with a log-power side-channel."""

    def __init__(self, backbone, feat_dim, n_chan, n_time, n_outputs=2,
                 n_seg=4, n_amp=32, mode="on"):
        super().__init__()
        self.backbone = backbone
        self.n_seg, self.mode = n_seg, mode
        d_amp = n_chan * n_seg
        self.amp = nn.Sequential(
            nn.BatchNorm1d(d_amp),      # per-FEATURE across batch: rescales the
                                        # log-power axis without collapsing the
                                        # within-batch differences that carry ERD
            nn.Linear(d_amp, n_amp), nn.ELU(), nn.Dropout(0.25),
        )
        self.head = nn.Linear(feat_dim + n_amp, n_outputs)

    def forward(self, x):
        f = self.backbone(x)
        f = f.flatten(1)
        a = log_power(x, self.n_seg)
        if self.mode == "shuffle" and self.training:
            # capacity control: same parameters, same gradients, but the
            # amplitude vector no longer belongs to its own trial
            a = a[torch.randperm(a.shape[0], device=a.device)]
        return self.head(torch.cat([f, self.amp(a)], dim=1))


def wrap(name, n_chan, n_time, n_outputs=2, mode="on", **kw):
    """Build a braindecode model as a feature extractor and attach the channel.

    braindecode exposes `final_layer` on every classifier; replacing it with
    Identity turns the network into a feature extractor without touching
    anything upstream, so the backbone is byte-for-byte the published one.
    """
    import braindecode.models as B
    net = getattr(B, name)(n_chans=n_chan, n_outputs=n_outputs, n_times=n_time,
                           sfreq=100.0, **kw)
    if not hasattr(net, "final_layer"):
        raise RuntimeError("%s exposes no final_layer to detach" % name)
    if isinstance(net.final_layer, nn.ModuleList):
        # ATCNet keeps one classifier per sliding window and combines them, so
        # `final_layer` is a ModuleList. Replacing the container with a single
        # Identity breaks its forward (it iterates); replacing each ELEMENT
        # leaves the combination logic intact and still yields features.
        net.final_layer = nn.ModuleList([nn.Identity() for _ in net.final_layer])
    else:
        net.final_layer = nn.Identity()
    net.eval()
    with torch.no_grad():
        feat_dim = net(torch.zeros(2, n_chan, n_time)).flatten(1).shape[1]
    net.train()
    return AmplitudeSideChannel(net, feat_dim, n_chan, n_time, n_outputs,
                                mode=mode)
