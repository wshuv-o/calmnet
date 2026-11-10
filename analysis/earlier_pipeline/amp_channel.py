"""Log-power side channel attached to a braindecode decoder (Table 3, shuffle control)."""
import torch
import torch.nn as nn

EPS = 1e-8


def log_power(x, n_seg=4):
    B, C, T = x.shape
    seg = T // n_seg
    if seg < 2:
        return torch.log(x.var(dim=-1) + EPS)
    xs = x[..., :seg * n_seg].reshape(B, C, n_seg, seg)
    xs = xs - xs.mean(dim=-1, keepdim=True)
    return torch.log(xs.var(dim=-1) + EPS).reshape(B, C * n_seg)


class AmplitudeSideChannel(nn.Module):
    """mode=shuffle permutes the amplitude vector across the batch during training."""

    def __init__(self, backbone, feat_dim, n_chan, n_time, n_outputs=2,
                 n_seg=4, n_amp=32, mode="on"):
        super().__init__()
        self.backbone = backbone
        self.n_seg, self.mode = n_seg, mode
        d_amp = n_chan * n_seg
        self.amp = nn.Sequential(nn.BatchNorm1d(d_amp), nn.Linear(d_amp, n_amp),
                                 nn.ELU(), nn.Dropout(0.25))
        self.head = nn.Linear(feat_dim + n_amp, n_outputs)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        a = log_power(x, self.n_seg)
        if self.mode == "shuffle" and self.training:
            a = a[torch.randperm(a.shape[0], device=a.device)]
        return self.head(torch.cat([f, self.amp(a)], dim=1))


def wrap(name, n_chan, n_time, n_outputs=2, mode="on", **kw):
    import braindecode.models as B
    net = getattr(B, name)(n_chans=n_chan, n_outputs=n_outputs, n_times=n_time,
                           sfreq=100.0, **kw)
    if isinstance(net.final_layer, nn.ModuleList):
        net.final_layer = nn.ModuleList([nn.Identity() for _ in net.final_layer])
    else:
        net.final_layer = nn.Identity()
    net.eval()
    with torch.no_grad():
        feat_dim = net(torch.zeros(2, n_chan, n_time)).flatten(1).shape[1]
    net.train()
    return AmplitudeSideChannel(net, feat_dim, n_chan, n_time, n_outputs, mode=mode)
