import torch
import torch.nn as nn

PUBLISHED = ("EEGNeX", "ShallowFBCSPNet", "EEGNet", "EEGConformer", "TSception",
             "FBLightConvNet", "Deep4Net", "EEGTCNet")

# FBLightConvNet's default window (250 samples) is longer than the 2 s windows
EXTRA_KWARGS = {"FBLightConvNet": {"win_len": 100}}


class BraindecodeDecoder(nn.Module):
    """A braindecode model with its final layer replaced by a shared linear head."""

    def __init__(self, name, n_chan, n_time, n_outputs=2, sfreq=100.0, d_model=128):
        super().__init__()
        from braindecode import models as M
        cls = getattr(M, name)
        kw = dict(EXTRA_KWARGS.get(name, {}))
        try:
            net = cls(n_chans=n_chan, n_times=n_time, n_outputs=2, sfreq=sfreq, **kw)
        except TypeError:
            net = cls(n_chans=n_chan, n_times=n_time, n_outputs=2, **kw)
        net.final_layer = nn.Identity()
        net.eval()
        with torch.no_grad():
            feat_dim = int(net(torch.zeros(2, n_chan, n_time)).reshape(2, -1).shape[1])
        net.train()
        self.backbone = net
        self.proj = nn.Sequential(nn.LayerNorm(feat_dim),
                                  nn.Linear(feat_dim, d_model), nn.GELU())
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.classify(self.norm(self.proj(f)))
