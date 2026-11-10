import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePower(nn.Module):
    """Temporal conv (64/128/256 ms) -> depthwise spatial conv -> square -> pool -> log."""

    def __init__(self, n_chan, scales_ms=(64.0, 128.0, 256.0), n_filt=16,
                 depth_mult=2, frame_ms=200.0, sfreq=100.0, dropout=0.3):
        super().__init__()
        self.pool = max(2, int(round(frame_ms / 1000.0 * sfreq)))
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

    def forward(self, x):
        outs = []
        for b in self.branches:
            h = b["bt"](b["t"](x.unsqueeze(1)))
            h = b["bs"](b["s"](h)).squeeze(2)
            p = F.avg_pool1d(h.pow(2), self.pool, self.pool)
            outs.append(torch.log(p + 1e-6))
        return self.drop(torch.cat(outs, dim=1))
