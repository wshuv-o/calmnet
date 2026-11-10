import torch
import torch.nn as nn

from .stem import MultiScalePower
from .tangent_branch import TangentBranch


class CALMNet(nn.Module):
    """Log-power stem + adaptive tangent-space branch. use_tangent=False gives the stem."""

    # layers are created in the order of the original code, so seeds reproduce
    def __init__(self, n_chan, n_outputs=2, n_filt=16, depth_mult=2,
                 d_model=128, dropout=0.3, sfreq=100.0, use_tangent=True,
                 shrink=0.1, momentum=0.2, ref_mode="running",
                 causal_update=False):
        super().__init__()
        self.stem = MultiScalePower(n_chan, (64.0, 128.0, 256.0), n_filt,
                                    depth_mult, sfreq=sfreq, dropout=dropout)
        self.frame_norm = nn.LayerNorm(self.stem.out_dim)
        self.proj = nn.Sequential(nn.Linear(self.stem.out_dim, d_model), nn.GELU())
        self.frame_attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.tangent = (TangentBranch(n_chan, d_model, shrink=shrink,
                                      momentum=momentum, dropout=dropout,
                                      ref_mode=ref_mode, causal=causal_update)
                        if use_tangent else None)
        self.fuse = nn.Linear(d_model * 2, d_model) if use_tangent else None
        self.classify = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        if self.tangent is not None:
            t = self.tangent(x)
        z = self.stem(x)
        z = self.proj(self.frame_norm(z.transpose(1, 2)))
        w = torch.softmax(self.frame_attn(z), dim=1)
        e = (z * w).sum(1)
        if self.tangent is not None:
            e = self.fuse(torch.cat([e, t], dim=-1))
        return self.classify(self.drop(self.norm(e)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
