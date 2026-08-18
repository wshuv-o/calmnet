"""EEG decoders for Walk/Stop MI: EEGNet baseline and EEG Conformer backbone.

Both take input (B, 1, C, T) and output 2-class logits. Dropout can be kept
active at inference (MC-dropout) for epistemic uncertainty.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# EEGNet (Lawhern et al. 2018), sized for 100 Hz / 60 ch / 2 s windows.
# --------------------------------------------------------------------------- #
class EEGNet(nn.Module):
    def __init__(self, n_chan=60, n_time=200, n_classes=2,
                 F1=16, D=2, F2=32, kern_len=50, p_drop=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # depthwise spatial conv across all channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p_drop),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p_drop),
        )
        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, 1, n_chan, n_time)).shape[1]
        self.feat_dim = n_feat
        self.classify = nn.Linear(n_feat, n_classes)

    def _features(self, x):
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return torch.flatten(x, 1)

    def features(self, x):        # pooled embedding before the classifier (for MID)
        return self._features(x)

    def forward(self, x):
        return self.classify(self._features(x))


# --------------------------------------------------------------------------- #
# EEG Conformer (Song et al. 2023): conv patch embedding + transformer encoder.
# Scaled to 100 Hz / 200-sample windows.
# --------------------------------------------------------------------------- #
class _PatchEmbedding(nn.Module):
    def __init__(self, n_chan, emb=40, p_drop=0.3):
        super().__init__()
        self.temporal = nn.Conv2d(1, emb, (1, 25), (1, 1))
        self.spatial = nn.Conv2d(emb, emb, (n_chan, 1), (1, 1))
        self.bn = nn.BatchNorm2d(emb)
        self.pool = nn.AvgPool2d((1, 25), (1, 5))
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Conv2d(emb, emb, (1, 1))

    def forward(self, x):                       # (B,1,C,T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = F.elu(self.bn(x))
        x = self.pool(x)
        x = self.drop(x)
        x = self.proj(x)                        # (B,emb,1,Tok)
        x = x.squeeze(2).transpose(1, 2)        # (B,Tok,emb)
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, emb=40, depth=6, heads=4, ff=2, p_drop=0.3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=emb * ff,
            dropout=p_drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.enc(x)


class EEGConformer(nn.Module):
    def __init__(self, n_chan=60, n_time=200, n_classes=2,
                 emb=40, depth=6, heads=4, p_drop=0.3):
        super().__init__()
        self.patch = _PatchEmbedding(n_chan, emb, p_drop)
        self.transformer = _TransformerEncoder(emb, depth, heads, 2, p_drop)
        with torch.no_grad():
            tok = self.patch(torch.zeros(1, 1, n_chan, n_time)).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(emb * tok, 128), nn.ELU(), nn.Dropout(p_drop),
            nn.Linear(128, 32), nn.ELU(), nn.Dropout(p_drop),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.patch(x)
        x = self.transformer(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# CALMNet (ours): a compact, physiologically-grounded MI decoder built for
# calibrated abstention. Design choices, each motivated by the problem:
#   * multi-scale temporal filter bank -> learnable band-pass covering mu & beta
#     at once (a differentiable FBCSP front-end);
#   * grouped spatial conv over all channels -> CSP-like sensorimotor spatial
#     filters, one bank per temporal scale;
#   * square -> average-pool -> log  -> the ERD/ERS *band-power* feature that
#     actually drives motor imagery (ShallowConvNet-style, smoother and far less
#     overconfident than raw-activation pooling);
#   * temporal attention pooling -> down-weights movement-contaminated / noisy
#     sub-windows instead of averaging them in;
#   * heavy dropout -> doubles as MC-dropout epistemic uncertainty for the gate.
# The band-power + attention design yields softmax scores that carry usable
# confidence (high AUROC) and calibrate well under temperature scaling -- exactly
# what the abstention layer needs.
# --------------------------------------------------------------------------- #
class _AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens):                  # (B, Tok, C)
        w = torch.softmax(self.score(tokens), dim=1)   # (B, Tok, 1)
        return (w * tokens).sum(dim=1)                 # (B, C)


class CALMNet(nn.Module):
    def __init__(self, n_chan=60, n_time=200, n_classes=2,
                 F=8, D=2, kernels=(13, 25, 51), pool=25, stride=5, p_drop=0.5):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(1, F, (1, k), padding=(0, k // 2), bias=False) for k in kernels])
        Fc = F * len(kernels)
        self.bn_t = nn.BatchNorm2d(Fc)
        # CSP-like spatial filters, grouped per temporal filter
        self.spatial = nn.Conv2d(Fc, Fc * D, (n_chan, 1), groups=Fc, bias=False)
        self.bn_s = nn.BatchNorm2d(Fc * D)
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(p_drop)
        self.attn = _AttentionPool(Fc * D)
        with torch.no_grad():
            tok = self._tokens(torch.zeros(1, 1, n_chan, n_time)).shape[1]
        self.feat_dim = Fc * D
        self.norm = nn.LayerNorm(Fc * D)
        self.classify = nn.Linear(Fc * D, n_classes)
        self._tok = tok

    def _tokens(self, x):
        x = torch.cat([b(x) for b in self.branches], dim=1)   # (B, Fc, C, T)
        x = self.bn_t(x)
        x = self.spatial(x)                                   # (B, Fc*D, 1, T)
        x = self.bn_s(x)
        x = x ** 2                                            # power
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))              # log-band-power
        x = self.drop(x)
        return x.squeeze(2).transpose(1, 2)                  # (B, Tok, Fc*D)

    def features(self, x):
        """Pooled representation before the classifier (used by MID)."""
        return self.norm(self.attn(self._tokens(x)))

    def forward(self, x):
        return self.classify(self.features(x))


def build_model(name: str, n_chan=60, n_time=200, **kw) -> nn.Module:
    name = name.lower()
    if name == "eegnet":
        return EEGNet(n_chan, n_time, **kw)
    if name in ("conformer", "eegconformer"):
        return EEGConformer(n_chan, n_time, **kw)
    if name in ("calmnet", "calm"):
        return CALMNet(n_chan, n_time, **kw)
    raise ValueError(f"unknown model {name}")


def enable_mc_dropout(model: nn.Module):
    """Put the model in eval mode but re-activate dropout layers for MC sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


if __name__ == "__main__":
    for name in ("eegnet", "conformer", "calmnet"):
        m = build_model(name)
        x = torch.randn(4, 1, 60, 200)
        out = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:10s} out {tuple(out.shape)}  params {n:,}")
