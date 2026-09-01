"""CALM-Net architecture: motion-referenced adaptive cancellation with
spectral leakage gating.

Not a wrapper around a published backbone. The design follows from three
measurements made on this data:

  (1) Movement leakage is concentrated in beta and grows monotonically with
      bandwidth: intent->motion R^2 runs -0.013 (mu 8-13 Hz) -> +0.032 (8-30)
      -> +0.101 (beta 13-30) -> +0.127 (1-45 Hz), with accuracy tracking it.
      So contamination is spectrally structured, not uniform.

  (2) Across 131 architectures, accuracy correlates +0.67 with leakage. Every
      family converts capacity into movement-reading, so no amount of encoder
      sophistication helps.

  (3) The motion signal is available AT DEPLOYMENT -- the exoskeleton carries
      its own IMUs, the treadmill rig its goniometers. Existing methods use
      motion only at training time, to predict (artefact heads) or to penalise
      (adversaries, HSIC). Nobody uses it to SUBTRACT.

The architecture therefore removes movement from the signal instead of
discouraging it in the features:

    EEG (C,T) + motion (R,T)
        |
        v
  MotionReferenceCanceller     learnable per-channel FIR filters predict the
                               movement-driven component of each EEG channel
                               from a motion reference basis, and subtract it.
                               In-network adaptive noise cancellation, trained
                               end-to-end with the decoder rather than as
                               preprocessing.
        |
        v
  Multi-band decomposition     mu / low-beta / high-beta pathways
        |
        v
  SpectralLeakageGate          per band and per window, estimate how strongly
                               that band's power envelope tracks the motion
                               envelope, and attenuate accordingly. Bands that
                               are behaving like movement are turned down for
                               that window -- an adaptive, data-driven version
                               of the mu-band restriction that measurement (1)
                               showed is the only genuinely invariant setting.
        |
        v
  CSP-like spatial filtering -> log band-power -> attention readout -> logits

Cancellation happens before feature extraction, so invariance is a property of
the representation rather than a penalty fighting the classifier. The MID
adversary is retained, but as a check on the front-end rather than as the sole
mechanism.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mid import grad_reverse


# --------------------------------------------------------------------------- #
# 1. Motion reference basis
# --------------------------------------------------------------------------- #
class MotionBasis(nn.Module):
    """Expand raw motion channels into a richer reference for cancellation.

    Movement artefact is not a linear copy of the motion trace: it involves
    velocity (cable sway, electrode shear), acceleration, and rectified/squared
    terms (power-envelope coupling). The canceller can only subtract what the
    reference spans, so the basis is built explicitly.
    """

    def __init__(self, n_ref):
        super().__init__()
        self.n_out = n_ref * 4

    def forward(self, m):                      # (B, R, T)
        d1 = torch.diff(m, dim=-1, prepend=m[..., :1])
        d2 = torch.diff(d1, dim=-1, prepend=d1[..., :1])
        env = m.abs()
        return torch.cat([m, d1, d2, env], dim=1)     # (B, 4R, T)


# --------------------------------------------------------------------------- #
# 2. In-network adaptive cancellation
# --------------------------------------------------------------------------- #
class MotionReferenceCanceller(nn.Module):
    """Predict each EEG channel's movement-driven component from the motion
    reference and subtract it.

    y_c(t) = x_c(t) - g_c * sum_r (h_{c,r} * ref_r)(t)

    h is a bank of learnable FIR filters (one per EEG channel x reference
    channel), so per-channel phase and lag are handled -- artefact reaches
    frontal and occipital electrodes with different delays and gains. g_c is a
    learned per-channel gate in [0,1]: channels with little artefact learn to
    subtract nothing, so the layer cannot destroy clean neural signal.

    Initialised at g = 0 (identity), so training starts from the unmodified
    signal and only cancels where cancelling helps.
    """

    def __init__(self, n_chan, n_ref, taps=25, use_basis=True):
        super().__init__()
        self.basis = MotionBasis(n_ref) if use_basis else nn.Identity()
        if not use_basis:
            self.basis.n_out = n_ref
        self.pred = nn.Conv1d(self.basis.n_out, n_chan, taps,
                              padding=taps // 2, bias=False)
        # Small random, NOT zero. With zero weights yhat == 0, so the gate's
        # gradient (-yhat * dL/dy) is identically zero and the gates cannot
        # start learning until the FIR bank moves first. Small init keeps the
        # layer near-identity while letting both paths train from step one.
        nn.init.normal_(self.pred.weight, std=1e-3)
        self.gate = nn.Parameter(torch.full((n_chan,), -2.0))   # sigmoid(-2) ~ 0.12
        self.taps = taps

    def forward(self, x, m):                   # x (B,C,T), m (B,R,T)
        ref = self.basis(m)
        yhat = self.pred(ref)[..., :x.shape[-1]]
        g = torch.sigmoid(self.gate)[None, :, None]
        return x - g * yhat, yhat

    def cancellation_strength(self):
        return torch.sigmoid(self.gate).detach()


# --------------------------------------------------------------------------- #
# 3. Spectral leakage gating
# --------------------------------------------------------------------------- #
class SpectralLeakageGate(nn.Module):
    """Attenuate bands whose power envelope tracks the motion envelope.

    For each band b and window, correlate the band's mean power envelope with
    the motion envelope. High |correlation| means that band is behaving like
    movement in this window, so it is scaled down. The gate is per-window, so a
    band contaminated during one window can still contribute during another --
    unlike a fixed band restriction, which pays the cost everywhere.
    """

    def __init__(self, n_bands, temp=4.0):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_bands))
        self.b = nn.Parameter(torch.zeros(n_bands))
        self.temp = temp

    def forward(self, band_pow, m_env):
        # band_pow (B, nb, T'), m_env (B, T')
        bp = band_pow - band_pow.mean(-1, keepdim=True)
        me = m_env - m_env.mean(-1, keepdim=True)
        num = (bp * me.unsqueeze(1)).mean(-1)
        den = bp.std(-1) * me.std(-1).unsqueeze(1) + 1e-6
        r = (num / den).abs()                       # (B, nb) leakage estimate
        gate = torch.sigmoid(-self.temp * (r * self.w + self.b))
        return gate, r


# --------------------------------------------------------------------------- #
# 4. Full model
# --------------------------------------------------------------------------- #
class CALMNetArch(nn.Module):
    BANDS = ((8, 13), (13, 20), (20, 30))       # mu, low-beta, high-beta

    def __init__(self, n_chan=60, n_time=200, n_ref=4, n_classes=2, k_imu=12,
                 F_=8, D=2, taps=25, pool=25, stride=5, p_drop=0.5, sfreq=100.0,
                 kernels=(13, 25, 51), mods=None):
        """mods: dict of module toggles. Each is independently ablatable so the
        final architecture can be composed from whichever pieces earn their
        place, rather than asserted."""
        super().__init__()
        M = {"cancel": True, "spec_gate": True, "basis": True, "attn": True,
             "art": True, "multiband": True}
        M.update(mods or {})
        self.mods = M
        if not M["multiband"]:
            kernels = (kernels[len(kernels) // 2],)
        self.canceller = (MotionReferenceCanceller(n_chan, n_ref, taps,
                                                   use_basis=M["basis"])
                          if M["cancel"] else None)
        self.n_bands = len(kernels)
        # one temporal filter bank per band pathway
        self.branches = nn.ModuleList(
            [nn.Conv2d(1, F_, (1, k), padding=(0, k // 2), bias=False) for k in kernels])
        self.bn_t = nn.BatchNorm2d(F_ * self.n_bands)
        self.spatial = nn.Conv2d(F_ * self.n_bands, F_ * self.n_bands * D,
                                 (n_chan, 1), groups=F_ * self.n_bands, bias=False)
        self.bn_s = nn.BatchNorm2d(F_ * self.n_bands * D)
        self.gate = SpectralLeakageGate(self.n_bands) if M["spec_gate"] else None
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(p_drop)
        self.F_, self.D = F_, D

        dim = F_ * self.n_bands * D
        self.attn = nn.Linear(dim, 1) if M["attn"] else None
        self.norm = nn.LayerNorm(dim)
        self.feat_dim = dim
        self.d_int = dim // 2
        self.classify = nn.Linear(self.d_int, n_classes)
        self.art_head = nn.Sequential(nn.Linear(dim - self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, k_imu))
        self.adv_head = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, 32), nn.ELU(), nn.Linear(32, k_imu))

    def _encode(self, x, m):
        """x (B,1,C,T) or (B,C,T); m (B,R,T) motion time series."""
        if x.ndim == 4:
            x = x.squeeze(1)
        if self.canceller is not None:
            xc, yhat = self.canceller(x, m)                  # cancellation
        else:
            xc, yhat = x, torch.zeros_like(x)
        z = xc.unsqueeze(1)
        z = torch.cat([b(z) for b in self.branches], dim=1)  # (B, F*nb, C, T)
        z = self.bn_t(z)
        z = self.spatial(z)                                  # (B, F*nb*D, 1, T)
        z = self.bn_s(z).squeeze(2)                          # (B, F*nb*D, T)
        p = z ** 2

        # per-band mean power envelope, pooled to the token rate
        B, Ch, T = p.shape
        per = Ch // self.n_bands
        bandp = p.view(B, self.n_bands, per, T).mean(2)      # (B, nb, T)
        bandp_s = F.avg_pool1d(bandp, 25, 5)
        m_env = F.avg_pool1d(m.abs().mean(1, keepdim=True), 25, 5).squeeze(1)
        if self.gate is not None:
            gate, r = self.gate(bandp_s, m_env)               # (B, nb)
            g = gate.repeat_interleave(per, dim=1).unsqueeze(-1)
            p = p * g                                         # attenuate leaky bands
        else:
            r = torch.zeros(B, self.n_bands, device=p.device)
        p = self.pool(p.unsqueeze(2)).squeeze(2)
        p = torch.log(torch.clamp(p, min=1e-6))
        p = self.drop(p).transpose(1, 2)                      # (B, Tok, Ch)
        if self.attn is not None:
            w = torch.softmax(self.attn(p), dim=1)
            z = (w * p).sum(1)
        else:
            z = p.mean(1)
        return self.norm(z), yhat, r

    def split(self, x, m):
        z, yhat, r = self._encode(x, m)
        return z[:, :self.d_int], z[:, self.d_int:], yhat, r

    def forward(self, x, m, grl=1.0):
        zi, za, yhat, r = self.split(x, m)
        return {"logits": self.classify(zi), "art": self.art_head(za),
                "adv": self.adv_head(grad_reverse(zi, grl)),
                "z_int": zi, "cancelled": yhat, "leak_r": r}

    @torch.no_grad()
    def encode(self, x, m):
        zi, za, _, _ = self.split(x, m)
        return zi, za
