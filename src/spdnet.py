"""Aligned SPD Network (ASPD-Net) -- a deep architecture on the SPD manifold
with a LEARNABLE alignment layer.

Design is forced by this project's own measurements, not chosen by taste:

  * Band-limited EEG is near-Gaussian, so the spatial covariance is a sufficient
    statistic and ERD/ERS is second-order by construction. A CNN on raw samples
    spends its capacity rediscovering covariance from 1.1k windows. Every one of
    131 architectures did exactly that and lost.
  * Euclidean Alignment -- whitening each session by its own mean covariance --
    was the single most effective operation found (0.776 vs 0.695 for the best
    deep model, and the only genuinely movement-invariant result). But EA is
    FIXED: it whitens by the arithmetic mean, fully, always.
  * Closed-form alignment beat every learned adversarial scheme (GRL, HSIC,
    decorrelation, in-network cancellation).

The gap that leaves: nothing operates on the manifold AND learns its alignment.
SPDNet (Huang & Van Gool 2017) gives manifold-preserving layers but has no
alignment. Euclidean Alignment gives alignment but is a fixed preprocessing
step with no free parameters and no depth. This joins them:

    multi-band covariances  (SPD, one per band)
        |
    LearnableAlignment      G = ((1-a)I + a*M)^b , per band, a and b learned.
                            a=1, b=1 recovers Euclidean Alignment exactly, so
                            the winning method is a point in this layer's
                            parameter space and the network can only improve on
                            it. M is the batch/session mean covariance --
                            computed from the data itself, no labels, so this
                            stays usable at deployment on an unseen session.
        |
    BiMap -> ReEig          SPD-preserving bilinear projection with orthogonal
                            weights, then eigenvalue rectification. Reduces
                            60x60 to a compact SPD representation without
                            leaving the manifold.
        |
    CrossBandCoupling       bands interact on the manifold rather than being
                            concatenated after flattening.
        |
    LogEig -> tangent -> linear classifier

Parameter count is tiny by design (~10-40k): the data is 1.1k windows per
subject, and the whole lesson of the architecture sweep is that capacity buys
movement leakage faster than it buys accuracy.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5


# --------------------------------------------------------------------------- #
# Differentiable SPD helpers
# --------------------------------------------------------------------------- #
def _sym(X):
    return 0.5 * (X + X.transpose(-1, -2))


def _eig(C):
    """Symmetric eigendecomposition with a symmetrisation guard and a fallback.

    torch.linalg.eigh backpropagates through repeated eigenvalues poorly, and
    on ill-conditioned augmented covariances the CUDA solver can fail outright
    ("algorithm failed to converge"). Shrinkage in cov_from_signal is the
    primary defence; this adds a jittered float64 retry so one bad batch cannot
    kill a whole run.
    """
    C = _sym(C)
    try:
        return torch.linalg.eigh(C)
    except Exception:
        n = C.shape[-1]
        I = torch.eye(n, device=C.device, dtype=C.dtype)
        jit = 1e-3 * C.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None].clamp(min=EPS)
        try:
            return torch.linalg.eigh(C + jit * I)
        except Exception:
            w, V = torch.linalg.eigh((C + jit * I).double().cpu())
            return w.to(C.device, C.dtype), V.to(C.device, C.dtype)


def spd_pow(C, p):
    """Matrix power of an SPD matrix via its eigendecomposition.

    `p` may be a scalar or a tensor broadcastable over the leading dims, which
    lets a whole band-stack be raised to per-band powers in ONE batched
    eigendecomposition instead of a Python loop.
    """
    w, V = _eig(C)
    w = torch.clamp(w, min=EPS)
    if torch.is_tensor(p):
        p = p.reshape(*p.shape, *([1] * (w.dim() - p.dim())))
    return _sym(V @ torch.diag_embed(w.pow(p)) @ V.transpose(-1, -2))


def spd_log(C):
    w, V = _eig(C)
    w = torch.clamp(w, min=EPS)
    return _sym(V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2))


def cov_from_signal(x, ridge=1e-4, shrink=None):
    """(B, C, T) -> (B, C, C) trace-normalised covariance.

    `shrink` applies Ledoit-Wolf style shrinkage toward the scaled identity:

        C <- (1 - g) C + g * (tr(C)/n) I

    This is not cosmetic. A delay-embedded covariance is 180x180 estimated from
    200 samples, and lagged copies x(t), x(t-tau) are near-duplicates, so the
    matrix is rank-deficient with tightly clustered eigenvalues -- exactly the
    regime where torch.linalg.eigh fails to converge (observed on the
    high-capacity arms). Shrinkage separates the spectrum and makes the
    eigendecomposition well posed, at the cost of a controlled bias.

    A plain additive ridge does NOT fix this: it shifts every eigenvalue
    equally and leaves the clustering intact. Shrinkage pulls the small
    eigenvalues toward the mean, which is what the solver needs.
    """
    x = x - x.mean(-1, keepdim=True)
    C = x @ x.transpose(-1, -2) / x.shape[-1]
    n = C.shape[-1]
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=EPS)[:, None, None]
    C = C / tr * n
    I = torch.eye(n, device=C.device, dtype=C.dtype)
    if shrink is None:
        # scale shrinkage with how under-determined the estimate is
        shrink = 0.0 if n < x.shape[-1] // 2 else min(0.5, float(n) / x.shape[-1])
    if shrink > 0:
        C = (1.0 - shrink) * C + shrink * I
    return _sym(C) + ridge * I


def tangent_vec(C):
    """Upper triangle of the matrix log, with off-diagonals scaled by sqrt(2)
    so the vector's Euclidean norm equals the matrix Frobenius norm."""
    L = spd_log(C)
    n = L.shape[-1]
    iu = torch.triu_indices(n, n, device=L.device)
    w = torch.ones(n, n, device=L.device, dtype=L.dtype)
    w = w + (torch.triu(w, 1) * (np.sqrt(2.0) - 1.0))
    return (L * w)[..., iu[0], iu[1]]


# --------------------------------------------------------------------------- #
# Learnable alignment -- the novel layer
# --------------------------------------------------------------------------- #
class LearnableAlignment(nn.Module):
    """Generalised, differentiable Euclidean Alignment.

        G = ((1 - a) I + a M)^b        C_aligned = G^(-1/2) C G^(-1/2)

    M is the mean covariance of the current batch/session, computed WITHOUT
    labels, so this remains applicable at deployment on an unseen session --
    the property that makes Euclidean Alignment work for cross-session drift in
    the first place.

    a in [0,1] interpolates between no alignment (a=0) and full whitening by the
    session mean (a=1). b rescales how aggressively the reference is inverted.
    (a=1, b=1) reproduces Euclidean Alignment exactly, so the strongest known
    method is a specific point in this layer's parameter space; training can
    only move away from it if that improves the objective.

    Both are per-band, so mu and beta can be aligned by different amounts --
    which matters here, because movement leakage was measured to be concentrated
    in beta (R^2 +0.101) and near-absent in mu (-0.013).
    """

    def __init__(self, n_bands=1, init_a=1.0, init_b=1.0):
        super().__init__()
        # a is stored directly and clamped, NOT squashed through a sigmoid.
        # Initialising at the desired a=1 (exact Euclidean Alignment) puts a
        # sigmoid at saturation, where its gradient is ~2e-6 and the layer can
        # never learn away from EA -- which is the entire point of making the
        # alignment learnable. A clamp keeps a in range with unit gradient
        # inside it.
        self.a = nn.Parameter(torch.full((n_bands,), float(init_a)))
        self.b_raw = nn.Parameter(torch.full((n_bands,), float(np.log(np.expm1(init_b)))))

    def coefficients(self):
        return self.a.clamp(0.0, 1.0), F.softplus(self.b_raw)

    def forward(self, C, ref=None):
        """C: (B, K, n, n) covariances for K bands. ref: optional (K, n, n)
        reference computed elsewhere (e.g. the whole session rather than the
        minibatch); defaults to the batch mean."""
        a, b = self.coefficients()
        M = C.mean(0) if ref is None else ref                  # (K, n, n)
        n = C.shape[-1]
        I = torch.eye(n, device=C.device, dtype=C.dtype).expand_as(M)
        # Batched over bands. The previous Python loop issued 2 eigendecompo-
        # sitions of an n x n matrix per band SERIALLY (16 for 8 bands, plus
        # their backwards), which left the GPU at ~1% while the CPU saturated
        # on dispatch: 226 s to train one subject against 4 s for the 3-band
        # model. torch.linalg.eigh is batched, so one call does all bands.
        G = (1 - a).view(-1, 1, 1) * I + a.view(-1, 1, 1) * M
        G = spd_pow(_sym(G) + EPS * I, b)
        Gi = spd_pow(G, -0.5).unsqueeze(0)                     # (1, K, n, n)
        return _sym(Gi @ C @ Gi.transpose(-1, -2))


# --------------------------------------------------------------------------- #
# SPD-preserving layers
# --------------------------------------------------------------------------- #
class BiMap(nn.Module):
    """C -> W^T C W with W semi-orthogonal. Maps SPD(n) to SPD(m), m < n.

    W is re-orthogonalised by QR on every forward pass, which keeps the output
    positive definite without a Stiefel-manifold optimiser.
    """

    def __init__(self, n_in, n_out, n_bands=1):
        super().__init__()
        W = torch.randn(n_bands, n_in, n_out)
        self.W = nn.Parameter(torch.linalg.qr(W)[0])

    def forward(self, C):
        Q, _ = torch.linalg.qr(self.W)                          # (K, n_in, n_out)
        return _sym(Q.transpose(-1, -2).unsqueeze(0) @ C @ Q.unsqueeze(0))


class ReEig(nn.Module):
    """Eigenvalue rectification: the SPD-manifold analogue of ReLU. Floors the
    spectrum, which both adds nonlinearity and conditions the matrices."""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, C):
        w, V = _eig(C)
        w = torch.clamp(w, min=self.eps)
        return _sym(V @ torch.diag_embed(w) @ V.transpose(-1, -2))


class CrossBandCoupling(nn.Module):
    """Mix bands on the manifold via a convex combination with learned weights,
    keeping the result SPD (a positive combination of SPD matrices is SPD).

    Cross-frequency structure is otherwise lost: concatenating tangent vectors
    per band lets a linear classifier weight bands but never lets them interact.
    """

    def __init__(self, n_bands):
        super().__init__()
        self.mix = nn.Parameter(torch.eye(n_bands) * 3.0)

    def forward(self, C):
        w = torch.softmax(self.mix, dim=-1)                     # (K, K)
        return torch.einsum("kj,bjmn->bkmn", w, C)


# --------------------------------------------------------------------------- #
# Full network
# --------------------------------------------------------------------------- #
class ASPDNet(nn.Module):
    """Aligned SPD Network.

    Input is the raw window; band covariances are formed inside the model so the
    alignment layer sits in the gradient path.
    """

    BANDS = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))

    def __init__(self, n_chan=60, n_classes=2, dims=(32, 16), sfreq=100.0,
                 bands=None, p_drop=0.3, learn_align=True, couple=True,
                 cond_cov=True, n_ref=4, selective=True):
        super().__init__()
        self.bands = list(bands or self.BANDS)
        K = len(self.bands)
        self.sfreq = sfreq
        self.register_buffer("filters", self._make_filters(n_chan))
        self.cond = ConditionalCovariance(K) if cond_cov else None
        self.align = LearnableAlignment(K) if learn_align else None
        self.couple = CrossBandCoupling(K) if couple else None

        layers, d_prev = [], n_chan
        for d in dims:
            layers += [BiMap(d_prev, d, K), ReEig()]
            d_prev = d
        self.spd = nn.Sequential(*layers)
        self.d_out = d_prev
        feat = K * d_prev * (d_prev + 1) // 2
        self.norm = nn.LayerNorm(feat)
        self.drop = nn.Dropout(p_drop)
        self.classify = nn.Linear(feat, n_classes)
        # selective head: SelectiveNet-style gate, trained jointly under a
        # coverage constraint so the model can decline to act
        self.select = (nn.Sequential(nn.Linear(feat, 32), nn.ELU(),
                                     nn.Dropout(p_drop), nn.Linear(32, 1))
                       if selective else None)
        self.feat_dim = feat

    def _make_filters(self, n_chan):
        """FIR band-pass bank as a fixed buffer (windowed-sinc), so band
        decomposition happens on-device without leaving the graph."""
        from scipy.signal import firwin
        taps = 51
        fs = []
        for lo, hi in self.bands:
            fs.append(firwin(taps, [lo, hi], pass_zero=False, fs=self.sfreq))
        return torch.tensor(np.stack(fs), dtype=torch.float32).unsqueeze(1)

    def _decompose(self, x):
        """(B, C, T) -> (B, K, C, T) through the fixed FIR band bank."""
        B, C, T = x.shape
        z = x.reshape(B * C, 1, T)
        z = F.conv1d(z, self.filters, padding=self.filters.shape[-1] // 2)
        return z.reshape(B, C, len(self.bands), -1).permute(0, 2, 1, 3)

    def covariances(self, x, m=None):
        """Band covariances, conditioned on motion when the layer is enabled
        and a synchronised motion waveform is supplied."""
        if x.ndim == 4:
            x = x.squeeze(1)
        xb = self._decompose(x)
        if self.cond is not None and m is not None:
            return self.cond(xb, self._decompose(m))
        return torch.stack([cov_from_signal(xb[:, k]) for k in range(xb.shape[1])], 1)

    def features(self, x, m=None, ref=None):
        C = self.covariances(x, m)
        if self.align is not None:
            C = self.align(C, ref)
        if self.couple is not None:
            C = self.couple(C)
        C = self.spd(C)
        return self.norm(tangent_vec(C).flatten(1))

    def forward(self, x, m=None, ref=None):
        z = self.drop(self.features(x, m, ref))
        logits = self.classify(z)
        if self.select is None:
            return logits
        return logits, torch.sigmoid(self.select(z)).squeeze(-1)

    def module_state(self):
        return {"lam": self.cond.state() if self.cond is not None else None,
                "align": self.alignment_state()}

    @torch.no_grad()
    def session_reference(self, x, batch=128):
        """Mean band covariance over a whole session -- label-free, so it can be
        computed on unseen test data exactly as Euclidean Alignment does."""
        acc, n = None, 0
        for i in range(0, len(x), batch):
            C = self.covariances(x[i:i + batch])
            acc = C.sum(0) if acc is None else acc + C.sum(0)
            n += len(C)
        return acc / max(n, 1)

    def alignment_state(self):
        if self.align is None:
            return None
        a, b = self.align.coefficients()
        return {"a": a.detach().cpu().numpy().round(3).tolist(),
                "b": b.detach().cpu().numpy().round(3).tolist()}


def ea_equivalent(model):
    """True when the learned alignment has stayed at Euclidean Alignment.
    If training moves a or b away from (1, 1) and accuracy improves, the
    learnable layer is doing something EA cannot."""
    st = model.alignment_state()
    if st is None:
        return None
    return all(abs(x - 1.0) < 0.05 for x in st["a"] + st["b"])


# --------------------------------------------------------------------------- #
# Conditional covariance -- removes the movement-explained second-order
# structure per window, in closed form, on the manifold
# --------------------------------------------------------------------------- #
class ConditionalCovariance(nn.Module):
    """Schur-complement conditioning of the EEG covariance on measured motion.

        C_resid = C_ee - lam * C_em C_mm^-1 C_me

    C_ee is the EEG covariance, C_mm the motion covariance and C_em their
    cross-covariance, all computed on the SAME window. The subtracted term is
    exactly the part of the EEG's second-order structure linearly explained by
    movement, so what remains is the covariance CONDITIONED on motion.

    Why this is the right place to do it: for band-limited, near-Gaussian EEG
    the covariance is a sufficient statistic, so removing the motion-explained
    block removes the movement contribution at the level of the statistic the
    decoder actually consumes -- rather than penalising it in a loss, which was
    measured to fail here (adversarial, HSIC, decorrelation and learned
    cancellation all lost to a closed-form whitening).

    lam is learned PER BAND and squashed to [0, 1]. This is the part fixed
    preprocessing cannot do: movement leakage was measured to be concentrated in
    beta (intent->motion R^2 +0.101) and near-absent in mu (-0.013), so the
    right amount of conditioning differs by band. lam = 0 disables the layer,
    lam = 1 is full conditioning.

    Positive-definiteness: the Schur complement of an SPD joint covariance is
    itself SPD, and for lam < 1 the result is C_ee minus a fraction of that
    term, which stays SPD. A ridge is added for numerical safety.
    """

    def __init__(self, n_bands=1, init_lam=0.5, ridge=1e-4):
        super().__init__()
        self.lam = nn.Parameter(torch.full((n_bands,), float(init_lam)))
        self.ridge = ridge

    def coefficients(self):
        return self.lam.clamp(0.0, 1.0)

    def forward(self, x_band, m_band):
        """x_band (B,K,C,T) band-passed EEG, m_band (B,K,R,T) band-passed motion
        from the SAME window. Returns the conditioned covariance (B,K,C,C).

        Every block is computed from the raw band signals here. Trace
        normalisation is applied AFTER the Schur complement, not before -- a
        pre-normalised C_ee minus an unnormalised explained term is not positive
        definite (measured: min eigenvalue -55.9), because the two are on
        different scales.
        """
        lam = self.coefficients()
        xc = x_band - x_band.mean(-1, keepdim=True)
        mc = m_band - m_band.mean(-1, keepdim=True)
        T = xc.shape[-1]
        C_ee = xc @ xc.transpose(-1, -2) / T                      # (B,K,C,C)
        C_em = xc @ mc.transpose(-1, -2) / T                      # (B,K,C,R)
        C_mm = mc @ mc.transpose(-1, -2) / T                      # (B,K,R,R)

        R = C_mm.shape[-1]
        Ir = torch.eye(R, device=C_mm.device, dtype=C_mm.dtype)
        scale = C_mm.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=EPS)
        C_mm = _sym(C_mm) + self.ridge * scale[..., None, None] * Ir
        explained = _sym(C_em @ torch.linalg.solve(C_mm, C_em.transpose(-1, -2)))

        out = _sym(C_ee) - lam.view(1, -1, 1, 1) * explained
        n = out.shape[-1]
        In = torch.eye(n, device=out.device, dtype=out.dtype)
        tr = out.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=EPS)
        out = out / tr[..., None, None] * n
        return _sym(out) + self.ridge * In

    def state(self):
        return self.coefficients().detach().cpu().numpy().round(3).tolist()


# --------------------------------------------------------------------------- #
# Capacity: components that capture MORE information, not just more parameters
# --------------------------------------------------------------------------- #
def delay_embed(x, order=2, tau=4):
    """Stack time-lagged copies: (B,C,T) -> (B, C*(order+1), T).

    The plain spatial covariance is instantaneous -- it throws away every
    temporal relationship inside the window. The covariance of a delay-embedded
    signal contains the lagged cross-covariances, so it captures spectral shape
    and directed channel interactions that C = XX^T/T cannot represent. This is
    the augmented-covariance idea, and it multiplies the information content of
    the SPD representation rather than just its parameter count.
    """
    parts = [x]
    for k in range(1, order + 1):
        parts.append(torch.roll(x, shifts=k * tau, dims=-1))
    return torch.cat(parts, dim=1)


class MultiScaleCovariance(nn.Module):
    """Covariances at several temporal scales within the window, fused on the
    manifold.

    One covariance per window assumes stationarity across the whole 2 s. Walk
    and stop differ in their temporal dynamics, so splitting the window into
    halves and quarters and combining those covariances retains non-stationary
    structure that a single estimate averages away.
    """

    def __init__(self, scales=(1, 2, 4), learn_weights=True):
        super().__init__()
        self.scales = scales
        self.w = nn.Parameter(torch.zeros(len(scales)), requires_grad=learn_weights)

    def forward(self, x):                      # (B, C, T)
        w = torch.softmax(self.w, 0)
        out = 0.0
        for i, s in enumerate(self.scales):
            T = x.shape[-1] // s * s
            xs = x[..., :T].reshape(x.shape[0], x.shape[1], s, T // s)
            c = torch.stack([cov_from_signal(xs[:, :, j]) for j in range(s)], 0).mean(0)
            out = out + w[i] * c
        return _sym(out)


class BandAttention(nn.Module):
    """Multi-head attention across band tokens in the tangent space.

    CrossBandCoupling can only form convex combinations of whole covariance
    matrices -- it cannot express that mu should attend to beta differently
    depending on the window. Attention on the tangent vectors can, and
    cross-frequency coupling is a real phenomenon in sensorimotor rhythms.
    """

    def __init__(self, dim, heads=4, p_drop=0.2):
        super().__init__()
        heads = max(1, min(heads, dim // 16))
        while dim % heads:
            heads -= 1
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=p_drop)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                nn.Dropout(p_drop), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tok):                    # (B, K, dim) tangent vectors per band
        a, _ = self.attn(tok, tok, tok)
        tok = self.norm(tok + a)
        return self.norm2(tok + self.ff(tok))


class RiemannianReadout(nn.Module):
    """Tangent vector PLUS log-determinant and per-band trace statistics.

    The tangent projection is taken at the identity for every window; the
    log-det and trace carry global scale information that the trace-normalised
    tangent vector deliberately discards, and they are cheap.
    """

    def forward(self, C):                      # (B, K, n, n)
        t = tangent_vec(C).flatten(1)
        w, _ = _eig(C)
        w = torch.clamp(w, min=EPS)
        logdet = torch.log(w).sum(-1)                       # (B, K)
        tr = w.sum(-1)
        return torch.cat([t, logdet, torch.log(tr.clamp(min=EPS))], dim=-1)


# --------------------------------------------------------------------------- #
# Large configuration
# --------------------------------------------------------------------------- #
class ASPDNetLarge(nn.Module):
    """High-capacity ASPD-Net.

    Capacity is added where it buys INFORMATION, not merely parameters:

      bands 3 -> 8          finer spectral resolution; the band series showed
                            leakage and signal are distributed unevenly across
                            frequency, so 3 bands is a coarse summary
      delay embedding       augmented covariance carries lagged cross-channel
                            structure; a plain spatial covariance is
                            instantaneous and discards all of it
      multi-scale windows   covariances at 1/2/4 sub-windows, fused, so
                            non-stationarity inside the 2 s window survives
      deeper SPD stack      60 -> 48 -> 32 -> 24 -> 16 instead of 60 -> 32 -> 16
      band attention        cross-frequency interaction that a convex mixing
                            matrix cannot express
      richer readout        tangent + log-det + log-trace per band

    The previous capacity evidence in this project came from CNNs on RAW
    signals, where more parameters bought movement leakage. That does not
    transfer to the manifold: this is the representation where depth was
    measured to help (+0.047 over a linear model), and capacity here has never
    been tested.
    """

    BANDS_8 = ((4, 8), (8, 10), (10, 13), (13, 16), (16, 20),
               (20, 25), (25, 30), (30, 40))

    def __init__(self, n_chan=60, n_classes=2, dims=(48, 32, 24, 16), sfreq=100.0,
                 bands=None, p_drop=0.3, n_ref=4, cond_cov=True, learn_align=True,
                 attend=True, selective=True, delay_order=2, delay_tau=4,
                 multiscale=True, heads=4):
        super().__init__()
        self.bands = list(bands or self.BANDS_8)
        K = len(self.bands)
        self.sfreq = sfreq
        self.delay_order, self.delay_tau = delay_order, delay_tau
        self.register_buffer("filters", self._make_filters())

        n_aug = n_chan * (delay_order + 1)
        self.cond = ConditionalCovariance(K) if cond_cov else None
        self.ms = MultiScaleCovariance() if multiscale else None
        self.align = LearnableAlignment(K) if learn_align else None

        layers, d_prev = [], n_aug
        for d in dims:
            layers += [BiMap(d_prev, d, K), ReEig()]
            d_prev = d
        self.spd = nn.Sequential(*layers)
        self.d_out = d_prev

        tri = d_prev * (d_prev + 1) // 2
        self.attend = BandAttention(tri, heads, p_drop) if attend else None
        self.readout = RiemannianReadout()
        feat = K * tri + 2 * K
        self.norm = nn.LayerNorm(feat)
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Sequential(nn.Linear(feat, 256), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(256, 64), nn.GELU())
        self.classify = nn.Linear(64, n_classes)
        self.select = (nn.Sequential(nn.Linear(64, 32), nn.GELU(),
                                     nn.Dropout(p_drop), nn.Linear(32, 1))
                       if selective else None)
        self.feat_dim = feat
        self.tri = tri

    def _make_filters(self):
        from scipy.signal import firwin
        taps = 65
        fs = [firwin(taps, [lo, hi], pass_zero=False, fs=self.sfreq)
              for lo, hi in self.bands]
        return torch.tensor(np.stack(fs), dtype=torch.float32).unsqueeze(1)

    def _decompose(self, x):
        B, C, T = x.shape
        z = F.conv1d(x.reshape(B * C, 1, T), self.filters,
                     padding=self.filters.shape[-1] // 2)
        return z.reshape(B, C, len(self.bands), -1).permute(0, 2, 1, 3)

    def covariances(self, x, m=None):
        if x.ndim == 4:
            x = x.squeeze(1)
        xb = self._decompose(x)                             # (B,K,C,T)
        if self.cond is not None and m is not None:
            xb_c = xb                                        # condition pre-embedding
            C0 = self.cond(xb_c, self._decompose(m))
            # conditioning returns a C x C covariance; rebuild the augmented
            # covariance from the conditioned signal by whitening the band
            # signal with the conditioned/unconditioned ratio is not exact, so
            # the delay embedding is applied to the raw band signal and the
            # conditioned block is used for the un-lagged sub-block only.
            aug = torch.stack([
                cov_from_signal(delay_embed(xb[:, k], self.delay_order, self.delay_tau))
                for k in range(xb.shape[1])], 1)
            n = C0.shape[-1]
            aug = aug.clone()
            aug[..., :n, :n] = C0
            return _sym(aug)
        if self.ms is not None:
            return torch.stack([
                self.ms(delay_embed(xb[:, k], self.delay_order, self.delay_tau))
                for k in range(xb.shape[1])], 1)
        return torch.stack([
            cov_from_signal(delay_embed(xb[:, k], self.delay_order, self.delay_tau))
            for k in range(xb.shape[1])], 1)

    def features(self, x, m=None, ref=None):
        C = self.covariances(x, m)
        if self.align is not None:
            C = self.align(C, ref)
        C = self.spd(C)
        if self.attend is not None:
            t = tangent_vec(C)                               # (B,K,tri)
            t = self.attend(t)
            w, _ = _eig(C)
            w = torch.clamp(w, min=EPS)
            extra = torch.cat([torch.log(w).sum(-1),
                               torch.log(w.sum(-1).clamp(min=EPS))], dim=-1)
            z = torch.cat([t.flatten(1), extra], dim=-1)
        else:
            z = self.readout(C)
        return self.norm(z)

    def forward(self, x, m=None, ref=None):
        z = self.head(self.drop(self.features(x, m, ref)))
        logits = self.classify(z)
        if self.select is None:
            return logits
        return logits, torch.sigmoid(self.select(z)).squeeze(-1)

    def module_state(self):
        return {"lam": self.cond.state() if self.cond is not None else None,
                "align": (None if self.align is None else
                          {k: v for k, v in zip(("a", "b"),
                           [c.detach().cpu().numpy().round(3).tolist()
                            for c in self.align.coefficients()])})}


# --------------------------------------------------------------------------- #
# Cholesky-metric variant -- same manifold, no eigendecomposition
# --------------------------------------------------------------------------- #
def chol_tangent(C):
    """Log-Cholesky tangent coordinates of an SPD matrix (Lin, 2019).

    C = L L^T with L lower-triangular; the tangent representation is the strict
    lower triangle of L together with log(diag(L)). This is a genuine Riemannian
    metric on SPD matrices, not an approximation of the log-Euclidean one -- it
    simply uses a different (and cheaper) chart.

    Motivation is measured, not aesthetic: linalg_eigh accounted for 95% of this
    model's runtime (54.9 s of 58 s under the profiler), and it is latency-bound
    on batches of small matrices, which is why the GPU sat at ~13%. Cholesky on
    the same shapes is 47x faster at 180x180 and 6.5x faster through
    forward+backward.
    """
    L = torch.linalg.cholesky(C)
    n = L.shape[-1]
    iu = torch.tril_indices(n, n, offset=-1, device=L.device)
    off = L[..., iu[0], iu[1]]
    dia = torch.log(L.diagonal(dim1=-2, dim2=-1).clamp(min=EPS))
    return torch.cat([off, dia], dim=-1)


class CholAlign(nn.Module):
    """Alignment by Cholesky whitening: C -> L^-1 C L^-T where G = L L^T.

    Equivalent in purpose to the symmetric-square-root whitening of
    LearnableAlignment (it removes the reference's second-order structure) but
    uses a triangular solve instead of two eigendecompositions. The difference
    is a rotation, which the learned BiMap layers downstream absorb.
    """

    def __init__(self, n_bands=1, init_a=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((n_bands,), float(init_a)))

    def coefficients(self):
        return self.a.clamp(0.0, 1.0)

    def forward(self, C, ref=None):
        a = self.coefficients()
        M = C.mean(0) if ref is None else ref
        n = C.shape[-1]
        I = torch.eye(n, device=C.device, dtype=C.dtype).expand_as(M)
        G = (1 - a).view(-1, 1, 1) * I + a.view(-1, 1, 1) * M
        L = torch.linalg.cholesky(_sym(G) + EPS * I).unsqueeze(0)     # (1,K,n,n)
        Y = torch.linalg.solve_triangular(L, C, upper=False)
        Y = torch.linalg.solve_triangular(L, Y.transpose(-1, -2), upper=False)
        return _sym(Y)


class ASPDNetChol(nn.Module):
    """ASPD-Net in the log-Cholesky chart. Eigendecomposition-free.

    ReEig is dropped: its role was to floor the spectrum for conditioning, which
    the covariance shrinkage already does, and it cost one batched eigh per
    layer. Nonlinearity now comes from the log in the Cholesky readout and the
    classifier head.
    """

    BANDS_8 = ASPDNetLarge.BANDS_8

    def __init__(self, n_chan=60, n_classes=2, dims=(48, 24), sfreq=100.0,
                 bands=None, p_drop=0.3, n_ref=4, cond_cov=True, learn_align=True,
                 attend=True, selective=True, delay_order=2, delay_tau=4, heads=4):
        super().__init__()
        self.bands = list(bands or self.BANDS_8)
        K = len(self.bands)
        self.sfreq = sfreq
        self.delay_order, self.delay_tau = delay_order, delay_tau
        self.register_buffer("filters", ASPDNetLarge._make_filters(self))
        n_aug = n_chan * (delay_order + 1)
        self.cond = ConditionalCovariance(K) if cond_cov else None
        self.align = CholAlign(K) if learn_align else None
        layers, d = [], n_aug
        for o in dims:
            layers.append(BiMap(d, o, K)); d = o
        self.bimaps = nn.ModuleList(layers)
        tri = d * (d + 1) // 2
        self.attend = BandAttention(tri, heads, p_drop) if attend else None
        feat = K * tri
        self.norm = nn.LayerNorm(feat)
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Sequential(nn.Linear(feat, 256), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(256, 64), nn.GELU())
        self.classify = nn.Linear(64, n_classes)
        self.select = (nn.Sequential(nn.Linear(64, 32), nn.GELU(),
                                     nn.Dropout(p_drop), nn.Linear(32, 1))
                       if selective else None)
        self.feat_dim, self.tri = feat, tri

    def _decompose(self, x):
        B, C, T = x.shape
        z = F.conv1d(x.reshape(B * C, 1, T), self.filters,
                     padding=self.filters.shape[-1] // 2)
        return z.reshape(B, C, len(self.bands), -1).permute(0, 2, 1, 3)

    def covariances(self, x, m=None):
        if x.ndim == 4:
            x = x.squeeze(1)
        xb = self._decompose(x)
        aug = torch.stack([
            cov_from_signal(delay_embed(xb[:, k], self.delay_order, self.delay_tau))
            for k in range(xb.shape[1])], 1)
        if self.cond is not None and m is not None:
            C0 = self.cond(xb, self._decompose(m))
            n = C0.shape[-1]
            aug = aug.clone()
            aug[..., :n, :n] = C0
        return _sym(aug)

    def features(self, x, m=None, ref=None):
        C = self.covariances(x, m)
        if self.align is not None:
            C = self.align(C, ref)
        for bm in self.bimaps:
            C = bm(C)
            n = C.shape[-1]
            C = C + EPS * torch.eye(n, device=C.device, dtype=C.dtype)
        t = chol_tangent(C)
        if self.attend is not None:
            t = self.attend(t)
        return self.norm(t.flatten(1))

    def forward(self, x, m=None, ref=None):
        z = self.head(self.drop(self.features(x, m, ref)))
        logits = self.classify(z)
        if self.select is None:
            return logits
        return logits, torch.sigmoid(self.select(z)).squeeze(-1)

    def module_state(self):
        return {"lam": self.cond.state() if self.cond is not None else None,
                "align": (None if self.align is None else
                          {"a": self.align.coefficients().detach().cpu()
                           .numpy().round(3).tolist()})}
