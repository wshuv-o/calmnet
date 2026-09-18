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

    def __init__(self, n_chan, momentum=None, gate_init=2.0):
        import os as _os
        if momentum is None:
            momentum = float(_os.environ.get("DN_MOMENTUM", "0.2"))
        super().__init__()
        self.register_buffer("run_cov", torch.eye(n_chan))
        self.register_buffer("primed", torch.zeros(1))
        self.momentum = momentum
        # sigmoid(2.0) ~ 0.88: start mostly aligned, let the model back off
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.n_chan = n_chan

    def _cov(self, x, per_window=True):       # (N, C, T) -> (C, C)
        """Mean spatial covariance over a batch of windows.

        `per_window` normalises EACH window's covariance before averaging, so
        every window contributes equally regardless of its power. Summing raw
        covariances and normalising once at the end instead weights the estimate
        by power, which is how this layer failed on the MoBI cohort: walking
        windows are 87.7% of that data and carry 1.52x the power of standing
        windows, so they supplied 91.6% of a power-weighted estimate. Whitening
        by an estimate that is essentially one class removes the very variance
        that separates the classes -- measured at -0.166 accuracy there, against
        +0.051 on the more balanced cohort.

        Equal weighting does not make the estimate class-balanced (87.7% is
        still a majority) but it removes the extra power-driven skew on top of
        the count imbalance, without needing labels.
        """
        xc = x - x.mean(dim=-1, keepdim=True)
        if per_window:
            c = torch.einsum("nct,ndt->ncd", xc, xc) / x.shape[-1]
            tr = torch.diagonal(c, dim1=1, dim2=2).mean(-1)[:, None, None]
            return (c / (tr + EPS)).mean(0)
        c = torch.einsum("nct,ndt->cd", xc, xc) / (x.shape[0] * x.shape[-1])
        return c / (torch.diagonal(c).mean() + EPS)

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


class TangentBranch(nn.Module):
    """Second-order branch: per-window covariance -> tangent space -> linear.

    Why this module and not another convolutional block
    ---------------------------------------------------
    The stem reads log band power, which is a diagonal summary: it sees how
    much each channel carries in each band and nothing about how channels
    covary. A probe settled that this is a real omission rather than a
    theoretical one. Logistic regression on log-Euclidean tangent vectors
    reached 0.982 on sub-01 where the trained network reaches 0.961, on
    strictly less data, so the information is present in the covariance and the
    network is not currently reading it.

    Reference point
    ---------------
    The tangent map needs a base point on the SPD manifold. AdaptiveAlignment
    already maintains one -- the running covariance -- which is updated
    label-free and keeps updating at test time, so the branch inherits the
    paper's adaptation mechanism instead of introducing a second one. When
    alignment is disabled the branch keeps its own running mean under the same
    momentum, so the ablation isolates the branch rather than silently removing
    adaptation as well.

    Why the tangent map carries no gradient
    ---------------------------------------
    The map itself has no parameters: it is covariance, shrinkage, whitening by
    the reference, matrix logarithm, vectorisation. Only the projection that
    follows is learned. Backward through torch.linalg.eigh carries
    1 / (lambda_i - lambda_j) terms which explode on near-degenerate spectra,
    and 60-channel covariances estimated from 32-window batches are routinely
    near-degenerate. Detaching costs nothing that was measured to matter: the
    probe that motivated this branch was itself a linear map on fixed tangent
    features, and that is exactly what is being trained here, jointly with the
    rest of the network rather than after it.

    Shrinkage
    ---------
    Covariances are shrunk toward a scaled identity before the logarithm. With
    60 channels from 400 samples the sample covariance is poorly conditioned,
    and the existing code handles that by clamping eigenvalues at 1e-6, which
    is a floor rather than an estimator. Shrinkage also bounds the condition
    number the logarithm sees.
    """

    def __init__(self, n_chan, d_out, shrink=None, momentum=None, dropout=0.3,
                 ref_mode=None):
        import os as _os
        super().__init__()
        if momentum is None:
            momentum = float(_os.environ.get("DN_MOMENTUM", "0.2"))
        if shrink is None:
            shrink = float(_os.environ.get("DN_TAN_SHRINK", "0.1"))
        # Where the base point of the tangent map comes from. The running
        # reference collapsed on cohort B (0.744 -> 0.565, barely above the
        # 0.500 chance line) and the cause looks structural rather than
        # incidental: that cohort is 88 % Walk in blocks of 309 windows, so an
        # EMA over the test stream becomes the Walk covariance, and whitening a
        # Walk window by the Walk covariance removes the very variance that
        # separates the classes. That is the failure this project already
        # documents for the alignment layer, reappearing because the branch was
        # built on the same statistic.
        #
        #   running  EMA over the test stream. Adapts, and tracks the class.
        #   frozen   mean covariance of the FITTING split, fixed thereafter.
        #            The fitting split is a pool rather than a stream, so it
        #            cannot chase a class block. Gives up test-time adaptation,
        #            which on cohort B was already costing accuracy.
        #   none     no whitening: log of the trace-normalised covariance.
        #            No reference exists to be corrupted.
        if ref_mode is None:
            ref_mode = _os.environ.get("DN_TAN_REF", "running")
        if ref_mode not in ("running", "frozen", "none"):
            raise ValueError("DN_TAN_REF must be running, frozen or none")
        self.ref_mode = ref_mode
        self.n_chan, self.shrink, self.momentum = n_chan, shrink, momentum
        self.register_buffer("run_cov", torch.eye(n_chan))
        self.register_buffer("primed", torch.zeros(1))
        iu = torch.triu_indices(n_chan, n_chan)
        self.register_buffer("iu", iu)
        # Off-diagonal entries are counted once but appear twice in the matrix,
        # so they are scaled by sqrt(2) to keep the vector's Euclidean norm
        # equal to the matrix Frobenius norm. Without it the diagonal is
        # over-weighted and the branch drifts back toward being a power feature.
        w = torch.full((iu.shape[1],), 2.0 ** 0.5)
        w[iu[0] == iu[1]] = 1.0
        self.register_buffer("vec_w", w)
        d_in = n_chan * (n_chan + 1) // 2
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out),
                                  nn.GELU(), nn.Dropout(dropout))
        self.d_in, self.d_out = d_in, d_out

    def _shrunk_cov(self, x):
        """Per-window covariance, trace-normalised then shrunk to identity."""
        xc = x - x.mean(dim=-1, keepdim=True)
        c = torch.einsum("nct,ndt->ncd", xc, xc) / x.shape[-1]
        tr = torch.diagonal(c, dim1=1, dim2=2).mean(-1)[:, None, None]
        c = c / (tr + EPS)
        eye = torch.eye(self.n_chan, device=c.device, dtype=c.dtype)
        return (1.0 - self.shrink) * c + self.shrink * eye

    @torch.no_grad()
    def freeze_reference(self, x, chunk=256):
        """Set the base point from the fitting split, once, before training.

        Called with fitting data only, so nothing from the held-out sessions
        enters the reference. Averaged over the whole split in chunks: a
        subsample would make the base point depend on which windows were drawn,
        and the point of this mode is that it does not move.
        """
        tot, n = None, 0
        for i in range(0, len(x), chunk):
            c = self._shrunk_cov(x[i:i + chunk].detach().double()).sum(0)
            tot = c if tot is None else tot + c
            n += len(x[i:i + chunk])
        self.run_cov.copy_((tot / max(n, 1)).to(self.run_cov.dtype))
        self.primed.fill_(1)

    def forward(self, x, ref=None):                      # (N, C, T)
        with torch.no_grad():
            c = self._shrunk_cov(x.detach().double())
            if self.ref_mode == "none":
                # log of the covariance itself. It is already trace-normalised
                # per window, so the logarithm is well scaled without a base
                # point and there is no reference to track the streaming class.
                w, v = torch.linalg.eigh(c)
                logm = v @ torch.diag_embed(torch.clamp(w, min=1e-6).log())                     @ v.transpose(-1, -2)
                t = logm[:, self.iu[0], self.iu[1]] * self.vec_w
                return self.proj(t.to(x.dtype))
            if self.ref_mode == "frozen":
                # Whatever the caller passes, a frozen branch uses its own
                # fixed estimate: taking the alignment layer's running
                # covariance here would reintroduce the tracking this mode
                # exists to remove.
                ref = self.run_cov
            elif ref is None:
                m = c.mean(0)
                if self.primed.item() == 0:
                    self.run_cov.copy_(m.to(self.run_cov.dtype))
                    self.primed.fill_(1)
                else:
                    self.run_cov.mul_(1 - self.momentum).add_(
                        m.to(self.run_cov.dtype), alpha=self.momentum)
                ref = self.run_cov
            rw, rv = torch.linalg.eigh(ref.double())
            rinv = rv @ torch.diag(torch.clamp(rw, min=1e-6).rsqrt()) @ rv.t()
            w, v = torch.linalg.eigh(rinv @ c @ rinv)
            logm = v @ torch.diag_embed(torch.clamp(w, min=1e-6).log())                 @ v.transpose(-1, -2)
            t = logm[:, self.iu[0], self.iu[1]] * self.vec_w
        return self.proj(t.to(x.dtype))


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
                 use_align=True, use_ctx=True, use_gate=True, n_scales=3,
                 use_tangent=False, tangent_shrink=0.1):
        super().__init__()
        self.use_align, self.use_ctx, self.use_gate = use_align, use_ctx, use_gate
        self.use_tangent = use_tangent
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
        # The branch fuses back to d_model, so the classifier and the gate are
        # untouched by adding it. Their shapes stay identical with the branch
        # on or off, which is what lets the ablation be a single module in or
        # out rather than a different network.
        self.tangent = (TangentBranch(n_chan, d_model, shrink=tangent_shrink,
                                      dropout=dropout) if use_tangent else None)
        self.fuse = (nn.Linear(d_model * 2, d_model) if use_tangent else None)
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
        if self.tangent is not None:
            # Referenced to the alignment layer's running covariance when it
            # exists, so the branch adapts through the same label-free
            # statistic rather than adding a second adaptation path.
            ref = self.align.run_cov if self.align is not None else None
            t = self.tangent(f, ref).reshape(B, K, -1)[:, -1]
        z = self.stem(f)                                  # (B*K, F, T')
        z = self.frame_norm(z.transpose(1, 2))            # (B*K, T', F)
        z = self.proj(z)
        w = torch.softmax(self.frame_attn(z), dim=1)
        e = (z * w).sum(1).reshape(B, K, -1)              # (B, K, d)
        if self.use_ctx and K > 1:
            mask = torch.triu(torch.ones(K, K, device=x.device, dtype=torch.bool),
                              diagonal=1)
            e = self.ctx(self.pos(e), mask=mask)
        last = e[:, -1]
        if self.tangent is not None:
            last = self.fuse(torch.cat([last, t], dim=-1))
        h = self.drop(self.norm(last))
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
