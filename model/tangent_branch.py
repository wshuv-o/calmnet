import torch
import torch.nn as nn

EPS = 1e-5


class TangentBranch(nn.Module):
    """Window covariance -> tangent space at a running reference -> linear (Eqs. 2-4).

    ref_mode: running (EMA of batch covariances), frozen (fitting-split mean) or none.
    causal: map each batch at the reference from before it, then update.
    """

    def __init__(self, n_chan, d_out, shrink=0.1, momentum=0.2, dropout=0.3,
                 ref_mode="running", causal=False):
        super().__init__()
        assert ref_mode in ("running", "frozen", "none")
        self.ref_mode, self.causal = ref_mode, causal
        self.n_chan, self.shrink, self.momentum = n_chan, shrink, momentum
        self.register_buffer("run_cov", torch.eye(n_chan))
        self.register_buffer("primed", torch.zeros(1))
        iu = torch.triu_indices(n_chan, n_chan)
        self.register_buffer("iu", iu)
        w = torch.full((iu.shape[1],), 2.0 ** 0.5)       # off-diagonal weight sqrt(2)
        w[iu[0] == iu[1]] = 1.0
        self.register_buffer("vec_w", w)
        d_in = n_chan * (n_chan + 1) // 2
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out),
                                  nn.GELU(), nn.Dropout(dropout))

    def _shrunk_cov(self, x):
        xc = x - x.mean(dim=-1, keepdim=True)
        c = torch.einsum("nct,ndt->ncd", xc, xc) / x.shape[-1]
        tr = torch.diagonal(c, dim1=1, dim2=2).mean(-1)[:, None, None]
        c = c / (tr + EPS)
        eye = torch.eye(self.n_chan, device=c.device, dtype=c.dtype)
        return (1.0 - self.shrink) * c + self.shrink * eye

    @torch.no_grad()
    def freeze_reference(self, x, chunk=256):
        tot, n = None, 0
        for i in range(0, len(x), chunk):
            c = self._shrunk_cov(x[i:i + chunk].detach().double()).sum(0)
            tot = c if tot is None else tot + c
            n += len(x[i:i + chunk])
        self.run_cov.copy_((tot / max(n, 1)).to(self.run_cov.dtype))
        self.primed.fill_(1)

    def _logm_vec(self, c):
        w, v = torch.linalg.eigh(c)
        logm = v @ torch.diag_embed(torch.clamp(w, min=1e-6).log()) @ v.transpose(-1, -2)
        return logm[:, self.iu[0], self.iu[1]] * self.vec_w

    def forward(self, x):
        with torch.no_grad():                            # the map carries no gradient
            c = self._shrunk_cov(x.detach().double())
            if self.ref_mode == "none":
                return self.proj(self._logm_vec(c).to(x.dtype))
            if self.ref_mode == "frozen":
                ref = self.run_cov
            else:
                m = c.mean(0)
                if self.primed.item() == 0:
                    self.run_cov.copy_(m.to(self.run_cov.dtype))
                    self.primed.fill_(1)
                    ref = self.run_cov
                elif self.causal:
                    ref = self.run_cov.clone()
                    self.run_cov.mul_(1 - self.momentum).add_(
                        m.to(self.run_cov.dtype), alpha=self.momentum)
                else:
                    self.run_cov.mul_(1 - self.momentum).add_(
                        m.to(self.run_cov.dtype), alpha=self.momentum)
                    ref = self.run_cov
            rw, rv = torch.linalg.eigh(ref.double())
            rinv = rv @ torch.diag(torch.clamp(rw, min=1e-6).rsqrt()) @ rv.t()
            t = self._logm_vec(rinv @ c @ rinv)
        return self.proj(t.to(x.dtype))
