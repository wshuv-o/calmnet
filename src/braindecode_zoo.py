"""Published EEG architectures from braindecode, wired into the CALM-Net
invariance harness.

Replaces the hand-rolled backbones with peer-reviewed, maintained
implementations (braindecode 1.8) so the architecture question is asked with
models a reviewer already recognises: ShallowFBCSPNet, Deep4Net, EEGNet,
EEGConformer, ATCNet, EEGNeX, EEGITNet, EEGInceptionMI, FBCNet, IFNet, CTNet,
MSVTNet, TSception, EEGSimpleConv and others, spanning 2.3k to 3.7M parameters.

Every braindecode classifier exposes `final_layer`. Swapping it for Identity
turns the model into a feature extractor, which GenericMID then splits into
intent and artefact subspaces -- so the same movement-invariance probe applies
to a 3.7M-parameter published network and to the 7k band-power baseline, and the
numbers are directly comparable.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

SFREQ = 100.0

# name -> extra constructor kwargs needed to fit 60ch x 200 samples @ 100 Hz
BD_MODELS = {
    "ShallowFBCSPNet": {},
    "Deep4Net": {},
    "EEGNet": {},
    "EEGConformer": {},
    "ATCNet": {},
    "EEGNeX": {},
    "EEGITNet": {},
    "EEGInceptionMI": {},
    "FBCNet": {},
    "IFNet": {},
    "EEGSimpleConv": {},
    "SincShallowNet": {},
    "CTNet": {},
    "MSVTNet": {},
    "TSception": {},
    "EEGTCNet": {},
    "FBMSNet": {},
    "SCCNet": {},
    "BDTCN": {},
    "FBLightConvNet": {"win_len": 100},      # default 250 > 200 samples
}


class BDBackbone(nn.Module):
    """Wrap a braindecode classifier as a feature extractor.

    Accepts (B, 1, C, T) -- the shape the rest of this codebase uses -- and
    squeezes to the (B, C, T) braindecode expects.
    """

    def __init__(self, name, n_chan=60, n_time=200, sfreq=SFREQ, **kw):
        super().__init__()
        from braindecode import models as M
        cls = getattr(M, name)
        extra = dict(BD_MODELS.get(name, {}))
        extra.update(kw)
        try:
            self.net = cls(n_chans=n_chan, n_times=n_time, n_outputs=2,
                           sfreq=sfreq, **extra)
        except TypeError:                      # a few take no sfreq
            self.net = cls(n_chans=n_chan, n_times=n_time, n_outputs=2, **extra)

        if not hasattr(self.net, "final_layer"):
            raise AttributeError(f"{name} has no final_layer to strip")
        self.net.final_layer = nn.Identity()

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.zeros(2, n_chan, n_time))
        self.feat_dim = int(out.reshape(2, -1).shape[1])
        self.net.train()
        self.name = name

    def features(self, x):
        if x.ndim == 4:
            x = x.squeeze(1)                   # (B,1,C,T) -> (B,C,T)
        return self.net(x).reshape(x.shape[0], -1)

    def forward(self, x):
        raise NotImplementedError("use GenericMID")


def build_bd_variant(name, n_chan=60, n_time=200, k_imu=12, head="linear",
                     int_frac=0.5, **kw):
    from arch_zoo import GenericMID
    bb = BDBackbone(name, n_chan=n_chan, n_time=n_time, **kw)
    return GenericMID(bb, k_imu=k_imu, head=head, int_frac=int_frac)


def probe(n_chan=60, n_time=200):
    """Return {name: (feat_dim, n_params)} for every model that instantiates."""
    ok = {}
    x = torch.randn(2, 1, n_chan, n_time)
    for name in BD_MODELS:
        try:
            m = build_bd_variant(name, n_chan, n_time)
            m(x, 1.0)
            ok[name] = (m.backbone.feat_dim,
                        sum(p.numel() for p in m.parameters()))
        except Exception as e:
            ok[name] = (None, f"{type(e).__name__}: {str(e)[:70]}")
    return ok


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    r = probe()
    good = {k: v for k, v in r.items() if v[0] is not None}
    print(f"{len(good)}/{len(r)} braindecode models usable as MID backbones\n")
    print(f"{'model':20}{'feat_dim':>10}{'params':>12}")
    for k, (d, p) in sorted(good.items(), key=lambda kv: kv[1][1]):
        print(f"{k:20}{d:>10}{p:>12,}")
    bad = {k: v for k, v in r.items() if v[0] is None}
    if bad:
        print(f"\nunusable ({len(bad)}):")
        for k, (_, e) in bad.items():
            print(f"  {k:20} {e}")
