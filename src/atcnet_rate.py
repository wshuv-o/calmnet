"""ATCNet with its temporal hyperparameters matched to OUR sampling rate.

ATCNet (Altaheri et al. 2022, IEEE TII) is the strongest decoder measured on this
task once the training set is large enough: 0.881 at 4881 windows versus 0.828 at
905. But it is being run in a configuration it was never designed for, and it is
winning anyway.

THE MISMATCH

Every temporal hyperparameter in ATCNet is specified in SAMPLES and was tuned on
BCI Competition IV-2a: 22 channels, 1125 samples at 250 Hz. Read as durations
(from the reference implementation, github.com/Altaheri/EEG-ATCNet):

    eegn_kernelSize = 64  @ 250 Hz  ->  256 ms temporal filter
    second conv     = 16  @ 250 Hz  ->   64 ms
    pooling         = 8 x 7         ->  224 ms per output frame
    1125 / 56       = 20 feature frames, over which 5 sliding windows attend

Our data is 100 Hz. Applying the same SAMPLE counts stretches every receptive
field by 2.5x, and worse, 400 samples is below the minimum the default
configuration needs (616), so braindecode silently degrades the model:

    n_windows        5 -> 3     the sliding-window ensemble, ATCNet's core idea
    tcn_kernel_size  4 -> 2
    kernel_length_1 64 -> 41
    pool sizes     8,7 -> 5,4

Losing 2 of 5 attention windows is not a cosmetic adjustment; the windowed
attention ensemble is the "AT" in ATCNet.

THE FIX

Match the durations rather than the sample counts:

    kernel_length_1 = 26   (256 ms at 100 Hz)
    kernel_length_2 =  6   ( 64 ms at 100 Hz)
    pool_size_1, _2 = 4, 5 (200 ms per frame -> 400/20 = 20 frames, as intended)

This reproduces the original's 20-frame feature map, restores the intended
receptive fields in seconds, and drops the minimum required input to 220 samples
-- below our 400 -- so n_windows=5 and tcn_kernel_size=4 survive.

It is a porting fix, not a new idea: the contribution is recognising that a
published architecture's sample-specified hyperparameters do not transfer across
sampling rates, and that the fallback path hides the problem by producing a model
that still runs.

VARIANTS

  default   braindecode's ATCNet as-is (auto-degraded)      -- the baseline
  rate      durations matched to 100 Hz                     -- the fix
  rate_nw3  rate-matched but n_windows forced back to 3     -- isolates whether
            the gain comes from restoring the ensemble or from the kernels
"""
from __future__ import annotations
import warnings

SFREQ_SRC = 250.0       # ATCNet's design sampling rate
KERN1_MS = 64 / SFREQ_SRC * 1000     # 256 ms
KERN2_MS = 16 / SFREQ_SRC * 1000     # 64 ms


def rate_matched_kwargs(sfreq, n_times, n_windows=5):
    """Temporal hyperparameters for `sfreq`, preserving ATCNet's durations."""
    k1 = max(1, int(round(KERN1_MS / 1000 * sfreq)))
    k2 = max(1, int(round(KERN2_MS / 1000 * sfreq)))
    # choose pooling so the feature map keeps ~20 frames, as at 250 Hz
    target_frames = 20
    total_pool = max(2, int(round(n_times / target_frames)))
    p1 = max(2, int(round(total_pool ** 0.5)))
    p2 = max(2, int(round(total_pool / p1)))
    return dict(conv_block_kernel_length_1=k1, conv_block_kernel_length_2=k2,
                conv_block_pool_size_1=p1, conv_block_pool_size_2=p2,
                n_windows=n_windows)


def build_atcnet(variant, n_chan, n_time, n_outputs=2, sfreq=100.0):
    import braindecode.models as B
    kw = dict(n_chans=n_chan, n_outputs=n_outputs, n_times=n_time, sfreq=sfreq)
    if variant == "default":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return B.ATCNet(**kw)
    nw = 3 if variant == "rate_nw3" else 5
    kw.update(rate_matched_kwargs(sfreq, n_time, n_windows=nw))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net = B.ATCNet(**kw)
        for m in w:
            if "smaller than the minimum" in str(m.message):
                raise RuntimeError(
                    "rate-matched config still triggers auto-degradation: %s"
                    % str(m.message)[:160])
    return net


if __name__ == "__main__":
    import torch
    for v in ("default", "rate", "rate_nw3"):
        net = build_atcnet(v, 60, 400)
        n = sum(p.numel() for p in net.parameters())
        y = net(torch.randn(2, 60, 400))
        print("%-9s params=%7d  n_windows=%d  k1=%d k2=%d pool=%d,%d tcn_k=%d  out=%s"
              % (v, n, net.n_windows, net.conv_block_kernel_length_1,
                 net.conv_block_kernel_length_2, net.conv_block_pool_size_1,
                 net.conv_block_pool_size_2, net.tcn_kernel_size, tuple(y.shape)))
