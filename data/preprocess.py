import numpy as np
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi, sosfiltfilt


def bandpass_zero_phase(x, fs, band, order=4):
    sos = butter(order, band, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def bandpass_ba_zero_phase(x, fs, band, order=4):
    b, a = butter(order, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def bandpass_causal(x, fs, band, order=4):
    """Forward-only Butterworth, state initialised from the first sample."""
    sos = butter(order, band, btype="band", fs=fs, output="sos")
    zi = sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    y, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return y


def hold_resample(x, fs_in, fs_out):
    """Zero-order-hold resampling (causal)."""
    if fs_in == fs_out:
        return x
    n_out = int(np.floor(x.shape[-1] * fs_out / fs_in))
    idx = np.floor(np.arange(n_out) * fs_in / fs_out + 1e-9).astype(int)
    return x[..., np.minimum(idx, x.shape[-1] - 1)]
