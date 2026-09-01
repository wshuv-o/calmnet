"""Per-window motion TIME SERIES for the in-network canceller.

`dataio.EpochSet` carries a 12-D summary of the IMU per window, which is all the
adversarial/penalty methods need -- they only ever ask "how much movement was
there". Subtracting movement from the signal needs the waveform itself, aligned
sample-for-sample with the EEG window, so this module rebuilds it from the raw
BIDS motion files and caches it alongside the epochs.

Four reference channels per window, resampled to the EEG rate:
    head accel magnitude, head gyro magnitude, exo accel magnitude, exo gyro magnitude

Sessions whose motion file is absent yield an all-zero reference and are flagged
invalid, so the canceller can be told to leave those windows alone rather than
learning to subtract a constant (see calmnet_msa.imu_valid_mask for the same
issue in the summary features).
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np

from dataio import (DATA_ROOT, MI_TASKS, list_sessions, _imu_mag_series,
                    _rexstate_segments, STATE_TO_LABEL)

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache_motion"
N_REF = 4


def _resample(v: np.ndarray, n_out: int) -> np.ndarray:
    if len(v) == 0:
        return np.zeros(n_out, np.float32)
    if len(v) == n_out:
        return v.astype(np.float32)
    xi = np.linspace(0, len(v) - 1, n_out)
    return np.interp(xi, np.arange(len(v)), v).astype(np.float32)


def build_motion_ts(subject="sub-01", sessions=None, tasks=MI_TASKS,
                    win=2.0, step=0.5, margin=0.5, sfreq=100.0, use_cache=True):
    """Return (M, valid) with M (N, 4, T) aligned to build_epochs() window order.

    The window enumeration mirrors dataio._windows_from_task exactly -- same
    segment order, same margin, same stride, same clamping -- so row i here is
    the same window as row i of the EpochSet.
    """
    sessions = sessions or list_sessions(subject)
    tag = f"{subject}_s{'-'.join(map(str, sessions))}_w{win}_st{step}_ref{N_REF}"
    cf = CACHE / f"{tag}.npz"
    if use_cache and cf.exists():
        try:
            d = np.load(cf)
            return d["M"], d["valid"]
        except Exception:
            cf.unlink(missing_ok=True)

    wlen, wstep, mrg = round(win * sfreq), round(step * sfreq), round(margin * sfreq)
    out, val = [], []
    for ses in sessions:
        eeg_dir = DATA_ROOT / subject / f"ses-{ses:02d}" / "eeg"
        for task in tasks:
            edf = eeg_dir / f"{subject}_ses-{ses:02d}_task-{task}_eeg.edf"
            evt = eeg_dir / f"{subject}_ses-{ses:02d}_task-{task}_acq-rexstate_events.tsv"
            if not (edf.exists() and evt.exists()):
                continue
            seg = _rexstate_segments(evt)
            if not seg:
                continue
            import mne
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
            sf = round(raw.info["sfreq"])
            n_samp = raw.n_times
            head = _imu_mag_series(subject, ses, task, "head")
            exo = _imu_mag_series(subject, ses, task, "exo")
            ok = head is not None or exo is not None

            for onset, dur, lab in seg:
                s0 = int(onset * sf) + mrg
                s1 = min(int((onset + dur) * sf), n_samp)
                for a in range(s0, s1 - wlen + 1, wstep):
                    chans = []
                    for imu in (head, exo):
                        if imu is None:
                            chans += [np.zeros(wlen, np.float32)] * 2
                            continue
                        acc, gyr, msf = imu
                        ma, mb = int(a / sf * msf), int((a + wlen) / sf * msf)
                        chans.append(_resample(acc[ma:mb], wlen))
                        chans.append(_resample(gyr[ma:mb], wlen))
                    out.append(np.stack(chans))
                    val.append(ok)

    if not out:
        return np.zeros((0, N_REF, wlen), np.float32), np.zeros(0, bool)
    M = np.asarray(out, np.float32)
    valid = np.asarray(val, bool)
    # per-channel standardisation over valid windows: the canceller should see a
    # unit-scale reference, not raw accelerometer units
    if valid.any():
        mu = M[valid].mean(axis=(0, 2), keepdims=True)
        sd = M[valid].std(axis=(0, 2), keepdims=True) + 1e-6
        M = ((M - mu) / sd).astype(np.float32)
        M[~valid] = 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cf.with_name(cf.stem + ".tmp.npz")
    np.savez_compressed(tmp, M=M, valid=valid)
    tmp.replace(cf)
    return M, valid


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from dataio import build_epochs
    for i in range(1, 8):
        sub = f"sub-0{i}"
        es = build_epochs(subject=sub)
        M, valid = build_motion_ts(sub)
        ok = len(M) == len(es)
        print(f"{sub}: epochs {len(es):5d}  motion_ts {M.shape}  valid {int(valid.sum()):5d}"
              f"  aligned={ok}", flush=True)
        if not ok:
            print(f"   !! MISALIGNED: {len(M)} motion rows vs {len(es)} epochs")
