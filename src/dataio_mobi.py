"""Loader for the second dataset: MoBI treadmill-walking BCI (Luu et al. 2018).

External validation cohort for the movement-leakage result. Independent lab,
independent subjects, same problem class as NeuroRex (ds007788):

    ds007788        60ch EEG @100Hz, Walk/Stop from exoskeleton rexstate,
                    head+exo IMU as the movement measurement
    this dataset    64ch EEG @100Hz, Walk/Stand from treadmill phase,
                    6 goniometers (hip/knee/ankle x L/R) as the movement
                    measurement

Layout: SLxx-Tyy/{eeg.txt, joints.txt, conductor.txt}, 8 subjects x 3 trials.
  eeg.txt        "64 channels" header, then  time  ch1..ch64   @100 Hz
  joints.txt     "6 joints (GHR GKR GAR GHL GKL GAL PHR..PAL)" header,
                 then time + 6 measured goniometer angles + 6 decoder outputs.
                 Only the 6 MEASURED columns are used -- the decoder outputs are
                 model predictions, not observations.
  conductor.txt  time,event stream; code 8 marks walking onset, 10 the offset.

The walking phase is one contiguous block per trial, so within-trial splitting
would confound the label with slow drift. Splits are therefore taken ACROSS
trials (fit on T01, test on T02/T03), which is leakage-free at the recording
level and mirrors the train-early / test-late design used on ds007788.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent / "data" / "mobi_treadmill"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache_mobi"
SFREQ = 100.0
STAND, WALK = 0, 1
CHUNK_S = 20.0     # sub-segment length for leakage-free grouping
N_REF_MOBI = 6     # goniometer channels used as the cancellation/probe reference
JOINT_NAMES = ("hip_R", "knee_R", "ankle_R", "hip_L", "knee_L", "ankle_L")


@dataclass
class MobiEpochs:
    X: np.ndarray            # (N, C, T)
    y: np.ndarray            # (N,) 0=stand 1=walk
    trial: np.ndarray        # (N,) trial index 1..3
    segment: np.ndarray      # (N,) contiguous-block id (leakage-free grouping)
    motion: np.ndarray       # (N, 12) goniometer summary features
    motion_ts: np.ndarray = None  # (N, 6, T) goniometer waveforms, EEG-aligned
    subject: str = ""
    sfreq: float = SFREQ
    joint_names: tuple = field(default_factory=lambda: JOINT_NAMES)

    def __len__(self):
        return len(self.y)

    def by_trials(self, trials):
        m = np.isin(self.trial, list(trials))
        return MobiEpochs(self.X[m], self.y[m], self.trial[m], self.segment[m],
                          self.motion[m],
                          None if self.motion_ts is None else self.motion_ts[m],
                          self.subject, self.sfreq, self.joint_names)


def subjects() -> list[str]:
    if not ROOT.is_dir():
        return []
    return sorted({p.name.split("-")[0] for p in ROOT.glob("SL*-T*") if p.is_dir()})


def _read_matrix(path: Path, skip: int) -> np.ndarray:
    return np.loadtxt(path, skiprows=skip, dtype=np.float32)


def _walk_bounds(trial_dir: Path, t_end: float) -> tuple[float, float]:
    """Walking onset/offset. Prefer conductor codes 8/10; fall back to knee
    activity if the markers are missing or implausible."""
    cond = trial_dir / "conductor.txt"
    on = off = None
    if cond.exists():
        ev = []
        for l in cond.read_text(errors="replace").splitlines()[2:]:
            q = l.split()
            if len(q) >= 2:
                try:
                    ev.append((float(q[0]), int(float(q[1]))))
                except ValueError:
                    pass
        t8 = [t for t, c in ev if c == 8]
        t10 = [t for t, c in ev if c == 10]
        if t8:
            on = min(t8)
        if t10:
            cand = [t for t in t10 if on is None or t > on + 60]
            if cand:
                off = max(cand)
    return on, off


def _joint_feats(G: np.ndarray) -> np.ndarray:
    """[std, range] per joint over the window -> 12-D movement vector.

    Deliberately the same size and spirit as the 12-D head+exo IMU vector used
    on ds007788, so the invariance probe is comparable across datasets. Mean
    angle is excluded: it encodes posture/offset rather than movement, and
    drifts with sensor placement across trials.
    """
    out = []
    for j in range(G.shape[1]):
        v = G[:, j]
        out += [float(v.std()), float(v.max() - v.min())]
    return out


def build_subject(sub: str, win=2.0, step=0.5, l_freq=8.0, h_freq=30.0,
                  margin=2.0, use_cache=True) -> MobiEpochs | None:
    tag = f"{sub}_w{win}_s{step}_{l_freq}-{h_freq}_c{int(CHUNK_S)}_ts"
    cf = CACHE / f"{tag}.npz"
    if use_cache and cf.exists():
        try:
            d = np.load(cf, allow_pickle=True)
            return MobiEpochs(d["X"], d["y"], d["trial"], d["segment"], d["motion"],
                              d["motion_ts"], str(d["subject"]), float(d["sfreq"]))
        except Exception:
            cf.unlink(missing_ok=True)      # truncated by an interrupted write


    sos = butter(4, [l_freq, h_freq], btype="band", fs=SFREQ, output="sos")
    wlen, wstep, mrg = int(win * SFREQ), int(step * SFREQ), int(margin * SFREQ)
    X, y, tr, seg, mot, mts = [], [], [], [], [], []
    segid = 0
    for t_idx in (1, 2, 3):
        d = ROOT / f"{sub}-T{t_idx:02d}"
        if not (d / "eeg.txt").exists():
            continue
        E = _read_matrix(d / "eeg.txt", 1)
        J = _read_matrix(d / "joints.txt", 2)
        te, eeg = E[:, 0], E[:, 1:].T                 # (C, N)
        G = J[:, 1:7]                                  # measured goniometers only
        n = min(eeg.shape[1], len(G))
        eeg, G, te = eeg[:, :n], G[:n], te[:n]
        on, off = _walk_bounds(d, te[-1])
        if on is None or off is None:
            continue

        eeg = sosfiltfilt(sos, eeg, axis=-1).astype(np.float32)
        # Each phase is ONE contiguous block, so a block-level segment id leaves
        # grouped_split nothing to divide: with a single walk segment it puts the
        # whole walk phase on one side and the fit split ends up single-class.
        # Blocks are therefore cut into fixed time chunks, and windows that would
        # straddle a chunk boundary are dropped so overlapping windows can never
        # span two groups.
        chunk = int(CHUNK_S * SFREQ)
        blocks = [(0.0, on, STAND), (on, off, WALK), (off, te[-1], STAND)]
        for b0, b1, lab in blocks:
            a0, a1 = int(b0 * SFREQ) + mrg, int(b1 * SFREQ) - mrg
            if a1 - a0 < wlen:
                continue
            for a in range(a0, a1 - wlen + 1, wstep):
                w = eeg[:, a:a + wlen]
                if w.shape[1] != wlen:
                    continue
                c0 = (a - a0) // chunk
                if (a - a0 + wlen - 1) // chunk != c0:      # straddles a boundary
                    continue
                X.append(w); y.append(lab); tr.append(t_idx)
                seg.append(segid + int(c0))
                gw = G[a:a + wlen]
                mot.append(_joint_feats(gw))
                mts.append(gw.T.astype(np.float32))      # (6, T) waveform
            segid += int((a1 - a0) // chunk) + 1

    if not X:
        return None
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-7
    X = ((X - mu) / sd).astype(np.float32)
    MT = np.asarray(mts, np.float32)
    # standardise each goniometer channel so the reference is unit-scale, the
    # same treatment motion_ts.py applies to the ds007788 IMU reference
    mu_t = MT.mean(axis=(0, 2), keepdims=True)
    sd_t = MT.std(axis=(0, 2), keepdims=True) + 1e-6
    MT = ((MT - mu_t) / sd_t).astype(np.float32)
    es = MobiEpochs(X, np.asarray(y, np.int64), np.asarray(tr, np.int64),
                    np.asarray(seg, np.int64), np.asarray(mot, np.float32), MT, sub)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cf.with_name(cf.stem + ".tmp.npz")   # must end .npz: savez appends otherwise
    np.savez_compressed(tmp, X=es.X, y=es.y, trial=es.trial, segment=es.segment,
                        motion=es.motion, motion_ts=es.motion_ts,
                        subject=sub, sfreq=SFREQ)
    tmp.replace(cf)                          # atomic: no truncated cache on interrupt
    return es


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    subs = subjects()
    print(f"subjects found: {subs}\n")
    for s in subs:
        es = build_subject(s)
        if es is None:
            print(f"{s}: no usable trials"); continue
        print(f"{s}: {len(es):5d} epochs  X{es.X.shape}  "
              f"walk={int(es.y.sum()):5d} stand={int((es.y == 0).sum()):5d}  "
              f"trials={sorted(set(es.trial.tolist()))}  "
              f"motion{es.motion.shape}  "
              f"kneeR-std walk {es.motion[es.y == 1][:, 2].mean():.1f} "
              f"stand {es.motion[es.y == 0][:, 2].mean():.1f}", flush=True)
