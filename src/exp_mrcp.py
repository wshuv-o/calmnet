"""Is there a movement-related cortical potential in this dataset?

Every experiment in this project has band-passed 8-30 Hz, which removes the MRCP
(readiness potential) entirely. The MRCP is a slow negative deflection over
sensorimotor cortex beginning ~1.5-2 s BEFORE movement onset, at 0.1-3 Hz. It is
a different generator from mu/beta ERD, it is PRE-movement, and it underpins a
real lower-limb intent-detection literature. For an exoskeleton that is the
attractive property: intent before motion, rather than detection after it.

Before building a pathway for it, establish that it exists here.

WHY A NAIVE TEST FAILS, AND DID

Averaging steady-state Walk epochs against steady-state Stop epochs at 0.1-3 Hz
produced a slow difference of ~2e-06 that was IDENTICAL over central and
occipital electrodes. That is a global drift -- movement artefact or common-mode
shift -- not a focal cortical potential, and it would have been read as an MRCP
by anyone not checking topography. The MRCP is ONSET-LOCKED, so averaging over
sustained walking washes it out whether or not it is present.

THE CORRECT TEST

  * epochs time-locked to each Stop->Walk transition, -2.5 s to +1.0 s
  * baseline-corrected on [-2.5, -2.0] s, before any anticipatory activity
  * averaged across transitions within subject, then across subjects
  * compared between CENTRAL (C3/Cz/C4/C1/C2/FCz/CPz) and OCCIPITAL (O1/Oz/O2/POz)

The signature being tested for is specific and falsifiable: a NEGATIVE-going
deflection in the ~1 s before onset, LARGER over central than occipital. A
deflection of equal size at both is drift and counts as a negative result.

Two controls:

  surrogate   the same extraction at randomly chosen times inside Stop periods.
              Any "MRCP" that also appears at random times is drift.
  ratio       central-minus-occipital amplitude, which cancels anything spatially
              uniform -- the quantity a global artefact cannot produce.

Writes results/mrcp.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import DATA_ROOT, _load_raw, _rexstate_segments, list_sessions

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "mrcp.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
PRE, POST = 2.5, 1.0            # seconds around onset
BASE = (-2.5, -2.0)             # baseline window, before anticipation
CENTRAL = ("C3", "CZ", "C4", "C1", "C2", "FCZ", "CPZ")
OCCIP = ("O1", "OZ", "O2", "POZ")


def onsets_from_segments(seg):
    """Stop->Walk transition times: a Walk segment whose predecessor is Stop."""
    out = []
    for i in range(1, len(seg)):
        if seg[i][2] == 1 and seg[i - 1][2] == 0:
            out.append(seg[i][0])
    return out


def extract(raw, times, sfreq, pre=PRE, post=POST):
    data = raw.get_data()
    n_pre, n_post = int(pre * sfreq), int(post * sfreq)
    ep = []
    for t in times:
        a = int(t * sfreq) - n_pre
        b = a + n_pre + n_post
        if a < 0 or b > data.shape[1]:
            continue
        ep.append(data[:, a:b])
    return np.asarray(ep) if ep else None


def run_subject(sub, rng):
    sess = list_sessions(sub)
    real, surro, ch_names, sfreq = [], [], None, 100.0
    for ses in sess[:5]:
        d = DATA_ROOT / sub / f"ses-{ses:02d}" / "eeg"
        edf = d / f"{sub}_ses-{ses:02d}_task-training_eeg.edf"
        evt = d / f"{sub}_ses-{ses:02d}_task-training_acq-rexstate_events.tsv"
        if not (edf.exists() and evt.exists()):
            continue
        seg = _rexstate_segments(evt)
        if not seg:
            continue
        raw = _load_raw(edf, 0.1, 3.0)          # MRCP band, EOG regressed
        ch_names, sfreq = raw.ch_names, raw.info["sfreq"]
        ons = onsets_from_segments(seg)
        e = extract(raw, ons, sfreq)
        if e is not None:
            real.append(e)
        # surrogate: random times inside Stop periods, same count
        stops = [(o, o + dur) for o, dur, lab in seg if lab == 0 and dur > PRE + POST]
        if stops and ons:
            rt = []
            for _ in range(len(ons)):
                a, b = stops[rng.integers(len(stops))]
                rt.append(float(rng.uniform(a + PRE, b - POST)))
            e2 = extract(raw, rt, sfreq)
            if e2 is not None:
                surro.append(e2)
    if not real:
        return None
    R = np.concatenate(real)
    S = np.concatenate(surro) if surro else None
    return R, S, ch_names, sfreq


def summarise(E, ch_names, sfreq):
    """Baseline-correct, then mean waveform over central and occipital picks."""
    up = [c.upper() for c in ch_names]
    ci = [i for i, c in enumerate(up) if c in CENTRAL]
    oi = [i for i, c in enumerate(up) if c in OCCIP]
    n_pre = int(PRE * sfreq)
    b0, b1 = int((BASE[0] + PRE) * sfreq), int((BASE[1] + PRE) * sfreq)
    E = E - E[:, :, b0:b1].mean(axis=2, keepdims=True)
    return (E[:, ci].mean(axis=(0, 1)) if ci else None,
            E[:, oi].mean(axis=(0, 1)) if oi else None, n_pre)


def main():
    rng = np.random.default_rng(0)
    out, C, O, CS, OS = {}, [], [], [], []
    for sub in SUBJECTS:
        try:
            r = run_subject(sub, rng)
        except Exception as e:
            print("  [skip] %s: %s: %s" % (sub, type(e).__name__, e), flush=True)
            continue
        if r is None:
            continue
        R, S, ch, sf = r
        c, o, n_pre = summarise(R, ch, sf)
        if c is None or o is None:
            continue
        C.append(c); O.append(o)
        if S is not None:
            cs, os_, _ = summarise(S, ch, sf)
            CS.append(cs); OS.append(os_)
        # amplitude in the 1 s before onset, relative to baseline
        pre_c = float(c[n_pre - int(sf):n_pre].mean())
        pre_o = float(o[n_pre - int(sf):n_pre].mean())
        out[sub] = {"n_onsets": int(len(R)), "pre_central": pre_c,
                    "pre_occipital": pre_o, "central_minus_occip": pre_c - pre_o}
        print("  %s  onsets=%3d  pre-onset central %+.3e  occipital %+.3e  "
              "diff %+.3e" % (sub, len(R), pre_c, pre_o, pre_c - pre_o), flush=True)

    if not C:
        print("no data"); return
    C, O = np.mean(C, 0), np.mean(O, 0)
    n_pre = int(PRE * 100.0)
    pc, po = C[n_pre - 100:n_pre].mean(), O[n_pre - 100:n_pre].mean()
    print("\n" + "=" * 66)
    print("GRAND AVERAGE, 1 s before Stop->Walk onset (baseline -2.5..-2.0 s)")
    print("  central    %+.4e V" % pc)
    print("  occipital  %+.4e V" % po)
    print("  central - occipital  %+.4e V   <- an artefact cannot produce this" % (pc - po))
    if CS:
        CSm, OSm = np.mean(CS, 0), np.mean(OS, 0)
        sc, so = CSm[n_pre - 100:n_pre].mean(), OSm[n_pre - 100:n_pre].mean()
        print("  SURROGATE (random times in Stop): central %+.4e  occip %+.4e  "
              "diff %+.4e" % (sc, so, sc - so))
        out["_surrogate"] = {"central": float(sc), "occipital": float(so),
                             "diff": float(sc - so)}
    print("-" * 66)
    neg = pc < 0
    focal = abs(pc) > 1.5 * abs(po)
    print("negative-going over central : %s" % neg)
    print("focal (central > 1.5x occip): %s" % focal)
    print("VERDICT: %s" % ("MRCP-like signature present" if (neg and focal)
                           else "NO MRCP signature -- slow activity is not focal"))
    out["_grand"] = {"central": float(pc), "occipital": float(po),
                     "diff": float(pc - po), "negative": bool(neg),
                     "focal": bool(focal)}
    out["_waveforms"] = {"central": C.tolist(), "occipital": O.tolist(),
                         "sfreq": 100.0, "pre_s": PRE}
    OUT.write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % OUT.name)


if __name__ == "__main__":
    main()
