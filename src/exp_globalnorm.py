"""Global normalisation instead of per-window, on the actual decoder.

Reading shonaka/EEG-neural-decoding (Contreras-Vidal lab, the group that
recorded the MoBI cohort) surfaced a preprocessing difference: they fit ONE
StandardScaler on the training set and apply it to test. `dataio.build_epochs`
instead z-scores every channel of every window to unit variance independently,
which removes absolute power and, because it is per channel, the spatial power
contrast that mu/beta ERD actually consists of.

Measured consequence so far (exp_zscore.py, logistic regression):

    log band-power   0.625 (per-window)  ->  0.760 (raw)   +0.134
    tangent + EA     0.762               ->  0.746         -0.016
    correlation only 0.738               ->  0.734         -0.004

So the damage is specific to variance-based features, exactly as predicted --
and the Riemannian pipeline barely moves because trace-normalised covariance was
already throwing amplitude away. The CNN, however, consumes the raw time series
directly and has never been tested on anything but per-window-normalised input.

Three schemes, same decoder, same splits, 3 seeds:

  perwindow   what the project has always done: each channel of each window
              forced to unit variance.
  global      the reference implementation's scheme: per-channel mean/std fitted
              on the FIT split only and applied unchanged to calibration and
              test. Preserves both absolute power and the spatial contrast,
              while still conditioning the input for the optimiser.
  none        raw volts. Included as a conditioning control -- if `global` beats
              `none`, the gain is normalisation quality, not merely the absence
              of per-window scaling.

Fitting the scaler on the fit split only matters: statistics taken over the
whole recording would leak test-session amplitude into training, which in a
longitudinal design is precisely the drift the model is supposed to survive.

Writes results/globalnorm.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from motion_ts import build_motion_ts
from splits import grouped_split
from calmnet_msa import imu_valid_mask
from exp_ablate import train_arch, predict_arch, BARE
from calibrate import fit_temperature, softmax_np
from abstain import balanced_accuracy
from exp_temporal import streams, onset_metrics
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "globalnorm.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3
SCHEMES = ["perwindow", "global", "none"]


def normalise(scheme, Xf, Xc, Xt):
    """Return (Xf, Xc, Xt) under one scheme. Statistics come from Xf only."""
    if scheme == "none":
        # still needs to be O(1) for the optimiser; a single global scalar
        # preserves every relative amplitude, unlike per-channel scaling
        s = float(Xf.std()) + 1e-30
        return Xf / s, Xc / s, Xt / s
    if scheme == "perwindow":
        f = lambda X: ((X - X.mean(axis=2, keepdims=True))
                       / (X.std(axis=2, keepdims=True) + 1e-7)).astype(np.float32)
        return f(Xf), f(Xc), f(Xt)
    if scheme == "global":
        mu = Xf.mean(axis=(0, 2), keepdims=True)         # per channel
        sd = Xf.std(axis=(0, 2), keepdims=True) + 1e-30
        f = lambda X: ((X - mu) / sd).astype(np.float32)
        return f(Xf), f(Xc), f(Xt)
    raise ValueError(scheme)


def load(sub, win, seed):
    # zscore=False: normalisation is applied here instead, per scheme
    es = build_epochs(subject=sub, win=win, step=0.5, zscore=False)
    M, _ = build_motion_ts(sub, win=win, step=0.5)
    if len(M) != len(es):
        raise RuntimeError("motion misaligned")
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, ci = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=seed)
    f = lambda a: a[tr][ti]
    c = lambda a: a[tr][ci]
    return {"Xf": f(es.X), "Mf": f(M), "yf": f(es.y),
            "Xc": c(es.X), "Mc": c(M), "yc": c(es.y),
            "Xt": es.X[~tr], "Mt": M[~tr], "yt": es.y[~tr],
            "st": es.session[~tr], "tt": es.task[~tr], "gt": es.segment[~tr],
            "imu_t": es.imu_feats[~tr], "vt": valid[~tr]}


def run(d, scheme):
    Xf, Xc, Xt = normalise(scheme, d["Xf"], d["Xc"], d["Xt"])
    model, _ = train_arch(dict(BARE), Xf, d["Mf"], d["yf"], Xc, d["Mc"], d["yc"],
                          seed=0)
    lgv, _ = predict_arch(model, Xc, d["Mc"])
    T = float(np.clip(fit_temperature(lgv, d["yc"]), 0.5, 5.0))
    lg, _ = predict_arch(model, Xt, d["Mt"])
    P = softmax_np(lg, T)
    if not np.isfinite(P).all():
        P = np.full_like(P, 0.5)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pred = P.argmax(1)
    vt = d["vt"]
    r2 = (FE.invariance_r2_conditional(pred[vt, None].astype(float),
                                       d["imu_t"][vt], d["yt"][vt], d["st"][vt])
          if vt.sum() > 40 else float("nan"))
    st = streams(d["st"], d["tt"], d["gt"])
    return (balanced_accuracy(d["yt"], pred), r2,
            onset_metrics(d["yt"], pred, st)["false_onsets_per_min"])


def main():
    wins = [float(w) for w in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4.0]
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for win in wins:
        for scheme in SCHEMES:
            per_seed = []
            for seed in seeds:
                A, R, O = [], [], []
                for sub in SUBJECTS:
                    try:
                        d = load(sub, win, seed)
                        a, r, o = run(d, scheme)
                    except Exception as e:
                        print("  [fail] %s/%s/s%d: %s: %s"
                              % (sub, scheme, seed, type(e).__name__, e), flush=True)
                        continue
                    A.append(a); R.append(r); O.append(o)
                if A:
                    per_seed.append((float(np.mean(A)), float(np.nanmean(R)),
                                     float(np.nanmean(O))))
                    print("    %-10s win=%s seed=%d  acc %.3f  cond_r2 %+.3f  onsets %.2f"
                          % (scheme, win, seed, *per_seed[-1]), flush=True)
            if not per_seed:
                continue
            acc = [p[0] for p in per_seed]
            rec = {"win": win, "scheme": scheme, "seeds": seeds,
                   "acc": float(np.mean(acc)), "acc_sd": float(np.std(acc)),
                   "acc_per_seed": acc,
                   "cond_r2": float(np.nanmean([p[1] for p in per_seed])),
                   "onsets": float(np.mean([p[2] for p in per_seed]))}
            out["w%s|%s" % (win, scheme)] = rec
            print("  %-10s win=%s  ACC %.3f+-%.3f  cond_r2 %+.3f  onsets %.2f"
                  % (scheme, win, rec["acc"], rec["acc_sd"], rec["cond_r2"],
                     rec["onsets"]), flush=True)
            OUT.write_text(json.dumps(out, indent=1))
        base = out.get("w%s|perwindow" % win, {}).get("acc")
        if base is not None:
            print("", flush=True)
            for scheme in SCHEMES:
                r = out.get("w%s|%s" % (win, scheme))
                if r:
                    print("  %-10s %.3f   delta vs per-window %+.3f"
                          % (scheme, r["acc"], r["acc"] - base), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
