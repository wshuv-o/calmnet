"""The dwell prior on the best EEG-only decoder, at the best window length.

Three results from this session were established separately and never combined:

  * the bare CNN is the strongest EEG-only decoder in the project (0.811),
    once the module ablation is run with an honest metric and the IMU-shortcut
    modules excluded;
  * 4 s windows beat 2 s on both accuracy and false-onset rate;
  * a label-space dwell prior cuts spurious activations ~3x.

The prior was only ever applied to tangent+logistic-regression (baseline 0.778),
because that was the pipeline under test when it was built. But the prior
operates on emitted posteriors and knows nothing about where they came from, so
it composes with any decoder. Running it on the better decoder, at the better
window, is one evaluation pass and is the last untried combination.

The bare model takes no motion at inference -- verified in exp_cancel_control,
where perturbing its motion reference moved accuracy by 0.000 -- so the motion
series enters here only as an array the training loop needs for shape and as
the target for the leakage probes. No motion reaches the decision.

Writes results/cnn_temporal.json.
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
from exp_temporal import (streams, transition_matrix, transition_from_tau,
                          forward_filter, viterbi, onset_metrics, calibrate_tau,
                          parse_arm)
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "cnn_temporal.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3
ARMS = ["none", "forward", "viterbi", "forward@auto"]


def load(sub, win):
    es = build_epochs(subject=sub, win=win, step=0.5)
    M, _ = build_motion_ts(sub, win=win, step=0.5)
    if len(M) != len(es):
        raise RuntimeError("motion misaligned: %d vs %d" % (len(M), len(es)))
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, ci = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    f = lambda a: a[tr][ti]
    c = lambda a: a[tr][ci]
    return {
        "Xf": f(es.X), "Mf": f(M), "yf": f(es.y),
        "sf": f(es.session), "tf": f(es.task), "gf": f(es.segment),
        "Xv": c(es.X), "Mv": c(M), "yv": c(es.y),
        "sv": c(es.session), "tv": c(es.task), "gv": c(es.segment),
        "Xt": es.X[~tr], "Mt": M[~tr], "yt": es.y[~tr],
        "st": es.session[~tr], "tt": es.task[~tr], "gt": es.segment[~tr],
        "imu_t": es.imu_feats[~tr], "vt": valid[~tr],
    }


def run_subject(d, arms):
    model, _ = train_arch(dict(BARE), d["Xf"], d["Mf"], d["yf"],
                          d["Xv"], d["Mv"], d["yv"], seed=0)
    lgv, _ = predict_arch(model, d["Xv"], d["Mv"])
    T = float(np.clip(fit_temperature(lgv, d["yv"]), 0.5, 5.0))
    lg, _ = predict_arch(model, d["Xt"], d["Mt"])
    P = softmax_np(lg, T)
    if not np.isfinite(P).all():
        P = np.full_like(P, 0.5)
    Pc = softmax_np(lgv, T)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sf = streams(d["sf"], d["tf"], d["gf"])
    sc = streams(d["sv"], d["tv"], d["gv"])
    st = streams(d["st"], d["tt"], d["gt"])
    pi = np.array([(d["yf"] == c).mean() for c in (0, 1)], float)

    out = {}
    for arm in arms:
        base, tau = parse_arm(arm)
        decode = {"forward": lambda p, A, q: forward_filter(p, A, q).argmax(1),
                  "viterbi": viterbi}.get(base)
        if tau == "auto":
            tau = calibrate_tau(Pc, d["yv"], sc, pi, decode)
        A = transition_matrix(d["yf"], sf) if tau is None else transition_from_tau(tau)
        if base == "none":
            pred = P.argmax(1)
        else:
            pred = np.zeros(len(P), int)
            for idx in st:
                pred[idx] = decode(P[idx], A, pi)
        vt = d["vt"]
        r2c = (FE.invariance_r2_conditional(pred[vt, None].astype(float),
                                            d["imu_t"][vt], d["yt"][vt], d["st"][vt])
               if vt.sum() > 40 else float("nan"))
        out[arm] = {"acc": balanced_accuracy(d["yt"], pred),
                    "r2_cond_dec": r2c,
                    "tau": float(A[1, 1]),
                    **onset_metrics(d["yt"], pred, st)}
    return out


def main():
    wins = [float(w) for w in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2.0, 4.0]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for win in wins:
        per = {}
        for sub in SUBJECTS:
            try:
                d = load(sub, win)
            except Exception as e:
                print("  [skip] %s: %s: %s" % (sub, type(e).__name__, e), flush=True)
                continue
            t0 = time.time()
            per[sub] = run_subject(d, ARMS)
            print("  [%s] win=%s %s (%.0fs)"
                  % (sub, win, " ".join("%s=%.3f" % (a, per[sub][a]["acc"]) for a in ARMS),
                     time.time() - t0), flush=True)
        if not per:
            continue
        print("\n######## bare CNN + dwell prior, win=%ss, %d subjects ########"
              % (win, len(per)), flush=True)
        for arm in ARMS:
            g = lambda k: float(np.nanmean([per[s][arm][k] for s in per]))
            rec = {"win": win, "arm": arm, "acc": g("acc"),
                   "acc_std": float(np.std([per[s][arm]["acc"] for s in per])),
                   "r2_cond_dec": g("r2_cond_dec"),
                   "false_onsets_per_min": g("false_onsets_per_min"),
                   "latency_s": g("latency_s"), "missed_onsets": g("missed_onsets"),
                   "tau": g("tau"),
                   "per_subject": {s: per[s][arm]["acc"] for s in per}}
            out["w%s|%s" % (win, arm)] = rec
            print("  %-13s acc %.3f+-%.3f  cond_dec %+.3f  onsets/min %.2f  "
                  "lat %.1fs  missed %.2f  tau %.3f"
                  % (arm, rec["acc"], rec["acc_std"], rec["r2_cond_dec"],
                     rec["false_onsets_per_min"], rec["latency_s"],
                     rec["missed_onsets"], rec["tau"]), flush=True)
            OUT.write_text(json.dumps(out, indent=1))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
