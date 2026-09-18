"""How far does a linear model on a fixed representation get, on cohort A?

Why this exists
---------------
The frozen-encoder pre-flight needed a control: a probe result means nothing
without knowing what the probe itself can reach. The control was logistic
regression on log band power, and on five participants it scored 0.905 where
dn_stem scores 0.911 on the same five. That was meant to bound the probe. It
also says something about the architecture, and the branch this work sits on is
called representation-beats-architecture, so it is worth measuring properly
rather than leaving as a side effect of another experiment.

This runs three fixed representations through one linear classifier:

  lin_bp       log band power, 4 bands x 60 channels = 240 features
  lin_tan      log-Euclidean tangent vectors after Euclidean Alignment, 1830
  lin_bp_tan   both, concatenated

against the same participants, the same session splits, the same three seeds
and the same balanced_accuracy as pipeline C, by importing pipeline C's own
loader. Nothing here is tuned per participant.

Two ways this is deliberately conservative, so a win cannot be an artefact of
giving the linear model more:

  - It trains on the fitting split ONLY. The networks additionally use the
    calibration split for early stopping and model selection, so they see more
    data, not less.
  - Temperature is fitted on the calibration split exactly as in pipeline C, so
    ECE is computed the same way.

One metric is NOT comparable and is named differently to prevent it being put
in the same column: pipeline C ranks its 90 %-coverage accuracy by the trained
selective gate, which a linear model does not have. Here the ranking is by
maximum posterior, reported as acc_at_90_conf.

    CX_SEEDS=0,1,2 python src/exp_linear_bandpower.py

Writes results/a_linear_bandpower.json in pipeline C's key shape, so
tools/collect_numbers.py picks it up.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abstain import balanced_accuracy                        # noqa: E402
from calibrate import expected_calibration_error, softmax_np, fit_temperature
from exp_calmnetx import load                                 # noqa: E402
import features as FE                                         # noqa: E402

SEEDS = [int(s) for s in os.environ.get("CX_SEEDS", "0,1,2").split(",")]
SUBS = [s for s in os.environ.get("CX_SUBS", "").split(",") if s] or \
       ["sub-0%d" % i for i in range(1, 8)]
ARMS = [a for a in os.environ.get("CX_ARMS", "").split(",") if a] or \
       ["lin_bp", "lin_tan", "lin_bp_tan"]
BANDS = ((4, 8), (8, 13), (13, 30), (30, 45))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   os.environ.get("CX_OUT", "a_linear_bandpower.json"))


def band_power(X, sfreq=100.0, bands=BANDS):
    """Log band power per channel. What the network's stem is built to see."""
    out = []
    for lo, hi in bands:
        b, a = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band")
        Xf = filtfilt(b, a, X.astype(np.float64), axis=-1)
        out.append(np.log(Xf.var(axis=-1) + 1e-12))
    return np.concatenate(out, axis=1)


def tangent(X):
    """Covariance -> Euclidean Alignment -> log-Euclidean tangent vector.

    Alignment is fitted per set, so no statistic crosses from the test sessions
    back into the fitting split (the same convention as exp_temporal._tangent).
    """
    return FE.tangent(FE.euclidean_align(FE.covariances(X)))


def featurise(d, arm):
    """Cache each representation once per participant; arms share them."""
    if "_bp" not in d:
        d["_bp"] = (band_power(d["Xf"]), band_power(d["Xc"]),
                    band_power(d["Xt"]))
    if arm in ("lin_tan", "lin_bp_tan") and "_tan" not in d:
        d["_tan"] = (tangent(d["Xf"]), tangent(d["Xc"]), tangent(d["Xt"]))
    if arm == "lin_bp":
        return d["_bp"]
    if arm == "lin_tan":
        return d["_tan"]
    return tuple(np.concatenate([a, b], axis=1)
                 for a, b in zip(d["_bp"], d["_tan"]))


def run_subject(d, arm, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Ff, Fc, Ft = featurise(d, arm)
    sc = StandardScaler().fit(Ff)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=seed)
    clf.fit(sc.transform(Ff), d["yf"])

    def logits(F):
        # decision_function is the log-odds of class 1, so [0, z] softmaxes
        # back to exactly the model's own sigmoid.
        z = clf.decision_function(sc.transform(F))
        return np.stack([np.zeros_like(z), z], axis=1)

    T = float(np.clip(fit_temperature(logits(Fc), d["yc"]), 0.5, 5.0))
    p = softmax_np(logits(Ft), T)
    pred = p.argmax(1)
    conf = p.max(1)
    k = max(1, int(0.9 * len(conf)))
    sel = np.argsort(-conf)[:k]
    return {"acc": float(balanced_accuracy(d["yt"], pred)),
            "ece": float(expected_calibration_error(p, d["yt"])),
            "acc_at_90_conf": float(balanced_accuracy(d["yt"][sel], pred[sel])),
            "n_feat": int(Ff.shape[1])}


def main():
    out = json.load(io.open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    print("cohort A, %d participants, seeds %s, arms %s"
          % (len(SUBS), SEEDS, ARMS), flush=True)
    out.setdefault("_note", (
        "Linear models on fixed representations, cohort A, pipeline C splits "
        "and metric code. Trained on the FITTING split only, while the "
        "networks also use the calibration split for early stopping and model "
        "selection, so the linear models see less data rather than more. "
        "acc_at_90_conf ranks by maximum posterior, NOT by the trained "
        "selective gate pipeline C uses, so it must not be placed in the same "
        "column as acc_at_90."))
    for seed in SEEDS:
        D = {}
        for sub in SUBS:
            try:
                D[sub] = load(sub, seed)
            except Exception as e:
                print("  [skip] %s: %s" % (sub, e), flush=True)
        for arm in ARMS:
            key = "%s|s%d" % (arm, seed)
            if key in out:
                print("  %s already present" % key, flush=True)
                continue
            t0, rows = time.time(), {}
            for sub, d in D.items():
                ts = time.time()
                rows[sub] = run_subject(d, arm, seed)
                print("    %s/%-11s seed=%d  acc %.3f  (%.0fs)"
                      % (sub, arm, seed, rows[sub]["acc"], time.time() - ts),
                      flush=True)
            acc = [r["acc"] for r in rows.values()]
            agg = {k: float(np.mean([r[k] for r in rows.values()]))
                   for k in ("acc", "ece", "acc_at_90_conf", "n_feat")}
            agg["acc_sd"] = float(np.std(acc, ddof=1)) if len(acc) > 1 else None
            agg["per_subject"] = rows
            out[key] = agg
            print("  %-12s seed=%d  acc %.4f +- %.3f  ece %.3f  (%d feat, %.0fs)"
                  % (arm, seed, agg["acc"], np.std(acc), agg["ece"],
                     agg["n_feat"], time.time() - t0), flush=True)
            io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    print("\nwrote %s" % os.path.relpath(OUT), flush=True)


if __name__ == "__main__":
    main()
