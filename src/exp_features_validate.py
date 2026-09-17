"""Validate the representation result with an honest invariance probe.

Two corrections over the first pass.

1. THE PROBE. The original measure fitted a ridge on the training sessions and
   scored it on the test sessions, which conflates invariance with distribution
   shift: a representation whose features merely move between sessions scores
   strongly negative while still encoding movement perfectly well within any
   session. Measured that way FBCSP looked like the most invariant thing tested
   (-0.501); probed honestly it is +0.181, i.e. clearly leaky. Every R^2 here
   uses features.invariance_r2_cv -- cross-validated inside the evaluation set,
   grouped by session.

2. MEMORY. The previous run died with MemoryError because it rebuilt every
   subject's epochs for every seed and every cell (~158 MB each). Subjects are
   now loaded once, all cells for that subject are computed, and the arrays are
   released before moving on.

Writes results/features_validation.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import gc
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from abstain import balanced_accuracy
from calmnet_msa import imu_valid_mask
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "features_validation.json"
SEEDS = [0, 1, 2]
CELLS = ["raw|tangent_ea", "raw|tangent+plv", "car|tangent_ea",
         "raw|fbcsp", "raw|bandpower", "raw|plv"]


def build_features(feat, Xf, yf, Xt):
    if feat == "bandpower":
        return FE.log_var(FE.bandpass(Xf, 8, 30)), FE.log_var(FE.bandpass(Xt, 8, 30))
    if feat == "fbcsp":
        return FE.fbcsp(Xf, yf, Xt)
    if feat == "tangent_ea":
        return (FE.tangent(FE.euclidean_align(FE.covariances(Xf))),
                FE.tangent(FE.euclidean_align(FE.covariances(Xt))))
    if feat == "plv":
        return FE.plv_features(Xf), FE.plv_features(Xt)
    if feat == "tangent+plv":
        a = np.concatenate([FE.tangent(FE.euclidean_align(FE.covariances(Xf))),
                            FE.plv_features(Xf)], 1)
        b = np.concatenate([FE.tangent(FE.euclidean_align(FE.covariances(Xt))),
                            FE.plv_features(Xt)], 1)
        return a, b
    raise ValueError(feat)


def score_cell(prep, feat, Xf, yf, Xt, yt, Mt, vt, grp):
    Pf = FE.car(Xf) if prep == "car" else Xf
    Pt = FE.car(Xt) if prep == "car" else Xt
    Ff, Ft = build_features(feat, Pf, yf, Pt)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=0.1,
                                           class_weight="balanced"))
    clf.fit(Ff, yf)
    acc = balanced_accuracy(yt, clf.predict(Ft))
    r2 = FE.invariance_r2_cv(Ft[vt], Mt[vt], grp[vt]) if vt.sum() > 40 else float("nan")
    del Ff, Ft, Pf, Pt
    return acc, r2


def run_ds(out):
    subs = [f"sub-0{i}" for i in range(1, 8)]
    print("ds007788 -- honest probe (session-grouped CV inside the test set)\n",
          flush=True)
    acc = {(c, s): [] for c in CELLS for s in SEEDS}
    r2 = {(c, s): [] for c in CELLS for s in SEEDS}
    for sub in subs:                      # subject OUTER: load once, release after
        es = build_epochs(subject=sub)
        valid = imu_valid_mask(es.imu_feats, es.session)
        pres = sorted(set(int(v) for v in np.unique(es.session)))
        sess = [s for s in list_sessions(sub) if s in pres]
        tr = np.isin(es.session, sess[:3])
        Xtr_all, ytr_all, seg = es.X[tr], es.y[tr], es.segment[tr]
        Xt, yt = es.X[~tr], es.y[~tr]
        Mt, vt, grp = es.imu_feats[~tr], valid[~tr], es.session[~tr]
        for sd in SEEDS:
            ti, _ = grouped_split(seg, ytr_all, frac=0.3, seed=sd)
            for cell in CELLS:
                prep, feat = cell.split("|")
                try:
                    a, r = score_cell(prep, feat, Xtr_all[ti], ytr_all[ti],
                                      Xt, yt, Mt, vt, grp)
                    acc[(cell, sd)].append(a); r2[(cell, sd)].append(r)
                except Exception as e:
                    print(f"    {sub} {cell} seed{sd}: {type(e).__name__}", flush=True)
        del es, Xtr_all, ytr_all, Xt, yt, Mt, valid
        gc.collect()
        print(f"  {sub} done", flush=True)

    print()
    for cell in CELLS:
        A = [np.mean(acc[(cell, s)]) for s in SEEDS if acc[(cell, s)]]
        R = [np.nanmean(r2[(cell, s)]) for s in SEEDS if r2[(cell, s)]]
        if not A:
            continue
        out["ds|" + cell] = {"acc_mean": float(np.mean(A)), "acc_sd": float(np.std(A)),
                             "r2_mean": float(np.mean(R)), "r2_sd": float(np.std(R)),
                             "n_seeds": len(A)}
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  {cell:20} acc {np.mean(A):.3f} +/- {np.std(A):.3f}   "
              f"R2 {np.mean(R):+.3f} +/- {np.std(R):.3f}   (n={len(A)})", flush=True)
    return out


def run_mobi(out):
    from dataio_mobi import subjects as msubs, build_subject as mbuild
    print("\nMoBI cohort -- 8 independent subjects, goniometer reference\n", flush=True)
    acc = {c: [] for c in CELLS}
    r2 = {c: [] for c in CELLS}
    for sub in msubs():
        es = mbuild(sub)
        if es is None:
            continue
        fit, test = es.by_trials([1]), es.by_trials([2, 3])
        if len(np.unique(fit.y)) < 2 or len(np.unique(test.y)) < 2:
            continue
        ti, _ = grouped_split(fit.segment, fit.y, frac=0.3, seed=0)
        vt = np.ones(len(test.y), bool)
        for cell in CELLS:
            prep, feat = cell.split("|")
            try:
                a, r = score_cell(prep, feat, fit.X[ti], fit.y[ti], test.X, test.y,
                                  test.motion, vt, test.trial)
                acc[cell].append(a); r2[cell].append(r)
            except Exception as e:
                print(f"    {sub} {cell}: {type(e).__name__}: {str(e)[:40]}", flush=True)
        del es, fit, test
        gc.collect()
        print(f"  {sub} done", flush=True)

    print()
    for cell in CELLS:
        if not acc[cell]:
            continue
        out["mobi|" + cell] = {"acc": float(np.mean(acc[cell])),
                               "r2": float(np.nanmean(r2[cell])),
                               "n": len(acc[cell])}
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  {cell:20} acc {np.mean(acc[cell]):.3f}   "
              f"R2 {np.nanmean(r2[cell]):+.3f}   ({len(acc[cell])} subjects)", flush=True)
    return out


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out = {k: v for k, v in out.items() if "acc_mean" in v or "acc" in v}
    out = run_ds(out)
    out = run_mobi(out)

    print("\n" + "=" * 80)
    print("VERDICT -- honest probe throughout")
    print("=" * 80)
    print(f"{'representation':20}{'ds007788 acc':>16}{'R2':>9}  |{'mobi acc':>10}{'R2':>9}")
    for cell in CELLS:
        d, m = out.get("ds|" + cell), out.get("mobi|" + cell)
        if not d:
            continue
        line = f"{cell:20}{d['acc_mean']:9.3f}+/-{d['acc_sd']:.3f}{d['r2_mean']:+9.3f}  |"
        line += f"{m['acc']:10.3f}{m['r2']:+9.3f}" if m else f"{'--':>10}{'--':>9}"
        print(line)
    print("\nreference: deep composed arch  ds 0.695 +/- 0.024  |  mobi 0.588 +/- 0.007")
    print("note: R2 > 0 means movement IS recoverable -- the representation is NOT invariant")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
