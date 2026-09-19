"""The classical Riemannian pipeline, as a baseline for the branch.

The paper argues that second-order covariance structure complements a
log-power stem. A Riemannian-BCI reader will ask the obvious question: if
tangent-space features are the point, why not just use the standard Riemannian
classifier and skip the network?

Table 3 answers that against neural decoders only, so this runs the classical
pipelines in the same harness:

  riem_ts    Covariances -> TangentSpace(riemann) -> scaler -> logistic
             regression. The reference is the Riemannian mean of the FITTING
             covariances, fitted once and then fixed, which is what the
             classical method does and what our branch replaces with a running
             estimate.
  riem_mdm   Covariances -> MDM(riemann). Nearest class mean on the manifold,
             with no feature learning at all.

exp_linear_bandpower.py already ran a linear model on log-Euclidean tangent
vectors after Euclidean Alignment. That is not the same comparison: the
standard pipeline uses the affine-invariant mean and the AIRM tangent map,
which is the geometry our branch uses, so this is the sharper test.

Same loader, same session splits, same seeds and the same balanced_accuracy as
the network arms, by importing the harness's own loader. The classical models
train on the fitting split only, while the networks additionally use the
calibration split for early stopping, so the classical models see less data
rather than more.

    CX_COHORT=ds007788 CX_SEEDS=0,1,2 python src/exp_riemann_baseline.py
    CX_COHORT=eegbci   CX_SEEDS=0,1,2 CX_OUT=c_riemann.json python src/...

Writes results/<CX_OUT> in the harness key shape, so the table builders and
tools/collect_numbers.py pick it up.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from abstain import balanced_accuracy                         # noqa: E402
from calibrate import expected_calibration_error, fit_temperature, softmax_np

SEEDS = [int(s) for s in os.environ.get("CX_SEEDS", "0,1,2").split(",")]
ARMS = [a for a in os.environ.get("CX_ARMS", "").split(",") if a] or \
       ["riem_ts", "riem_mdm"]
COHORT = os.environ.get("CX_COHORT", "ds007788")
OUT = os.path.join(HERE, "..", "results",
                   os.environ.get("CX_OUT", "a_riemann.json"))

NOTE = ("Classical Riemannian pipelines in the harness of exp_calmnetx: same "
        "loader, same session splits, same seeds and the same "
        "balanced_accuracy. Trained on the FITTING split only, while the "
        "network arms also use the calibration split for early stopping and "
        "model selection, so these models see less data rather than more. "
        "riem_ts fits the Riemannian mean on the fitting covariances and then "
        "holds it fixed, which is the classical static reference our branch "
        "replaces with a running estimate. Covariances use the "
        "Ledoit-Wolf estimator.")


def subjects_and_loader():
    """Exactly the dispatch exp_calmnetx.main uses, so the splits match."""
    import exp_calmnetx as X
    if COHORT.startswith("mobi"):
        from dataio_mobi import subjects as s
        return s(), X.load_mobi
    if COHORT == "eegbci":
        from dataio_eegbci import subjects as s
        return s(), X.load_eegbci
    if COHORT == "bnci":
        from dataio_bnci import subjects as s
        return s(), X.load_bnci
    if COHORT == "decoded":
        from dataio_decoded import subjects as s
        return s(), X.load_decoded
    return X.SUBJECTS, X.load


def fit_predict(arm, Xf, yf, Xc, Xt):
    """Return test logits and calibration-split logits."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cov = Covariances(estimator="lwf")
    Cf, Cc, Ct = cov.transform(Xf), cov.transform(Xc), cov.transform(Xt)
    if arm == "riem_mdm":
        clf = MDM(metric="riemann")
        clf.fit(Cf, yf)
        # MDM scores are negative distances to each class mean
        return -clf.transform(Ct), -clf.transform(Cc)
    pipe = make_pipeline(TangentSpace(metric="riemann"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(Cf, yf)
    return (pipe.decision_function(Ct), pipe.decision_function(Cc))


def as_logits(z):
    """MDM gives two columns, logistic regression one; make both 2-column."""
    z = np.asarray(z, float)
    return z if z.ndim == 2 else np.stack([-z, z], axis=1)


def _one(args):
    """One (subject, arm, seed) cell, for the worker pool.

    Each worker is pinned to a small thread count: the Karcher mean is the
    expensive step and its BLAS calls would otherwise oversubscribe the CPUs
    across workers, which is slower than running sequentially.
    """
    sub, arm, seed = args
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = os.environ.get("CX_THREADS", "2")
    import warnings as _w
    _w.filterwarnings("ignore")
    _, loader = subjects_and_loader()
    try:
        d = loader(sub, seed)
        lt, lc = fit_predict(arm, d["Xf"], d["yf"], d["Xc"], d["Xt"])
        lt, lc = as_logits(lt), as_logits(lc)
        T = fit_temperature(lc, d["yc"])
        p = softmax_np(lt / max(T, 1e-3))
        return sub, arm, seed, {
            "acc": float(balanced_accuracy(d["yt"], lt.argmax(1))),
            "ece": float(expected_calibration_error(p, d["yt"])),
        }, None
    except Exception as e:
        return sub, arm, seed, None, "%s: %s" % (type(e).__name__, str(e)[:90])


def main_parallel(n_workers):
    from multiprocessing import Pool
    subs, _ = subjects_and_loader()
    jobs = [(s, a, sd) for sd in SEEDS for a in ARMS for s in subs]
    out = json.load(io.open(OUT, encoding="utf-8")) if os.path.exists(OUT) \
        else {}
    cells, t0 = {}, time.time()
    print("  %d cells on %d workers" % (len(jobs), n_workers), flush=True)
    with Pool(n_workers) as pool:
        for sub, arm, seed, rec, err in pool.imap_unordered(_one, jobs):
            if err:
                print("    [skip] %s/%s s%d: %s" % (sub, arm, seed, err),
                      flush=True)
                continue
            cells.setdefault("%s|s%d" % (arm, seed), {})[sub] = rec
            print("    %s/%s seed=%d  acc %.3f"
                  % (sub, arm, seed, rec["acc"]), flush=True)
    for key, per in sorted(cells.items()):
        accs = [v["acc"] for v in per.values()]
        out[key] = {"acc": float(np.mean(accs)),
                    "acc_sd": float(np.std(accs)),
                    "ece": float(np.mean([v["ece"] for v in per.values()])),
                    "per_subject": per}
        print("  %s  acc %.4f +- %.4f  (%d subjects)"
              % (key, out[key]["acc"], out[key]["acc_sd"], len(per)),
              flush=True)
    out["_note"] = NOTE
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    print("wrote %s in %.0fs" % (os.path.relpath(OUT, HERE), time.time() - t0))


def main():
    n = int(os.environ.get("CX_WORKERS", "1"))
    if n > 1:
        return main_parallel(n)
    subs, loader = subjects_and_loader()
    out = json.load(io.open(OUT, encoding="utf-8")) if os.path.exists(OUT) \
        else {}
    out["_note"] = NOTE
    for seed in SEEDS:
        for arm in ARMS:
            key = "%s|s%d" % (arm, seed)
            per, t0 = {}, time.time()
            for sub in subs:
                try:
                    d = loader(sub, seed)
                    lt, lc = fit_predict(arm, d["Xf"], d["yf"], d["Xc"],
                                         d["Xt"])
                    lt, lc = as_logits(lt), as_logits(lc)
                    pred = lt.argmax(1)
                    T = fit_temperature(lc, d["yc"])
                    p = softmax_np(lt / max(T, 1e-3))
                    per[sub] = {
                        "acc": float(balanced_accuracy(d["yt"], pred)),
                        "ece": float(expected_calibration_error(p, d["yt"])),
                    }
                    print("    %s/%s seed=%d  acc %.3f"
                          % (sub, arm, seed, per[sub]["acc"]), flush=True)
                except Exception as e:
                    print("    [skip] %s/%s: %s: %s"
                          % (sub, arm, type(e).__name__, str(e)[:90]),
                          flush=True)
            if not per:
                continue
            accs = [v["acc"] for v in per.values()]
            out[key] = {"acc": float(np.mean(accs)),
                        "acc_sd": float(np.std(accs)),
                        "ece": float(np.mean([v["ece"] for v in per.values()])),
                        "per_subject": per}
            print("  %s  acc %.4f +- %.4f  (%d subjects, %.0fs)"
                  % (key, out[key]["acc"], out[key]["acc_sd"], len(per),
                     time.time() - t0), flush=True)
            io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    print("wrote %s" % os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
