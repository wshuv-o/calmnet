"""How much session drift does the adaptive alignment layer actually remove?

DriftNet's central claim is that a label-free alignment layer, updating its
covariance estimate at inference, absorbs the session-to-session shift that
makes longitudinal EEG decoding hard. That claim was first measured on a single
subject (60% reduction) and a single subject is not evidence.

This measures it on every subject in both cohorts, and reports the quantity the
claim is actually about: the Riemannian distance on the SPD manifold between the
fit-session mean covariance and each held-out session's, before and after the
layer. Riemannian rather than Frobenius because covariance matrices are not a
vector space -- Frobenius distance is dominated by overall scale and would
flatter any whitening operation.

Three things make this a fair test rather than a demonstration:

  * The layer is primed on FIT data only, then adapts on each held-out session
    using nothing but the covariance of incoming windows. No labels, no test
    statistics leak backwards into the fit.
  * Updates are BATCHED, as they are during real inference. A single
    whole-session update moves the running estimate by one momentum step and
    removes almost nothing (4.5%) -- which is how this was first mis-measured.
  * A no-op control (momentum 0) is included. It shares every code path and
    simply never updates, so any reduction it shows is an artefact of the
    measurement rather than of adaptation.

Writes results/drift.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json, sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from driftnet import AdaptiveAlignment

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / os.environ.get("DRIFT_OUT", "drift.json")
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, BATCH = 3, 32


def cov(a, trace_norm=False):
    """Mean spatial covariance. NOT trace-normalised by default.

    The Riemannian distance used here is affine-invariant, so dividing by the
    mean eigenvalue is redundant for a raw comparison. It is not harmless after
    a whitening transform: whitening rescales two sessions' traces by different
    factors, and dividing each by its own trace reintroduces a relative scale
    that the distance IS sensitive to.

    Measured consequence. With trace normalisation the momentum-0 control moved
    the distance by up to +122 % on cohort A, so it was not the null control the
    caption claimed. Without it the control is exactly null (6.925 -> 6.925 on
    sub-01), which is what affine invariance requires, and the reported drift
    reduction falls by roughly 5 points because part of it was this artefact.
    """
    ac = a - a.mean(-1, keepdims=True)
    c = np.einsum("nct,ndt->cd", ac, ac) / (len(a) * a.shape[-1])
    if trace_norm:
        return c / (np.trace(c) / c.shape[0] + 1e-12)
    return c


def riemannian(A, B):
    """Affine-invariant distance: sqrt(sum log^2 eig(A^-1/2 B A^-1/2))."""
    n = len(A)
    A = A + 1e-6 * np.eye(n)
    B = B + 1e-6 * np.eye(n)
    Ai = la.fractional_matrix_power(A, -0.5).real
    w = np.clip(np.linalg.eigvalsh(Ai @ B @ Ai), 1e-10, None)
    return float(np.sqrt((np.log(w) ** 2).sum()))


def apply_layer(lay, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            out.append(lay(torch.as_tensor(X[i:i + BATCH])).numpy())
    return np.concatenate(out)


def run_subject(X, groups, fit_groups, momentum):
    lay = AdaptiveAlignment(X.shape[1], momentum=momentum).eval()
    fit = np.isin(groups, fit_groups)
    Afit = cov(apply_layer(lay, X[fit]))          # prime on fit data only
    d = []
    for g in sorted(set(int(v) for v in np.unique(groups)) - set(fit_groups)):
        d.append(riemannian(Afit, cov(apply_layer(lay, X[groups == g]))))
    return d


def cohort_a():
    rows = {}
    for sub in SUBJECTS:
        try:
            es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        X = es.X / (es.X.std() + 1e-30)
        g = es.session.astype(int)
        pres = sorted(set(int(v) for v in np.unique(g)))
        sess = [s for s in list_sessions(sub) if s in pres]
        fitg = sess[:N_TRAIN]
        Cfit = cov(X[np.isin(g, fitg)])
        raw = [riemannian(Cfit, cov(X[g == s])) for s in sess[N_TRAIN:]]
        adapt = run_subject(X, g, fitg, 0.2)
        noop = run_subject(X, g, fitg, 0.0)
        rows[sub] = {"raw": float(np.mean(raw)), "aligned": float(np.mean(adapt)),
                     "noop": float(np.mean(noop)),
                     "reduction": float(1 - np.mean(adapt) / max(np.mean(raw), 1e-12)),
                     "n_test_sessions": len(raw)}
        print("  %s  raw %.3f -> aligned %.3f (noop %.3f)  reduction %5.1f%%"
              % (sub, rows[sub]["raw"], rows[sub]["aligned"], rows[sub]["noop"],
                 100 * rows[sub]["reduction"]), flush=True)
    return rows


def cohort_b():
    from dataio_mobi import build_subject, subjects
    rows = {}
    for sub in subjects():
        try:
            es = build_subject(sub, win=2.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        if es is None:
            continue
        X = es.X / (es.X.std() + 1e-30)
        g = es.trial.astype(int)
        Cfit = cov(X[g == 1])
        raw = [riemannian(Cfit, cov(X[g == t])) for t in (2, 3) if (g == t).any()]
        adapt = run_subject(X, g, [1], 0.2)
        noop = run_subject(X, g, [1], 0.0)
        rows[sub] = {"raw": float(np.mean(raw)), "aligned": float(np.mean(adapt)),
                     "noop": float(np.mean(noop)),
                     "reduction": float(1 - np.mean(adapt) / max(np.mean(raw), 1e-12)),
                     "n_test_sessions": len(raw)}
        print("  %s  raw %.3f -> aligned %.3f (noop %.3f)  reduction %5.1f%%"
              % (sub, rows[sub]["raw"], rows[sub]["aligned"], rows[sub]["noop"],
                 100 * rows[sub]["reduction"]), flush=True)
    return rows


def summarise(name, rows):
    if not rows:
        return None
    raw = np.array([r["raw"] for r in rows.values()])
    al = np.array([r["aligned"] for r in rows.values()])
    no = np.array([r["noop"] for r in rows.values()])
    red = 1 - al / np.maximum(raw, 1e-12)
    from scipy import stats
    t, p = stats.ttest_rel(al, raw)
    print("\n%s  n=%d subjects" % (name, len(rows)))
    print("  raw      %.3f +- %.3f" % (raw.mean(), raw.std()))
    print("  aligned  %.3f +- %.3f" % (al.mean(), al.std()))
    print("  no-op    %.3f +- %.3f   (control: shares the code path, never updates)"
          % (no.mean(), no.std()))
    print("  reduction %.1f%% +- %.1f%%   reduced in %d/%d subjects"
          % (100 * red.mean(), 100 * red.std(), int((red > 0).sum()), len(red)))
    print("  paired t-test aligned vs raw: t=%.3f  p=%.5f" % (t, p))
    return {"n": len(rows), "raw": float(raw.mean()), "aligned": float(al.mean()),
            "noop": float(no.mean()), "reduction": float(red.mean()),
            "reduction_sd": float(red.std()),
            "n_reduced": int((red > 0).sum()), "t": float(t), "p": float(p)}


def main():
    out = {}
    print("COHORT A -- ds007788 (fit sessions 1-3, test 4-9, weeks apart)")
    a = cohort_a()
    out["ds007788"] = {"per_subject": a, "summary": summarise("ds007788", a)}
    print("\nCOHORT B -- MoBI (fit trial 1, test trials 2-3)")
    b = cohort_b()
    out["mobi"] = {"per_subject": b, "summary": summarise("mobi", b)}
    OUT.write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % OUT.name)


if __name__ == "__main__":
    main()
