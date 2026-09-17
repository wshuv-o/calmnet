"""Measure the drift timescale without a decoder.

The admissible band tau_blk < mu < tau_drift is claimed to be checkable from
the recording protocol before any model is fitted. tau_blk is: it is the
class-block length, read off the protocol. tau_drift is not -- the paper so far
infers it from where alignment stops helping, which is the very thing the band
is supposed to predict.

This measures it directly and without a model. Within each session, windows are
taken in recording order, partitioned into consecutive blocks, and a spatial
covariance is computed per block. The affine-invariant Riemannian distance
between blocks separated by lag L is then averaged over all pairs at that lag.
If the signal's second-order statistics drift, d(L) rises with L and saturates
once the blocks are far enough apart to be effectively independent. The
saturation point is that timescale.

d(L) is fitted with d(L) = d_inf * (1 - exp(-L / tau)), and tau is reported in
windows so it is directly comparable with the estimator memory mu = 32 / m.

Only covariances and eigenvalues -- no training, no GPU.

Writes results/tau_drift.json.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "tau_drift.json"

# Windows per block. 32 matches the inference batch, so a lag of 1 block is one
# estimator update and lags read directly in units of the running estimate.
BLOCK = 32
MAX_LAG = 40


def cov(a):
    """Trace-normalised mean spatial covariance -- same estimator as exp_drift."""
    ac = a - a.mean(-1, keepdims=True)
    c = np.einsum("nct,ndt->cd", ac, ac) / (len(a) * a.shape[-1])
    return c / (np.trace(c) / c.shape[0] + 1e-12)


def riemannian(A, B):
    """Affine-invariant distance: sqrt(sum log^2 eig(A^-1/2 B A^-1/2))."""
    n = len(A)
    A = A + 1e-6 * np.eye(n)
    B = B + 1e-6 * np.eye(n)
    Ai = la.fractional_matrix_power(A, -0.5).real
    w = np.clip(np.linalg.eigvalsh(Ai @ B @ Ai), 1e-10, None)
    return float(np.sqrt((np.log(w) ** 2).sum()))


def curve_for_session(X):
    """Mean Riemannian distance against lag, in blocks, within one session."""
    nb = len(X) // BLOCK
    if nb < 4:
        return None
    covs = [cov(X[i * BLOCK:(i + 1) * BLOCK]) for i in range(nb)]
    out = {}
    for lag in range(1, min(MAX_LAG, nb - 1) + 1):
        ds = [riemannian(covs[i], covs[i + lag]) for i in range(nb - lag)]
        out[lag] = float(np.mean(ds))
    return out


def saturating(L, d_inf, tau):
    return d_inf * (1.0 - np.exp(-L / tau))


def fit_tau(lags, d):
    """Return (tau_windows, d_inf, ok). tau in WINDOWS, not blocks."""
    lags = np.asarray(lags, dtype=float)
    d = np.asarray(d, dtype=float)
    try:
        p0 = [float(d.max()), max(1.0, len(lags) / 4.0)]
        popt, _ = curve_fit(saturating, lags, d, p0=p0, maxfev=20000,
                            bounds=([1e-6, 1e-3], [np.inf, 1e4]))
        d_inf, tau_blocks = float(popt[0]), float(popt[1])
        return tau_blocks * BLOCK, d_inf, True
    except Exception as e:
        print("    [fit failed] %s" % type(e).__name__)
        return float("nan"), float("nan"), False


def run_cohort(name, sessions_of):
    """sessions_of yields (subject, [array of windows per session])."""
    per_sub, curves = {}, {}
    for sub, sess_list in sessions_of():
        cs = [c for c in (curve_for_session(Xs) for Xs in sess_list) if c]
        if not cs:
            print("  [skip] %s: no session long enough" % sub)
            continue
        lags = sorted(set(int(k) for c in cs for k in c))
        mean_d = [float(np.mean([c[l] for c in cs if l in c])) for l in lags]
        tau_w, d_inf, ok = fit_tau(lags, mean_d)
        per_sub[sub] = {"tau_windows": tau_w, "d_inf": d_inf,
                        "n_sessions": len(cs), "fit_ok": bool(ok)}
        curves[sub] = {"lags_blocks": lags, "mean_distance": mean_d}
        print("  %-8s tau = %8.1f windows   d_inf = %.3f   (%d sessions)"
              % (sub, tau_w, d_inf, len(cs)), flush=True)

    taus = [v["tau_windows"] for v in per_sub.values()
            if v["fit_ok"] and np.isfinite(v["tau_windows"])]
    summary = {
        "n_subjects": len(per_sub),
        "tau_windows_median": float(np.median(taus)) if taus else float("nan"),
        "tau_windows_mean": float(np.mean(taus)) if taus else float("nan"),
        "tau_windows_min": float(np.min(taus)) if taus else float("nan"),
        "tau_windows_max": float(np.max(taus)) if taus else float("nan"),
    }
    print("  %s: median tau = %.1f windows (range %.1f - %.1f over %d subjects)"
          % (name, summary["tau_windows_median"], summary["tau_windows_min"],
             summary["tau_windows_max"], len(taus)), flush=True)
    return {"summary": summary, "per_subject": per_sub, "curves": curves}


def cohort_a():
    from dataio import build_epochs, list_sessions
    subs = ["sub-0%d" % i for i in range(1, 8)]

    def gen():
        for sub in subs:
            try:
                es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False)
            except Exception as e:
                print("  [skip] %s: %s" % (sub, str(e)[:80]), flush=True)
                continue
            X = es.X / (es.X.std() + 1e-30)
            g = es.session.astype(int)
            yield sub, [X[g == s] for s in sorted(set(int(v) for v in np.unique(g)))]
    return gen


def cohort_b():
    from dataio_mobi import build_subject, subjects

    def gen():
        for sub in subjects():
            try:
                es = build_subject(sub, win=2.0, step=0.5, zscore=False)
            except Exception as e:
                print("  [skip] %s: %s" % (sub, str(e)[:80]), flush=True)
                continue
            if es is None:
                continue
            X = es.X / (es.X.std() + 1e-30)
            g = es.trial.astype(int)
            yield sub, [X[g == t] for t in sorted(set(int(v) for v in np.unique(g)))]
    return gen


def main():
    which = os.environ.get("TAU_COHORT", "A").upper()
    out = {}
    if which in ("A", "BOTH"):
        print("Cohort A (ds007788), block = %d windows" % BLOCK, flush=True)
        out["A"] = run_cohort("cohort A", cohort_a())
    if which in ("B", "BOTH"):
        print("Cohort B (MoBI), block = %d windows" % BLOCK, flush=True)
        out["B"] = run_cohort("cohort B", cohort_b())
    out["block_windows"] = BLOCK
    OUT.write_text(json.dumps(out, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
