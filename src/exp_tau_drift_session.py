"""Drift timescale measured ACROSS sessions, without a decoder.

Why this exists alongside exp_tau_drift.py
------------------------------------------
That script partitions each session into blocks and fits the rise of
Riemannian distance against lag WITHIN the session. On cohort A it returns
tau = 17.9 windows (about 9 s). That is the timescale on which the covariance
fluctuates moment to moment, not the timescale on which it drifts between
recordings, and the two differ by orders of magnitude.

Using the within-session number in the admissible band gives

    cohort A:  tau_blk = 18  <  mu  <  17.9 = tau_drift   ->  EMPTY

which contradicts cohort A being the cohort where the layer helps and where
fast adaptation (mu = 160 windows) beats slow adaptation by about 0.09. A
160-window memory sitting far above an 18-window fluctuation scale is the
estimator averaging that fluctuation out, which is the behaviour we want, not
a violation of the band.

The quantity the band needs is the timescale of session-to-session drift: the
thing the alignment layer exists to correct, and the thing Section 1 describes
as acting over weeks. This script measures that.

Method
------
For each subject: one mean spatial covariance per session, then the
affine-invariant Riemannian distance between every pair of sessions as a
function of their separation in session index. If the montage drifts
monotonically, d(g) rises with the gap g and saturates once sessions are far
enough apart to be effectively unrelated. Fit

    d(g) = d_inf * (1 - exp(-g / tau_sessions))

and report tau in sessions and in windows (tau_sessions * windows per session)
so it is directly comparable with the estimator memory mu = 32 / m.

Honesty note: a saturating fit needs several distinct gaps. Cohort A has 9
sessions (gaps 1-8) and supports one. Cohort B has 3 trials (gaps 1-2) and does
not; for it we report the raw distances and mark the fit as unavailable rather
than emitting a number the data cannot support. The earlier script returned
n=0 and nan for cohort B silently.

No training, no GPU. Writes results/tau_drift_session.json.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_drift import riemannian                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "tau_drift_session.json")


def cov(a):
    """Mean spatial covariance, NOT trace-normalised.

    Trace normalisation is a data-dependent rescale; it breaks the affine
    invariance the distance relies on (see exp_drift.cov for the measured
    consequence).
    """
    ac = a - a.mean(-1, keepdims=True)
    return np.einsum("nct,ndt->cd", ac, ac) / (len(a) * a.shape[-1])


def saturating(g, d_inf, tau):
    return d_inf * (1.0 - np.exp(-g / tau))


def analyse(sess_covs, windows_per_session):
    """sess_covs: list of (session_index, covariance). Returns a dict."""
    n = len(sess_covs)
    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(sess_covs[j][0] - sess_covs[i][0])
            pairs.setdefault(gap, []).append(
                riemannian(sess_covs[i][1], sess_covs[j][1]))
    gaps = sorted(pairs)
    curve = [(int(g), float(np.mean(pairs[g]))) for g in gaps]
    out = {"n_sessions": n, "curve": curve,
           "windows_per_session": float(windows_per_session)}

    if len(gaps) < 4:
        out.update({"fit_ok": False,
                    "reason": "only %d distinct session gaps; a saturating "
                              "fit needs at least 4" % len(gaps)})
        return out

    g = np.array([c[0] for c in curve], float)
    d = np.array([c[1] for c in curve], float)
    try:
        popt, _ = curve_fit(saturating, g, d,
                            p0=[float(d.max()), 2.0],
                            bounds=([0, 1e-3], [np.inf, 1e4]), maxfev=20000)
        d_inf, tau_s = float(popt[0]), float(popt[1])
        ok = np.isfinite(tau_s) and tau_s > 0
    except Exception as e:
        out.update({"fit_ok": False, "reason": "%s" % type(e).__name__})
        return out

    out.update({"fit_ok": bool(ok), "d_inf": d_inf,
                "tau_sessions": tau_s,
                "tau_windows": tau_s * windows_per_session})
    return out


def cohort_a():
    from dataio import build_epochs, list_sessions
    per = {}
    for sub in ["sub-0%d" % i for i in range(1, 8)]:
        try:
            es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        X = es.X / (es.X.std() + 1e-30)
        g = es.session.astype(int)
        pres = sorted(set(int(v) for v in np.unique(g)))
        order = [s for s in list_sessions(sub) if s in pres]
        cs = [(k, cov(X[g == s])) for k, s in enumerate(order)
              if (g == s).sum() >= 32]
        if len(cs) < 3:
            print("  [skip] %s: %d usable sessions" % (sub, len(cs)), flush=True)
            continue
        wps = float(len(X)) / max(len(cs), 1)
        per[sub] = analyse(cs, wps)
        r = per[sub]
        print("  %-8s sessions %d  tau = %s"
              % (sub, r["n_sessions"],
                 ("%.2f sessions = %.0f windows" % (r["tau_sessions"],
                                                    r["tau_windows"]))
                 if r.get("fit_ok") else "no fit (%s)" % r.get("reason")),
              flush=True)
    return per


def cohort_b():
    from dataio_mobi import build_subject, subjects
    per = {}
    for sub in subjects():
        try:
            es = build_subject(sub, win=2.0, step=0.5, zscore=False)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        if es is None:
            continue
        X = es.X / (es.X.std() + 1e-30)
        key = "session" if hasattr(es, "session") else None
        g = (es.session if key else es.trial).astype(int)
        order = sorted(set(int(v) for v in np.unique(g)))
        cs = [(k, cov(X[g == s])) for k, s in enumerate(order)
              if (g == s).sum() >= 32]
        if len(cs) < 2:
            print("  [skip] %s: %d usable trials" % (sub, len(cs)), flush=True)
            continue
        wps = float(len(X)) / max(len(cs), 1)
        per[sub] = analyse(cs, wps)
        r = per[sub]
        print("  %-8s trials %d  %s"
              % (sub, r["n_sessions"],
                 ("tau = %.2f trials = %.0f windows"
                  % (r["tau_sessions"], r["tau_windows"]))
                 if r.get("fit_ok")
                 else "no fit (%s); distances %s"
                      % (r.get("reason"),
                         ", ".join("gap %d: %.2f" % c for c in r["curve"]))),
              flush=True)
    return per


def summarise(per):
    taus = [v["tau_windows"] for v in per.values() if v.get("fit_ok")]
    if not taus:
        return {"n_fitted": 0, "note": "no subject supported a saturating fit"}
    return {"n_fitted": len(taus),
            "tau_windows_median": float(np.median(taus)),
            "tau_windows_min": float(min(taus)),
            "tau_windows_max": float(max(taus))}


def main():
    out = {}
    print("COHORT A -- ds007788 (9 sessions, weeks apart)", flush=True)
    a = cohort_a()
    out["A"] = {"per_subject": a, "summary": summarise(a)}
    print("COHORT B -- MoBI (3 trials, same day)", flush=True)
    try:
        b = cohort_b()
    except Exception as e:
        print("  cohort B unavailable: %s" % e, flush=True)
        b = {}
    out["B"] = {"per_subject": b, "summary": summarise(b)}

    print(flush=True)
    for coh, blk in (("A", "18"), ("B", "309")):
        s = out[coh]["summary"]
        if s.get("n_fitted"):
            print("cohort %s: tau_drift median %.0f windows (n=%d fitted). "
                  "Band is %s < mu < %.0f"
                  % (coh, s["tau_windows_median"], s["n_fitted"], blk,
                     s["tau_windows_median"]), flush=True)
        else:
            print("cohort %s: %s" % (coh, s.get("note")), flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/tau_drift_session.json", flush=True)


if __name__ == "__main__":
    main()
