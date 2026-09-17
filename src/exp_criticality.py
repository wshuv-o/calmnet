"""PATH A - Neural criticality / power-law features for walk/stop.

Question is NOT "does criticality beat band-power" (the confound caps everyone at
~0.65). The novel question: are criticality markers (1/f aperiodic slope, DFA long-
range temporal correlations, fractal dimension, spectral entropy) a MOVEMENT-INVARIANT
signature of walk intent? We compute them on broadband EEG, decode walk/stop, and test
how movement-confounded they are versus the band-power baseline.

Refs: Hardstone et al. 2012 (DFA/LRTC); aperiodic slope ~ E/I balance (Gao 2017);
DFA separates real vs imagined movement (Sole-Casals/roceedings 2019).
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from dataio import build_epochs, list_sessions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
SF = 100.0


# ---------- criticality / complexity measures ----------
def welch_psd(x, sf=SF):
    f, p = signal.welch(x, sf, nperseg=min(len(x), int(sf)), noverlap=None)
    return f, p


def aperiodic_slope(f, p, fmin=2.0, fmax=40.0):
    m = (f >= fmin) & (f <= fmax) & (f > 0)
    if m.sum() < 4:
        return np.nan
    lf, lp = np.log10(f[m]), np.log10(p[m] + 1e-20)
    A = np.vstack([lf, np.ones_like(lf)]).T
    return np.linalg.lstsq(A, lp, rcond=None)[0][0]        # slope (negative)


def spectral_entropy(f, p, fmin=1.0, fmax=45.0):
    m = (f >= fmin) & (f <= fmax)
    pp = p[m]
    pp = pp / (pp.sum() + 1e-20)
    return float(-(pp * np.log2(pp + 1e-20)).sum() / np.log2(len(pp) + 1e-20))


def bandpowers(f, p):
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
             "beta": (13, 30), "gamma": (30, 45)}
    tot = p[(f >= 1) & (f <= 45)].sum() + 1e-20
    return {k: float(p[(f >= a) & (f < b)].sum() / tot) for k, (a, b) in bands.items()}


def dfa(x, nmin=8, nmax=None, nscales=12):
    x = np.asarray(x, float)
    x = x - x.mean()
    y = np.cumsum(x)
    N = len(y)
    nmax = nmax or N // 4
    if nmax <= nmin:
        return np.nan
    scales = np.unique(np.floor(np.logspace(np.log10(nmin), np.log10(nmax), nscales)).astype(int))
    F = []
    for s in scales:
        nseg = N // s
        if nseg < 1:
            continue
        seg = y[:nseg * s].reshape(nseg, s)
        t = np.arange(s)
        # detrend each segment (linear)
        c1 = np.polyfit(t, seg.T, 1)
        fit = (np.outer(t, c1[0]) + c1[1]).T
        F.append(np.sqrt(((seg - fit) ** 2).mean()))
    F = np.asarray(F)
    scales = scales[:len(F)]
    m = F > 0
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(scales[m]), np.log(F[m]), 1)[0])


def higuchi_fd(x, kmax=8):
    x = np.asarray(x, float)
    N = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(1, (N - m) // k, dtype=int)
            if len(idx) == 0:
                continue
            dist = np.abs(x[m + idx * k] - x[m + (idx - 1) * k]).sum()
            norm = (N - 1) / (len(idx) * k) / k
            Lk.append(dist * norm)
        if Lk:
            L.append(np.log(np.mean(Lk) + 1e-20))
    if len(L) < 2:
        return np.nan
    lnk = np.log(1.0 / np.arange(1, len(L) + 1))
    return float(np.polyfit(lnk, L, 1)[0])


FEATS = ["slope", "dfa", "hfd", "sent", "delta", "theta", "alpha", "beta", "gamma", "ab_ratio"]


def window_features(w):
    """w: (C,T) broadband. Return per-feature mean & std across channels."""
    C = w.shape[0]
    rows = np.full((C, len(FEATS)), np.nan)
    for c in range(C):
        x = w[c]
        f, p = welch_psd(x)
        bp = bandpowers(f, p)
        rows[c] = [aperiodic_slope(f, p), dfa(x), higuchi_fd(x), spectral_entropy(f, p),
                   bp["delta"], bp["theta"], bp["alpha"], bp["beta"], bp["gamma"],
                   bp["alpha"] / (bp["beta"] + 1e-9)]
    mu = np.nanmean(rows, axis=0)
    sd = np.nanstd(rows, axis=0)
    return np.concatenate([mu, sd])


def build_crit(sub):
    # broadband, longer windows for reliable criticality, no z-score (keep amplitude shape)
    es = build_epochs(subject=sub, win=4.0, step=2.0, l_freq=1.0, h_freq=45.0, zscore=False)
    n = len(es)
    print(f"  [{sub}] {n} windows ({int(es.y.sum())} walk / {int((es.y==0).sum())} stop); extracting...", flush=True)
    F = np.stack([window_features(es.X[i]) for i in range(n)])
    ok = ~np.isnan(F).any(axis=1)
    return es.subset(ok), F[ok]


def run(sub):
    es, F = build_crit(sub)
    sess = sorted(set(int(s) for s in np.unique(es.session)))
    tr = np.array([s in set(sess[:3]) for s in es.session])
    te = ~tr
    if tr.sum() < 10 or te.sum() < 10:
        print(f"  [{sub}] not enough data"); return None
    # 1) criticality decode (train ses 1-3 -> test rest)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(F[tr], es.y[tr])
    from abstain import balanced_accuracy
    acc = balanced_accuracy(es.y[te], clf.predict(F[te]))
    # 2) IMU-only baseline on same split
    M = es.imu_feats
    imu = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    imu.fit(M[tr], es.y[tr])
    imu_acc = balanced_accuracy(es.y[te], imu.predict(M[te]))
    # 3) movement-invariance of criticality: how well do crit feats PREDICT motion?
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    motion = M[:, 1]  # head accel std (motion magnitude)
    rr = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    rr.fit(F[tr], motion[tr])
    mot_r2 = r2_score(motion[te], rr.predict(F[te]))   # high => crit feats ENCODE movement
    # 4) does criticality decode survive regressing motion out of the features?
    #    residualize each feature on IMU (12-d) using train fit
    res = Ridge(alpha=1.0).fit(M[tr], F[tr])
    Fres = F - res.predict(M)
    clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf2.fit(Fres[tr], es.y[tr])
    acc_res = balanced_accuracy(es.y[te], clf2.predict(Fres[te]))
    out = {"subject": sub, "n": int(len(es)), "crit_acc": acc, "imu_acc": imu_acc,
           "crit_predicts_motion_r2": float(mot_r2), "crit_acc_motion_removed": acc_res}
    print(f"  [{sub}] crit={acc:.3f} | IMU-only={imu_acc:.3f} | crit->motion R2={mot_r2:+.3f} "
          f"| crit(motion-removed)={acc_res:.3f}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (criticality) ########", flush=True)
        r = run(sub)
        if r:
            res[sub] = r
            (RESULTS / "criticality.json").write_text(json.dumps(res, indent=2))
    if res:
        g = lambda k: float(np.nanmean([res[s][k] for s in res]))
        print("\n============ CRITICALITY (mean over subjects) ============")
        print(f"  criticality decode        : {g('crit_acc'):.3f}")
        print(f"  IMU-only (movement) baseline: {g('imu_acc'):.3f}")
        print(f"  crit features predict motion (R2): {g('crit_predicts_motion_r2'):+.3f}   (high = confounded)")
        print(f"  criticality, motion removed : {g('crit_acc_motion_removed'):.3f}   (holds up => invariant signal)")
