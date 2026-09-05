"""IDEA 6 - Filter-Bank Common Spatial Patterns (FBCSP), the classic MI gold standard
(Ang et al. 2012, BCI Competition IV winner). Sub-band CSP spatial filters + log-variance,
concatenated across a filter bank, then LDA. Same decode + movement-invariance protocol.
"""
import os, sys, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from dataio import build_epochs
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 38)]
SF = 100.0


def bandfilter(X, lo, hi):
    b, a = signal.butter(4, [lo / (SF / 2), hi / (SF / 2)], btype="band")
    return signal.filtfilt(b, a, X, axis=2)


def fbcsp_feats(Xtr, ytr, Xte, n_comp=4):
    Ftr, Fte = [], []
    for lo, hi in BANDS:
        xt = bandfilter(Xtr, lo, hi).astype(np.float64)
        xe = bandfilter(Xte, lo, hi).astype(np.float64)
        try:
            csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True, norm_trace=False)
            csp.fit(xt, ytr)
            Ftr.append(csp.transform(xt)); Fte.append(csp.transform(xe))
        except Exception:
            continue
    if not Ftr:
        return None, None
    return np.concatenate(Ftr, 1), np.concatenate(Fte, 1)


def run(subject):
    es = build_epochs(subject=subject, win=2.0, step=0.5, l_freq=4.0, h_freq=40.0, zscore=False)
    X, y, g = es.X.astype(np.float64), es.y, es.session.astype(int)
    motion = es.imu_feats[:, 1]
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    if tr.sum() < 12 or te.sum() < 12 or len(set(y[te])) < 2:
        print(f"  [{subject}] bad split"); return None
    Ftr, Fte = fbcsp_feats(X[tr], y[tr], X[te])
    if Ftr is None:
        print(f"  [{subject}] CSP failed"); return None
    clf = LinearDiscriminantAnalysis()
    clf.fit(Ftr, y[tr])
    raw = balanced_accuracy(y[te], clf.predict(Fte))
    # movement-invariant: regress head-motion out of the CSP features (train fit), re-decode
    res = Ridge(alpha=1.0).fit(motion[tr].reshape(-1, 1), Ftr)
    Ftr2 = Ftr - res.predict(motion[tr].reshape(-1, 1))
    Fte2 = Fte - res.predict(motion[te].reshape(-1, 1))
    c2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    c2.fit(Ftr2, y[tr]); inv = balanced_accuracy(y[te], c2.predict(Fte2))
    out = {"subject": subject, "fbcsp_raw": raw, "fbcsp_inv": inv}
    print(f"  [{subject}] FBCSP raw={raw:.3f} | motion-removed={inv:.3f}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (FBCSP) ########", flush=True)
        try:
            r = run(sub)
            if r:
                res[sub] = r; (RESULTS / "fbcsp.json").write_text(json.dumps(res, indent=2))
        except Exception as e:
            print(f"  [{sub}] ERROR {e}")
    if res:
        m = lambda k: float(np.nanmean([res[s][k] for s in res]))
        print("\n============ FBCSP (mean) ============")
        print(f"  FBCSP raw            : {m('fbcsp_raw'):.3f}")
        print(f"  FBCSP motion-removed : {m('fbcsp_inv'):.3f}")
