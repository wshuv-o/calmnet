"""Per-window z-scoring on vs off for three feature types (Table 3, representation block).

    python exp_zscore.py                                       # cohort A -> zscore.json
    COHORT=mobi ZS_OUT=zscore_mobi.json python exp_zscore.py   # cohort B
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as FE
from common import balanced_accuracy, grouped_split, imu_valid_mask
from dataio import build_epochs, list_sessions

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = RESULTS / os.environ.get("ZS_OUT", "zscore.json")
COHORT = os.environ.get("COHORT", "ds007788")
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3


def load_mobi(sub, win, zscore):
    from dataio_mobi import build_subject
    es = build_subject(sub, win=win, step=0.5, zscore=zscore)
    if es is None or len(es) < 100:
        raise RuntimeError("no MoBI data for %s" % sub)
    fit, test = es.by_trials([1]), es.by_trials([2, 3])
    ti, _ = grouped_split(fit.segment, fit.y, frac=0.3, seed=0)
    return {"Xf": fit.X[ti], "yf": fit.y[ti], "Xt": test.X, "yt": test.y, "st": test.trial,
            "imu_t": test.motion, "vt": np.ones(len(test), bool)}


def load(sub, win, zscore):
    if COHORT.startswith("mobi"):
        return load_mobi(sub, win, zscore)
    es = build_epochs(subject=sub, win=win, step=0.5, zscore=zscore)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, _ = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    f = lambda a: a[tr][ti]
    return {"Xf": f(es.X), "yf": f(es.y), "Xt": es.X[~tr], "yt": es.y[~tr],
            "st": es.session[~tr], "imu_t": es.imu_feats[~tr], "vt": valid[~tr]}


def feats(kind, Xf, Xt):
    if kind == "tangent_ea":
        return (FE.tangent(FE.euclidean_align(FE.covariances(Xf))),
                FE.tangent(FE.euclidean_align(FE.covariances(Xt))))
    if kind == "bandpower":
        return FE.log_var(FE.bandpass(Xf, 8, 30)), FE.log_var(FE.bandpass(Xt, 8, 30))
    if kind == "corr_only":                          # covariance scaled to unit diagonal
        def cc(X):
            C = FE.covariances(X)
            d = np.sqrt(np.einsum("nii->ni", C))[:, :, None]
            return FE.tangent(C / (d * d.transpose(0, 2, 1) + 1e-12))
        return cc(Xf), cc(Xt)
    raise ValueError(kind)


def main():
    global SUBJECTS
    wins = [float(w) for w in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2.0]
    if COHORT.startswith("mobi"):
        from dataio_mobi import subjects
        SUBJECTS = subjects()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for win in wins:
        for kind in ("tangent_ea", "bandpower", "corr_only"):
            for zs in (True, False):
                t0, accs, r2s = time.time(), [], []
                for sub in SUBJECTS:
                    try:
                        d = load(sub, win, zs)
                    except Exception as e:
                        print("  [skip] %s: %s" % (sub, e), flush=True)
                        continue
                    Ff, Ft = feats(kind, d["Xf"], d["Xt"])
                    clf = make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=3000, C=0.1,
                                                           class_weight="balanced"))
                    clf.fit(Ff, d["yf"])
                    accs.append(balanced_accuracy(d["yt"], clf.predict(Ft)))
                    vt = d["vt"]
                    if vt.sum() > 40:
                        r2s.append(FE.invariance_r2_conditional(
                            Ft[vt], d["imu_t"][vt], d["yt"][vt], d["st"][vt]))
                if not accs:
                    continue
                key = "w%s|%s|z%d" % (win, kind, int(zs))
                out[key] = {"win": win, "feat": kind, "zscore": bool(zs),
                            "acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                            "cond_r2": float(np.nanmean(r2s)),
                            "per_subject": {s: a for s, a in zip(SUBJECTS, accs)}}
                print("  %-12s zscore=%-5s acc %.3f  (%.0fs)"
                      % (kind, zs, out[key]["acc"], time.time() - t0), flush=True)
                OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
