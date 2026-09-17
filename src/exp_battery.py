"""IDEAS 5, 7, 8 battery - Surface Laplacian, Riemannian (MDRM + Euclidean Alignment),
Phase connectivity. Same windows, same decode + movement-invariance protocol so results
tabulate directly against the ~0.65 ceiling.

Each method: decode walk/stop (train ses 1-3 / test rest) AND a movement-invariant decode
(regress the head-motion scalar out of the features first). If any method keeps a decode
after motion removal that the others cannot, it found something. Otherwise the ceiling holds.
"""
import os, sys, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import signal, linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from dataio import build_epochs, DATA_ROOT
from graphnet import electrode_adjacency
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]


# ---------- feature transforms ----------
def laplacian_op(ch_names):
    A = electrode_adjacency(ch_names, k=6).numpy()      # normalised kNN adjacency (+self loops)
    np.fill_diagonal(A, 0)
    A = A / (A.sum(1, keepdims=True) + 1e-9)
    return np.eye(len(ch_names)) - A                    # small Laplacian: x - mean(neighbours)


def logbp(w):
    v = np.log(w.var(axis=1) + 1e-12)
    return v


def feat_laplacian(X, L):
    return np.stack([logbp(L @ X[i]) for i in range(len(X))])


def feat_bandpower(X):
    return np.stack([logbp(X[i]) for i in range(len(X))])


def covs(X, reg=1e-3):
    C = X.shape[1]
    out = np.empty((len(X), C, C))
    for i in range(len(X)):
        c = X[i] @ X[i].T / X[i].shape[1]
        out[i] = c + reg * np.trace(c) / C * np.eye(C)
    return out


def euclidean_align(Cw, sess):
    """Per-session Euclidean Alignment: whiten by session mean covariance."""
    out = np.empty_like(Cw)
    for s in np.unique(sess):
        m = sess == s
        R = Cw[m].mean(0)
        Rinv = linalg.fractional_matrix_power(R, -0.5).real
        out[m] = np.einsum("ij,njk,kl->nil", Rinv, Cw[m], Rinv)
    return out


def riemann_mean(Cs, n_iter=8):
    M = Cs.mean(0)
    for _ in range(n_iter):
        Mi = linalg.fractional_matrix_power(M, -0.5).real
        S = np.mean([linalg.logm(Mi @ c @ Mi) for c in Cs], 0)
        M = linalg.fractional_matrix_power(M, 0.5).real @ linalg.expm(S) @ linalg.fractional_matrix_power(M, 0.5).real
        if np.linalg.norm(S) < 1e-6:
            break
    return M.real


def riemann_dist(A, B):
    Ai = linalg.fractional_matrix_power(A, -0.5).real
    return np.linalg.norm(linalg.logm(Ai @ B @ Ai))


def mdrm(Ctr, ytr, Cte):
    m0 = riemann_mean(Ctr[ytr == 0]); m1 = riemann_mean(Ctr[ytr == 1])
    pred = np.array([0 if riemann_dist(c, m0) < riemann_dist(c, m1) else 1 for c in Cte])
    return pred


def feat_plv(X):
    """node-strength phase-locking value per channel (60-D)."""
    out = np.empty((len(X), X.shape[1]))
    for i in range(len(X)):
        ph = np.angle(signal.hilbert(X[i], axis=1))
        z = np.exp(1j * ph)
        plv = np.abs(z @ z.conj().T) / X.shape[2]        # (C,C)
        np.fill_diagonal(plv, 0)
        out[i] = plv.mean(1)
    return out


def evl(F, y, g, motion):
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    if tr.sum() < 10 or te.sum() < 10 or len(set(y[te])) < 2:
        return np.nan, np.nan
    def acc(Z):
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        c.fit(Z[tr], y[tr]); return balanced_accuracy(y[te], c.predict(Z[te]))
    raw = acc(F)
    res = Ridge(alpha=1.0).fit(motion[tr].reshape(-1, 1), F[tr])
    Finv = F - res.predict(motion.reshape(-1, 1))
    return raw, acc(Finv)


def run(subject):
    es = build_epochs(subject=subject, win=2.0, step=0.5, l_freq=8.0, h_freq=30.0, zscore=False)
    X, y, g = es.X, es.y, es.session.astype(int)
    motion = es.imu_feats[:, 1]                          # head accel std
    L = laplacian_op(es.ch_names)
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    out = {"subject": subject, "n": int(len(y))}
    # band-power baseline + laplacian + connectivity via shared eval
    for name, F in (("bandpower", feat_bandpower(X)),
                    ("laplacian", feat_laplacian(X, L)),
                    ("plv", feat_plv(X))):
        raw, inv = evl(F, y, g, motion)
        out[name + "_raw"], out[name + "_inv"] = raw, inv
    # Riemannian MDRM + Euclidean Alignment (uses distances, no residualisation -> report raw only)
    try:
        Cw = euclidean_align(covs(X), g)
        acc_r = balanced_accuracy(y[te], mdrm(Cw[tr], y[tr], Cw[te]))
    except Exception as e:
        acc_r = np.nan
    out["riemann_ea"] = acc_r
    print(f"  [{subject}] BP {out['bandpower_raw']:.3f}/{out['bandpower_inv']:.3f}  "
          f"Lap {out['laplacian_raw']:.3f}/{out['laplacian_inv']:.3f}  "
          f"PLV {out['plv_raw']:.3f}/{out['plv_inv']:.3f}  Riem+EA {acc_r:.3f}  (raw/motion-removed)", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (battery: Laplacian/Riemann/PLV) ########", flush=True)
        try:
            res[sub] = run(sub)
            (RESULTS / "battery.json").write_text(json.dumps(res, indent=2))
        except Exception as e:
            print(f"  [{sub}] ERROR {e}")
    if res:
        m = lambda k: float(np.nanmean([res[s].get(k, np.nan) for s in res if res[s].get(k, np.nan) == res[s].get(k, np.nan)]))
        print("\n============ BATTERY (mean; raw / motion-removed) ============")
        for nm in ("bandpower", "laplacian", "plv"):
            print(f"  {nm:10}: {m(nm+'_raw'):.3f} / {m(nm+'_inv'):.3f}")
        print(f"  riemann+EA: {m('riemann_ea'):.3f} (raw)")
