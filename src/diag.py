"""CODE AUDIT - is the ~0.65 ceiling a bug or real? Stress-test every suspect component.

Checks (per subject, standard training-task pipeline):
  1. real decode            - band-power + LR, session split (baseline)
  2. LABEL-SHUFFLE x8       - break X<->y. MUST be ~0.50. If > 0.55 => LEAKAGE bug.
  3. SEGMENT LEAK           - any contiguous state-segment split across train & test?
  4. POSITIVE CTRL (session)- decode which session a window is from. MUST be high => features carry info.
  5. POSITIVE CTRL (inject) - add a tiny class-locked oscillation. MUST be recovered => pipeline works.
  6. IMU-only               - recompute movement baseline on this split (sanity vs 0.84).
  7. NORMALISATION          - per-window z-score vs raw amplitude for the SAME decoder.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from abstain import balanced_accuracy

SUBJECTS = sys.argv[1:] or ["sub-01", "sub-06"]
SF = 100.0
rng = np.random.default_rng(0)


def logbp(X):
    """log band-power mu+beta per channel (works on non-z-scored data)."""
    def bp(x, lo, hi):
        f, p = signal.welch(x, SF, nperseg=min(x.shape[-1], int(SF)), axis=-1)
        m = (f >= lo) & (f < hi)
        return np.log(p[..., m].sum(-1) + 1e-12)
    return np.concatenate([bp(X, 8, 13), bp(X, 13, 30)], axis=1)


def dec(F, y, tr, te):
    c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    c.fit(F[tr], y[tr])
    return balanced_accuracy(y[te], c.predict(F[te]))


def run(sub):
    print(f"\n######## {sub} ########", flush=True)
    es = build_epochs(subject=sub, win=2.0, step=0.5, l_freq=8.0, h_freq=30.0, zscore=False)
    X, y, g, seg = es.X, es.y, es.session.astype(int), es.segment
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    F = logbp(X)
    print(f"  windows={len(y)} walk={int(y.sum())} stop={int((y==0).sum())} | train={tr.sum()} test={te.sum()}")

    # 1. real
    real = dec(F, y, tr, te)
    print(f"  [1] real band-power decode         : {real:.3f}")

    # 2. label shuffle (leakage test)  -- shuffle WITHIN each split to preserve marginals
    shuf = []
    for _ in range(8):
        ys = y.copy()
        ys[tr] = rng.permutation(ys[tr]); ys[te] = rng.permutation(ys[te])
        shuf.append(dec(F, ys, tr, te))
    print(f"  [2] LABEL-SHUFFLE (must be ~0.50)  : {np.mean(shuf):.3f} +/- {np.std(shuf):.3f}   "
          f"{'<-- LEAK!' if np.mean(shuf) > 0.55 else 'ok (no leak)'}")

    # 3. segment leak: do any state-segments appear in BOTH train and test?
    seg_tr, seg_te = set(seg[tr]), set(seg[te])
    overlap = seg_tr & seg_te
    print(f"  [3] SEGMENT LEAK (must be 0)       : {len(overlap)} segments shared across train/test")

    # 4. positive control: decode session identity (should be EASY if features carry structure)
    #    binary: early (ses<=median) vs late sessions, on the TEST windows via inner split
    med = np.median(g)
    ylate = (g > med).astype(int)
    idx = np.arange(len(y)); rng.shuffle(idx)
    half = len(idx) // 2
    sp_tr = np.zeros(len(y), bool); sp_tr[idx[:half]] = True; sp_te = ~sp_tr
    sess_acc = dec(F, ylate, sp_tr, sp_te) if len(set(ylate)) == 2 else float("nan")
    print(f"  [4] POS CTRL decode session-half   : {sess_acc:.3f}   (high => pipeline extracts real info)")

    # 5. positive control: inject a faint class-locked 10 Hz burst into walk windows, recover it
    t = np.arange(X.shape[2]) / SF
    burst = 0.15 * np.sin(2 * np.pi * 10 * t)                 # small vs unit-variance signal
    Xinj = X.copy()
    Xinj[y == 1] += burst[None, None, :]                     # add to all channels of walk windows
    inj = dec(logbp(Xinj), y, tr, te)
    print(f"  [5] POS CTRL inject 10Hz -> recover: {inj:.3f}   (must jump above real => decoder works)")

    # 6. IMU-only on this split
    imu = dec(es.imu_feats, y, tr, te)
    print(f"  [6] IMU-only movement baseline     : {imu:.3f}")

    # 7. normalisation: same band-power decode but on per-window z-scored data
    esz = build_epochs(subject=sub, win=2.0, step=0.5, l_freq=8.0, h_freq=30.0, zscore=True)
    zacc = dec(logbp(esz.X), esz.y, tr, te)
    print(f"  [7] band-power on z-scored windows : {zacc:.3f}   (vs raw {real:.3f}; z-score kills power info)")

    # 8. THE KEY TEST: does the movement-invariant step over-remove real neural signal?
    from sklearn.linear_model import Ridge
    M = es.imu_feats
    burst2 = 0.15 * np.sin(2 * np.pi * 10 * (np.arange(X.shape[2]) / SF))
    def inv_decode(Feat, lab):
        r = Ridge(alpha=1.0).fit(M[tr], Feat[tr])
        return dec(Feat - r.predict(M), lab, tr, te)
    # 8a: signal injected into WALK windows (movement-correlated) -> invariance SHOULD remove it
    Xa = X.copy(); Xa[y == 1] += burst2[None, None, :]
    inv_walk = inv_decode(logbp(Xa), y)
    # 8b: signal injected for a MOVEMENT-INDEPENDENT synthetic label -> invariance MUST keep it
    ysyn = rng.integers(0, 2, len(y))
    Xb = X.copy(); Xb[ysyn == 1] += burst2[None, None, :]
    inv_syn = inv_decode(logbp(Xb), ysyn)
    imu_pred_syn = dec(M, ysyn, tr, te)                       # confirm synth is IMU-independent (~0.5)
    verdict = "ok: invariance KEEPS movement-independent signal => 0.65 is REAL" if inv_syn > 0.75 \
        else "<-- BUG: invariance destroys even movement-independent signal"
    print(f"  [8a] inject into walk (movement-corr) -> invariant decode {inv_walk:.3f}  (should drop, correct)")
    print(f"  [8b] inject movement-INDEP synth label -> invariant decode {inv_syn:.3f}  "
          f"(IMU predicts synth {imu_pred_syn:.2f}~0.5)")
    print(f"       VERDICT: {verdict}")
    return dict(subject=sub, real=real, shuffle=float(np.mean(shuf)), seg_leak=len(overlap),
                sess_ctrl=sess_acc, inject=inj, imu=imu, zscored=zacc,
                inv_walk=inv_walk, inv_syn=inv_syn)


if __name__ == "__main__":
    out = {}
    for s in SUBJECTS:
        try:
            out[s] = run(s)
        except Exception as e:
            import traceback; traceback.print_exc()
    (Path(__file__).resolve().parent.parent / "results" / "diag.json").write_text(json.dumps(out, indent=2))
    print("\n=== VERDICT GUIDE ===")
    print("  leak bug   if [2] shuffle > 0.55 or [3] segment-leak > 0")
    print("  broken code if [5] inject does NOT jump above [1] real")
    print("  ceiling real if shuffle~0.50, no seg-leak, inject recovers, yet real stays ~0.65-0.85")
