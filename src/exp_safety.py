"""Safety-metric improvements on the (primary) band-power CALM-Net, longitudinal:
  (1) deep ensemble (K models)         -> better confidence / calibration / abstention
  (2) per-session temperature recal    -> lower ECE under drift
  (3) class-conditional, safety-asymmetric threshold that CONTROLS the wrong-walk rate
These improve the metrics the paper claims (executed-command safety, calibrated
confidence, coverage), not the accuracy ceiling."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from mid import train_mid, predict_mid
from calibrate import fit_temperature, softmax_np, expected_calibration_error
from abstain import balanced_accuracy, confidence_auroc, selective_risk_at_coverage

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS, K, COV, BETA = 3, 100, 3, 0.8, 0.05   # K=ensemble size, BETA=wrong-walk target


def _T(logits, y):
    t = fit_temperature(logits, y)
    return 1.0 if not np.isfinite(t) else float(np.clip(t, 0.5, 5.0))


def exec_acc(y, p, cov=COV):
    c = p.max(1); k = max(1, int(round(cov * len(y)))); idx = np.argsort(-c)[:k]
    return balanced_accuracy(y[idx], p[idx].argmax(1))


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    train, test = sess[:N_TRAIN], sess[N_TRAIN:]
    tr = es.by_sessions(train); ti, ci = grouped_split(tr.segment, tr.y, frac=0.3, seed=0)

    # ---- deep ensemble of K MID models (PROBABILITY averaging of calibrated members) ----
    models = [train_mid(tr.X[ti], tr.y[ti], tr.imu_feats[ti], tr.X[ci], tr.y[ci],
                        epochs=EPOCHS, lam_adv=1.0, lam_dec=1.0, seed=s) for s in range(K)]
    logit = {}                                          # cache per-(model,split) logits
    def L(mi, X, key):
        kk = (mi, key)
        if kk not in logit:
            logit[kk] = predict_mid(models[mi], X)[0]
        return logit[kk]

    Tk = [_T(L(i, tr.X[ci], f"cal{i}"), tr.y[ci]) for i in range(K)]     # per-model global T
    Tg_single = Tk[0]

    def ens_prob(Xlogits, Ts):                          # Xlogits: list of K logit arrays
        return np.mean([softmax_np(Xlogits[i], Ts[i]) for i in range(K)], axis=0)

    p_cal = ens_prob([L(i, tr.X[ci], f"cal{i}") for i in range(K)], Tk)
    stop_mask = tr.y[ci] == 0
    tau_walk = float(np.quantile(p_cal[stop_mask, 1], 1 - BETA)) if stop_mask.any() else 0.5

    # ---- evaluate over test sessions ----
    per = {}
    for s in test:
        te = es.by_sessions([s]); ys = te.y
        lgs = [predict_mid(m, te.X)[0] for m in models]
        p_s = softmax_np(lgs[0], Tg_single)
        p_e = ens_prob(lgs, Tk)
        # per-session recal: refit each member's T on a 30% grouped slice of this session
        ri, ei = grouped_split(te.segment, ys, frac=0.3, seed=1)
        Tk_sess = [_T(lgs[i][ri], ys[ri]) for i in range(K)]
        p_ps = ens_prob([lgs[i][ei] for i in range(K)], Tk_sess)
        cor_s = (p_s.argmax(1) == ys).astype(int); cor_e = (p_e.argmax(1) == ys).astype(int)
        # safety: commit walk only if p_walk >= tau_walk, else stop
        commit_walk = p_e[:, 1] >= tau_walk
        wrong_walk_base = float(np.mean((p_e.argmax(1) == 1) & (ys == 0)))   # argmax false-walk
        wrong_walk_safe = float(np.mean(commit_walk & (ys == 0)))
        walk_recall_safe = float(np.mean(commit_walk[ys == 1])) if (ys == 1).any() else np.nan
        per[s] = {
            "auroc_single": confidence_auroc(p_s.max(1), cor_s),
            "auroc_ens": confidence_auroc(p_e.max(1), cor_e),
            "ece_single": expected_calibration_error(p_s, ys),
            "ece_ens": expected_calibration_error(p_e, ys),
            "ece_ens_persession": expected_calibration_error(p_ps, ys[ei]),
            "exec_single": exec_acc(ys, p_s), "exec_ens": exec_acc(ys, p_e),
            "wrong_walk_base": wrong_walk_base, "wrong_walk_safe": wrong_walk_safe,
            "walk_recall_safe": walk_recall_safe,
        }
    mn = lambda f: float(np.nanmean([v[f] for v in per.values()]))
    keys = ["auroc_single", "auroc_ens", "ece_single", "ece_ens", "ece_ens_persession",
            "exec_single", "exec_ens", "wrong_walk_base", "wrong_walk_safe", "walk_recall_safe"]
    out = {k: mn(k) for k in keys}
    print(f"  [{sub}] AUROC {out['auroc_single']:.3f}->{out['auroc_ens']:.3f} | "
          f"ECE {out['ece_single']:.3f}->ens {out['ece_ens']:.3f}->persess {out['ece_ens_persession']:.3f} | "
          f"exec@80 {out['exec_single']:.3f}->{out['exec_ens']:.3f} | wrong-walk "
          f"{out['wrong_walk_base']:.3f}->{out['wrong_walk_safe']:.3f} (walk-recall {out['walk_recall_safe']:.3f})",
          flush=True)
    return {"subject": sub, "summary": out}


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (safety) ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "safety.json").write_text(json.dumps(res, indent=2))
    g = lambda f: float(np.nanmean([res[s]["summary"][f] for s in res]))
    print("\n============ SAFETY-METRIC IMPROVEMENTS (mean over 7 subjects) ============")
    print(f"confidence-AUROC:      single {g('auroc_single'):.3f}  ->  ensemble {g('auroc_ens'):.3f}")
    print(f"ECE:                   single {g('ece_single'):.3f}  ->  ensemble {g('ece_ens'):.3f}  ->  per-session {g('ece_ens_persession'):.3f}")
    print(f"executed acc @80% cov: single {g('exec_single'):.3f}  ->  ensemble {g('exec_ens'):.3f}")
    print(f"wrong-walk rate:       argmax {g('wrong_walk_base'):.3f}  ->  safety-threshold {g('wrong_walk_safe'):.3f}  (target {BETA})")
    print(f"   (walk-recall retained at the safety threshold: {g('walk_recall_safe'):.3f})")
