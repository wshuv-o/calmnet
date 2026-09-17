"""CALM-Net v3 -- full system evaluation with module ablation, two cohorts.

The system, in order:

    band decomposition
      -> ConditionalCovariance   Schur-complement conditioning on the measured
                                 motion waveform, per window, lambda learned per
                                 band. Removes the movement-explained part of
                                 the covariance -- the sufficient statistic --
                                 in closed form rather than penalising it in a
                                 loss.
      -> LearnableAlignment      generalised Euclidean Alignment, a and b per
                                 band, initialised at exact EA.
      -> BiMap / ReEig           SPD-manifold depth.
      -> CrossBandCoupling       bands interact on the manifold.
      -> tangent -> classifier + selective head
      -> temperature scaling
      -> Mondrian (class-conditional) conformal
      -> wrong-walk bounded abstention

Reported per arm: balanced accuracy, movement recoverability under the honest
within-distribution probe, and the safety quantities -- coverage, executed
accuracy, and P(commit walk | true stop) against its 0.05 target.

Ablations remove one module at a time so each has to earn its place, and every
arm is run on BOTH cohorts, because the previous composed architecture was best
on ds007788 and worst on MoBI.

Writes results/calmnet3.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from dataio import build_epochs, list_sessions
from motion_ts import build_motion_ts
from splits import grouped_split
from spdnet import ASPDNet, ASPDNetLarge, ASPDNetChol
from train import set_seed, DEVICE
from abstain import balanced_accuracy
from calibrate import fit_temperature, softmax_np
from calmnet_v2 import (mondrian_qhat, calibrate_walk_threshold, safety_report,
                        calibrate_theta)
from calmnet_msa import imu_valid_mask
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "calmnet3.json"
SEEDS = [0, 1, 2]
EPOCHS = 60
# The large arms hold 8 bands of 180x180 covariances plus the autograd graph
# through four BiMap/ReEig layers; at batch 256 that is ~265 MB per tensor and
# it OOMs. Batch is therefore chosen per arm.
# Measured on this card: the large model runs at 2.00 ms/sample at batch 48 and
# 0.43 ms/sample at batch 384 -- 4.7x faster for 3.1 GB of 16. The earlier OOM
# was three concurrent jobs, not the batch size. TF32 matmul is worth a further
# 2.2x and is off by default.
BATCH, INFER_BATCH = 512, 1024
LARGE_BATCH, LARGE_INFER = 384, 512
TARGET_COV = 0.8

# arms marked large=True use the high-capacity model: 8 bands, delay-embedded
# covariance (60->180 channels), SPD depth 4, band attention. 543,894 params
# against 22,069 -- capacity in the manifold representation, which is the one
# place in this project where depth was measured to help.
ARMS = {
    # Cheap arms FIRST. Each small cell is ~34 s; each large cell is ~26 min
    # (7 subjects x 226 s, and 95% of that is batched eigh). Running the large
    # arms first meant ~5 hours before any complete comparison existed.
    "full":         {},
    "no_cond":      {"cond_cov": False},
    "no_align":     {"learn_align": False},
    "no_couple":    {"couple": False},
    "no_select":    {"selective": False},
    "bare":         {"cond_cov": False, "learn_align": False,
                     "couple": False, "selective": False},
    "large_full":   {"large": True},
    "large_nodelay": {"large": True, "delay_order": 0},
    "large_nocond": {"large": True, "cond_cov": False},
    "large_noattn": {"large": True, "attend": False},
}


def save(out):
    """Merge-then-write, so two concurrent runs cannot clobber each other.

    Each process holds the whole results dict in memory and rewrites the file,
    so a plain write loses everything the other process added since this one
    loaded it. That is how a completed MoBI sweep was lost. Re-reading and
    merging before writing makes the update additive.
    """
    cur = {}
    if OUT.exists():
        try:
            cur = json.loads(OUT.read_text())
        except Exception:
            cur = {}
    cur.update(out)
    out.update(cur)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    tmp.replace(OUT)


def _t(a, dt=torch.float32):
    return torch.as_tensor(a, dtype=dt)


def selective_loss(logits, y, g, w, target=TARGET_COV, lam=32.0):
    ce = F.cross_entropy(logits, y, weight=w, reduction="none")
    cov = g.mean().clamp(min=1e-3)
    pen = lam * torch.clamp(torch.tensor(target, device=g.device) - cov, min=0) ** 2
    return (g * ce).mean() / cov + pen


def train(arm, Xf, Mf, yf, Xv, Mv, yv, seed=0, epochs=EPOCHS):
    set_seed(seed)
    kw = dict(ARMS[arm])
    # ASPDNetChol is the log-Cholesky chart of the same manifold. Measured on
    # REAL data it is 18.5x faster than the eigh version (13.9s vs 258s for 60
    # epochs): cuSOLVER's eigensolver is iterative and stalls on ill-conditioned
    # EEG covariances, while Cholesky is a direct factorisation. Synthetic
    # benchmarks hid this entirely -- torch.randn gives well-conditioned
    # matrices that eigh handles fine.
    Net = ASPDNetChol if kw.pop("large", False) else ASPDNet
    net = Net(n_chan=Xf.shape[1], n_ref=Mf.shape[1], **kw).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cnt = np.bincount(yf, minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32,
                     device=DEVICE)
    is_large = ARMS[arm].get("large", False)
    bs = LARGE_BATCH if is_large else BATCH
    # Cap the batch so at least 2 full batches exist, and keep the remainder.
    # With 856 fit windows and batch 384, drop_last=True yielded 2 batches and
    # silently discarded 88 samples -- 10% of the training data -- every epoch.
    # There is no BatchNorm in this model, so a short final batch is safe.
    bs = min(bs, max(32, len(yf) // 2))
    dl = DataLoader(TensorDataset(_t(Xf), _t(Mf), _t(yf, torch.long)),
                    batch_size=bs, shuffle=True, drop_last=False)
    best, best_sc, best_ep = None, -1, 0
    for ep in range(epochs):
        net.train()
        for xb, mb, yb in dl:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = net(xb, mb)
            if isinstance(out, tuple):
                lg, g = out
                loss = selective_loss(lg, yb, g, w)
            else:
                lg = out
                loss = F.cross_entropy(lg, yb, weight=w)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            ib = LARGE_INFER if is_large else INFER_BATCH
            outs = []
            for i in range(0, len(Xv), ib):
                o = net(_t(Xv[i:i + ib]).to(DEVICE), _t(Mv[i:i + ib]).to(DEVICE))
                outs.append((o[0] if isinstance(o, tuple) else o))
            lg = torch.cat(outs)
        if not torch.isfinite(lg).all():
            continue
        sc = balanced_accuracy(yv, lg.argmax(1).cpu().numpy())
        if sc > best_sc:
            best_sc, best_ep = sc, ep
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep - best_ep >= 20:
            break
    if best is not None:
        net.load_state_dict(best)
    return net


@torch.no_grad()
def infer(net, X, M, batch=None):
    net.eval()
    if batch is None:
        batch = LARGE_INFER if getattr(net, "tri", None) else INFER_BATCH
    L, G, Z = [], [], []
    for i in range(0, len(X), batch):
        xb, mb = _t(X[i:i + batch]).to(DEVICE), _t(M[i:i + batch]).to(DEVICE)
        o = net(xb, mb)
        if isinstance(o, tuple):
            L.append(o[0].cpu().numpy()); G.append(o[1].cpu().numpy())
        else:
            L.append(o.cpu().numpy()); G.append(np.ones(len(xb), np.float32))
        Z.append(net.features(xb, mb).cpu().numpy())
    return np.concatenate(L), np.concatenate(G), np.concatenate(Z)


def evaluate(arm, D, seed):
    accs, r2s, ww, cov, ex = [], [], [], [], []
    states = []
    for sub, d in D.items():
        net = train(arm, d["Xf"], d["Mf"], d["yf"], d["Xv"], d["Mv"], d["yv"], seed)
        lgv, gv, _ = infer(net, d["Xv"], d["Mv"])
        T = float(np.clip(fit_temperature(lgv, d["yv"]), 0.5, 5.0))
        pv = softmax_np(lgv, T)
        q = mondrian_qhat(pv, d["yv"], alpha=0.1)
        tau = calibrate_walk_threshold(pv, d["yv"], max_wrong_walk=0.05)
        # calibrate on the FULL rule, not on g alone, so every arm sits at the
        # same operating point and its safety numbers are comparable
        theta = calibrate_theta(pv, d["yv"], gv, q, tau, TARGET_COV)

        lg, g, Z = infer(net, d["Xt"], d["Mt"])
        p = softmax_np(lg, T)
        accs.append(balanced_accuracy(d["yt"], p.argmax(1)))
        r2s.append(FE.invariance_r2_cv(Z[d["vt"]], d["Sm"][d["vt"]], d["grp"][d["vt"]])
                   if d["vt"].sum() > 40 else np.nan)
        rep = safety_report(d["yt"], p, g, theta, q, tau)
        ww.append(rep["wrong_walk_rate"]); cov.append(rep["coverage"])
        ex.append(rep["executed_bal_acc"])
        states.append(net.module_state())
        del net, Z
        torch.cuda.empty_cache(); gc.collect()
    return {"acc": float(np.mean(accs)), "r2": float(np.nanmean(r2s)),
            "wrong_walk": float(np.nanmean(ww)), "coverage": float(np.nanmean(cov)),
            "exec_acc": float(np.nanmean(ex)), "state": states[0]}


def load_ds():
    D = {}
    for i in range(1, 8):
        sub = f"sub-0{i}"
        es = build_epochs(subject=sub)
        M, valid = build_motion_ts(sub)
        if len(M) != len(es):
            continue
        pres = sorted(set(int(v) for v in np.unique(es.session)))
        sess = [s for s in list_sessions(sub) if s in pres]
        tr = np.isin(es.session, sess[:3])
        ti, vi = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
        D[sub] = {"Xf": es.X[tr][ti], "Mf": M[tr][ti], "yf": es.y[tr][ti],
                  "Xv": es.X[tr][vi], "Mv": M[tr][vi], "yv": es.y[tr][vi],
                  "Xt": es.X[~tr], "Mt": M[~tr], "yt": es.y[~tr],
                  "Sm": es.imu_feats[~tr], "vt": valid[~tr], "grp": es.session[~tr]}
        del es
        gc.collect()
    return D


def load_mobi():
    from dataio_mobi import subjects, build_subject
    D = {}
    for sub in subjects():
        es = build_subject(sub)
        if es is None or es.motion_ts is None:
            continue
        fit, test = es.by_trials([1]), es.by_trials([2, 3])
        if len(np.unique(fit.y)) < 2 or len(np.unique(test.y)) < 2:
            continue
        ti, vi = grouped_split(fit.segment, fit.y, frac=0.3, seed=0)
        D[sub] = {"Xf": fit.X[ti], "Mf": fit.motion_ts[ti], "yf": fit.y[ti],
                  "Xv": fit.X[vi], "Mv": fit.motion_ts[vi], "yv": fit.y[vi],
                  "Xt": test.X, "Mt": test.motion_ts, "yt": test.y,
                  "Sm": test.motion, "vt": np.ones(len(test.y), bool),
                  "grp": test.trial}
        del es
        gc.collect()
    return D


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    which = sys.argv[1] if len(sys.argv) > 1 else "ds"
    D = load_ds() if which == "ds" else load_mobi()
    print(f"{which}: {len(D)} subjects\n", flush=True)

    for arm in ARMS:
        for sd in SEEDS:
            key = f"{which}|{arm}|s{sd}"
            if key in out and "acc" in out[key]:
                continue
            t0 = time.time()
            try:
                r = evaluate(arm, D, sd)
                r["secs"] = round(time.time() - t0, 1)
                out[key] = r
                save(out)
                print(f"  {key:22} acc {r['acc']:.3f}  R2 {r['r2']:+.3f}  "
                      f"wrongwalk {r['wrong_walk']:.3f}  cov {r['coverage']:.2f}  "
                      f"lam {r['state']['lam']}  {r['secs']:.0f}s", flush=True)
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"}
                save(out)
                print(f"  {key:22} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)

    print("\n" + "=" * 84)
    print(f"CALM-Net v3 -- {which}")
    print("=" * 84)
    print(f"{'arm':12}{'acc':>16}{'R2':>16}{'wrong-walk':>13}{'cov':>7}{'exec':>8}")
    for arm in ARMS:
        A = [v for k, v in out.items() if k.startswith(f"{which}|{arm}|") and "acc" in v]
        if not A:
            continue
        f = lambda m: (np.mean([a[m] for a in A]), np.std([a[m] for a in A]))
        a, sa = f("acc"); r, sr = f("r2"); w, _ = f("wrong_walk")
        c, _ = f("coverage"); e, _ = f("exec_acc")
        print(f"{arm:12}{a:9.3f}+/-{sa:5.3f}{r:+11.3f}+/-{sr:5.3f}"
              f"{w:13.3f}{c:7.2f}{e:8.3f}")
    print("\nwrong-walk target 0.05 (distribution-free bound from the calibration split)")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
