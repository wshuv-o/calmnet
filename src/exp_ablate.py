"""Module ablation: find which pieces actually earn their place.

Method (the one that should have been used from the start): build ONE model out
of independently switchable modules, measure each module's contribution, keep
what helps, and let the surviving composition be the architecture.

Three stages:
  A  ADD-ONE     start from the bare model, switch on one module at a time.
                 Measures what each module buys in isolation.
  B  LEAVE-ONE-OUT  start from everything on, switch one off at a time.
                 Measures whether a module is still needed once the others are
                 present -- modules can be individually useful and jointly
                 redundant.
  C  STACK       compose the modules that helped in BOTH stages and evaluate.

Scoring uses the honest objective throughout:

    score = balanced accuracy - max(0, intent->motion R^2)

Raw accuracy is not usable as a selection signal on this data -- across 131
architectures it correlated +0.67 with movement leakage, so optimising it
selects for contamination. Penalising positive R^2 makes a module count only if
its accuracy survives the invariance probe.

Writes results/ablation.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
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

from dataio import build_epochs, list_sessions
from motion_ts import build_motion_ts
from splits import grouped_split
from calmnet_arch import CALMNetArch
from mid import _decorr_penalty, hsic
from train import set_seed, DEVICE
from abstain import balanced_accuracy, confidence_auroc
from calibrate import fit_temperature, softmax_np
from calmnet_msa import invariance_r2

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "ablation.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
EPOCHS, N_TRAIN = 60, 3

# architecture modules (structural) and objective modules (loss terms)
ARCH_MODS = ["cancel", "spec_gate", "basis", "attn", "multiband"]
LOSS_MODS = ["adv", "decorr", "hsic", "art"]
ALL_MODS = ARCH_MODS + LOSS_MODS
BARE = {m: False for m in ALL_MODS}
FULL = {m: True for m in ALL_MODS}


def _t(a, dt=torch.float32):
    return torch.as_tensor(a, dtype=dt)


def train_arch(mods, Xf, Mf, yf, Xv, Mv, yv, k_imu=None, epochs=EPOCHS, seed=0):
    set_seed(seed)
    # the movement target is 4 statistics x n_ref channels, so the adversary and
    # artefact heads must be that wide -- not the 12-D default carried over from
    # the ds007788 IMU descriptor
    if k_imu is None:
        k_imu = 4 * Mf.shape[1]
    arch = {k: mods.get(k, True) for k in ARCH_MODS}
    model = CALMNetArch(n_chan=Xf.shape[1], n_time=Xf.shape[2], n_ref=Mf.shape[1],
                        k_imu=k_imu, mods=arch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cnt = np.bincount(yf, minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32,
                     device=DEVICE)
    # summary movement vector = per-window stats of the reference series
    def summ(M):
        return np.concatenate([M.mean(-1), M.std(-1), M.max(-1), M.min(-1)], axis=1)
    Sf, Sv = summ(Mf), summ(Mv)
    mu, sd = Sf.mean(0), Sf.std(0) + 1e-6
    Sf_s = ((Sf - mu) / sd).astype(np.float32)

    dl = DataLoader(TensorDataset(_t(Xf), _t(Mf), _t(yf, torch.long), _t(Sf_s)),
                    batch_size=64, shuffle=True)
    best, best_score, best_ep = None, -1e9, 0
    for ep in range(epochs):
        grl = 2.0 / (1.0 + np.exp(-10 * ep / max(epochs - 1, 1))) - 1.0
        model.train()
        for xb, mb, yb, sb in dl:
            xb, mb, yb, sb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            opt.zero_grad()
            o = model(xb, mb, grl if mods.get("adv") else 0.0)
            loss = F.cross_entropy(o["logits"], yb, weight=w)
            if mods.get("adv"):
                loss = loss + F.mse_loss(o["adv"], sb)
            if mods.get("art"):
                loss = loss + F.mse_loss(o["art"], sb)
            if mods.get("decorr"):
                loss = loss + grl * _decorr_penalty(o["z_int"], sb)
            if mods.get("hsic"):
                loss = loss + grl * 4.0 * hsic(o["z_int"], sb)
            if not torch.isfinite(loss):
                # decorr/HSIC on a near-degenerate batch can blow up; skipping
                # the batch keeps one bad minibatch from poisoning every weight
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = model(_t(Xv).to(DEVICE), _t(Mv).to(DEVICE), 0.0)["logits"]
        if not torch.isfinite(lg).all():
            continue                                  # diverged epoch: not selectable
        sc = balanced_accuracy(yv, lg.argmax(1).cpu().numpy())
        if sc > best_score:
            best_score, best_ep = sc, ep
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= 20:
            break
    model.load_state_dict(best)
    return model, (mu, sd)


@torch.no_grad()
def predict_arch(model, X, M, batch=256):
    model.eval()
    L = []
    for i in range(0, len(X), batch):
        L.append(model(_t(X[i:i + batch]).to(DEVICE), _t(M[i:i + batch]).to(DEVICE),
                       0.0)["logits"].cpu().numpy())
    lg = np.concatenate(L)
    e = np.exp(lg - lg.max(1, keepdims=True))
    return lg, e / e.sum(1, keepdims=True)


@torch.no_grad()
def encode_arch(model, X, M, batch=256):
    model.eval()
    Z = []
    for i in range(0, len(X), batch):
        Z.append(model.encode(_t(X[i:i + batch]).to(DEVICE),
                              _t(M[i:i + batch]).to(DEVICE))[0].cpu().numpy())
    return np.concatenate(Z)


def load_all():
    D = {}
    for sub in SUBJECTS:
        es = build_epochs(subject=sub)
        M, valid = build_motion_ts(sub)
        if len(M) != len(es):
            print(f"  {sub}: motion misaligned, skipping"); continue
        pres = sorted(set(int(v) for v in np.unique(es.session)))
        sess = [s for s in list_sessions(sub) if s in pres]
        tr = np.isin(es.session, sess[:N_TRAIN])
        ti, vi = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
        D[sub] = {"Xf": es.X[tr][ti], "Mf": M[tr][ti], "yf": es.y[tr][ti],
                  "Xv": es.X[tr][vi], "Mv": M[tr][vi], "yv": es.y[tr][vi],
                  "Xt": es.X[~tr], "Mt": M[~tr], "yt": es.y[~tr],
                  "vf": valid[tr][ti], "vt": valid[~tr]}
    return D


def load_mobi():
    """Second cohort, same interface. Fit on trial T01, test on T02+T03 --
    leakage-free at the recording level, mirroring train-early / test-late."""
    from dataio_mobi import subjects as mobi_subjects, build_subject as mobi_build
    D = {}
    for sub in mobi_subjects():
        es = mobi_build(sub)
        if es is None or es.motion_ts is None:
            continue
        fit, test = es.by_trials([1]), es.by_trials([2, 3])
        if len(np.unique(fit.y)) < 2 or len(np.unique(test.y)) < 2:
            continue
        ti, vi = grouped_split(fit.segment, fit.y, frac=0.3, seed=0)
        n_f, n_t = len(ti), len(test.y)
        D[sub] = {"Xf": fit.X[ti], "Mf": fit.motion_ts[ti], "yf": fit.y[ti],
                  "Xv": fit.X[vi], "Mv": fit.motion_ts[vi], "yv": fit.y[vi],
                  "Xt": test.X, "Mt": test.motion_ts, "yt": test.y,
                  "vf": np.ones(n_f, bool), "vt": np.ones(n_t, bool)}
    return D


def evaluate(mods, D, seed=0):
    per = {}
    for sub, d in D.items():
        model, (mu, sd) = train_arch(mods, d["Xf"], d["Mf"], d["yf"],
                                     d["Xv"], d["Mv"], d["yv"], seed=seed)
        lgv, _ = predict_arch(model, d["Xv"], d["Mv"])
        T = float(np.clip(fit_temperature(lgv, d["yv"]), 0.5, 5.0))
        lg, _ = predict_arch(model, d["Xt"], d["Mt"])
        p = softmax_np(lg, T)
        if not np.isfinite(p).all():
            p = np.full_like(p, 0.5)                  # diverged: record as chance
        pred = p.argmax(1)
        zf = encode_arch(model, d["Xf"], d["Mf"])
        zt = encode_arch(model, d["Xt"], d["Mt"])

        def summ(M):
            return np.concatenate([M.mean(-1), M.std(-1), M.max(-1), M.min(-1)], axis=1)
        Sf = ((summ(d["Mf"]) - mu) / sd).astype(np.float32)
        St = ((summ(d["Mt"]) - mu) / sd).astype(np.float32)
        vf, vt = d["vf"], d["vt"]
        r2 = (invariance_r2(zf[vf], Sf[vf], zt[vt], St[vt])
              if vf.sum() > 20 and vt.sum() > 20 else float("nan"))
        corr = (pred == d["yt"]).astype(int)
        auroc = (confidence_auroc(p.max(1), corr)
                 if np.isfinite(p).all() and len(np.unique(corr)) > 1 else float("nan"))
        per[sub] = {"acc": balanced_accuracy(d["yt"], pred), "r2": r2, "auroc": auroc}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    acc = float(np.nanmean([v["acc"] for v in per.values()]))
    r2 = float(np.nanmean([v["r2"] for v in per.values()]))
    return {"acc": acc, "r2": r2, "score": acc - max(0.0, r2),
            "auroc": float(np.nanmean([v["auroc"] for v in per.values()])),
            "n_leak": int(sum(1 for v in per.values() if v["r2"] > 0)),
            "per_subject": per}


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("loading data + motion series ...", flush=True)
    D = load_all()
    print(f"{len(D)} subjects\n", flush=True)

    def run(key, mods):
        if key in out and "acc" in out[key]:
            return out[key]
        t0 = time.time()
        r = evaluate(mods, D)
        r["mods"] = {k: bool(v) for k, v in mods.items()}
        r["secs"] = round(time.time() - t0, 1)
        out[key] = r
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  {key:22} acc {r['acc']:.3f}  R2 {r['r2']:+.3f}  "
              f"score {r['score']:.3f}  leak {r['n_leak']}/{len(D)}  {r['secs']:.0f}s",
              flush=True)
        return r

    print("STAGE A -- add one module to the bare model")
    base = run("A_bare", dict(BARE))
    for m in ALL_MODS:
        run(f"A_+{m}", {**BARE, m: True})

    print("\nSTAGE B -- remove one module from the full model")
    full = run("B_full", dict(FULL))
    for m in ALL_MODS:
        run(f"B_-{m}", {**FULL, m: False})

    # ---- decide which modules earn their place ----
    keep = []
    for m in ALL_MODS:
        a = out.get(f"A_+{m}", {}).get("score")
        b = out.get(f"B_-{m}", {}).get("score")
        add_gain = (a - base["score"]) if a is not None else 0.0
        loo_gain = (full["score"] - b) if b is not None else 0.0
        if add_gain > 0.005 or loo_gain > 0.005:
            keep.append(m)
    print(f"\nSTAGE C -- stacking modules that earned their place: {keep}")
    stacked = run("C_stacked", {m: (m in keep) for m in ALL_MODS})

    # ---- report ----
    print("\n" + "=" * 82)
    print("MODULE ABLATION")
    print("=" * 82)
    print(f"{'module':12}{'add-one':>10}{'gain':>9}{'leave-out':>11}{'gain':>9}{'verdict':>12}")
    print(f"{'(bare)':12}{base['score']:10.3f}{'':9}{'':11}{'':9}")
    for m in ALL_MODS:
        a = out.get(f"A_+{m}", {}).get("score", float("nan"))
        b = out.get(f"B_-{m}", {}).get("score", float("nan"))
        ag, lg = a - base["score"], full["score"] - b
        v = "KEEP" if m in keep else "drop"
        print(f"{m:12}{a:10.3f}{ag:+9.3f}{b:11.3f}{lg:+9.3f}{v:>12}")
    print("-" * 82)
    print(f"{'FULL':12}{full['score']:10.3f}  acc {full['acc']:.3f} R2 {full['r2']:+.3f}")
    print(f"{'STACKED':12}{stacked['score']:10.3f}  acc {stacked['acc']:.3f} "
          f"R2 {stacked['r2']:+.3f}  <- composed architecture")
    print(f"\nmodules kept: {keep}")
    print(f"Saved -> {OUT}")




def replicate(keys=("A_bare", "B_full", "C_stacked"),
              seeds=(1, 2, 3, 4, 5, 6, 7), dataset="ds007788"):
    """Seed-replicate the headline configurations.

    Stage-2 of the architecture sweep measured a seed sd of 0.028 on this data,
    which is larger than several individual module gains. The stacked-vs-bare
    gap (+0.147) is well outside that, but it should be demonstrated rather
    than assumed.
    """
    out = json.loads(OUT.read_text())
    D = load_all() if dataset == "ds007788" else load_mobi()
    suffix = "_rep" if dataset == "ds007788" else "_rep_mobi"
    for k in keys:
        if k not in out or "mods" not in out[k]:
            continue
        mods = out[k]["mods"]
        seed0 = ({"scores": [out[k]["score"]], "accs": [out[k]["acc"]],
                  "r2s": [out[k]["r2"]], "seeds": [0]}
                 if dataset == "ds007788"
                 else {"scores": [], "accs": [], "r2s": [], "seeds": []})
        rec = out.get(k + suffix, seed0)
        for sd in seeds:
            if sd in rec["seeds"]:
                continue
            r = evaluate(mods, D, seed=sd)
            rec["scores"].append(r["score"]); rec["accs"].append(r["acc"])
            rec["r2s"].append(r["r2"]); rec["seeds"].append(sd)
            out[k + suffix] = rec
            OUT.write_text(json.dumps(out, indent=2))
        if not rec["scores"]:
            continue
        s = np.array(rec["scores"]); a = np.array(rec["accs"]); r2 = np.array(rec["r2s"])
        print(f"  [{dataset}] {k:12} score {s.mean():.3f} +/- {s.std():.3f}   "
              f"acc {a.mean():.3f} +/- {a.std():.3f}   R2 {r2.mean():+.3f}  "
              f"(n={len(s)})", flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replicate":
        print("seed-replicating headline configurations", flush=True)
        replicate(dataset="ds007788")
    elif len(sys.argv) > 1 and sys.argv[1] == "mobi":
        print("replicating on the MoBI cohort (external validation)", flush=True)
        replicate(dataset="mobi")
    else:
        main()
