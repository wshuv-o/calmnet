"""Does a learnable alignment layer beat fixed Euclidean Alignment?

This is the architecture claim, stated so it can fail.

  ASPD-Net initialises its alignment at exactly (a=1, b=1), which IS Euclidean
  Alignment. So the strongest known method is the network's starting point. If
  training moves the parameters away from (1,1) and accuracy or invariance
  improves, the learnable layer is doing something EA cannot. If it stays at
  (1,1), or moves and gets worse, the layer is decoration and should be dropped.

Four arms, identical everything else:

  aspd_learn    learnable alignment (the proposal)
  aspd_ea       alignment frozen at a=1, b=1  -> SPDNet on EA-aligned covariances
  aspd_noalign  alignment frozen at a=0       -> SPDNet, no alignment
  (reference)   tangent_EA + logistic regression, from the representation sweep

Comparing aspd_learn against aspd_ea isolates the learnable part; against
aspd_noalign isolates alignment itself; against the reference asks whether any
of the deep machinery earns its place over a linear model on tangent features.

Scored with the corrected within-distribution probe throughout.
Writes results/spdnet.json.
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

from dataio import build_epochs, list_sessions
from splits import grouped_split
from spdnet import ASPDNet, ea_equivalent
from train import set_seed, DEVICE
from abstain import balanced_accuracy
from calmnet_msa import imu_valid_mask
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "spdnet.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
SEEDS = [0, 1, 2]
EPOCHS = 60

ARMS = {
    "aspd_learn":   dict(learn_align=True,  freeze=None),
    "aspd_ea":      dict(learn_align=True,  freeze=(1.0, 1.0)),
    "aspd_noalign": dict(learn_align=True,  freeze=(0.0, 1.0)),
}


def _t(a, dt=torch.float32):
    return torch.as_tensor(a, dtype=dt)


def train_arm(cfg, Xf, yf, Xv, yv, seed=0, epochs=EPOCHS):
    set_seed(seed)
    model = ASPDNet(n_chan=Xf.shape[1], learn_align=cfg["learn_align"]).to(DEVICE)
    if cfg["freeze"] is not None:
        a, b = cfg["freeze"]
        with torch.no_grad():
            model.align.a.fill_(a)
            model.align.b_raw.fill_(float(np.log(np.expm1(b))))
        model.align.a.requires_grad_(False)
        model.align.b_raw.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cnt = np.bincount(yf, minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32,
                     device=DEVICE)
    dl = DataLoader(TensorDataset(_t(Xf), _t(yf, torch.long)), batch_size=64,
                    shuffle=True, drop_last=True)

    best, best_sc, best_ep = None, -1, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = model(_t(Xv).to(DEVICE))
        if not torch.isfinite(lg).all():
            continue
        sc = balanced_accuracy(yv, lg.argmax(1).cpu().numpy())
        if sc > best_sc:
            best_sc, best_ep = sc, ep
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= 20:
            break
    if best is not None:
        model.load_state_dict(best)
    return model


@torch.no_grad()
def encode(model, X, batch=128):
    model.eval()
    return np.concatenate([model.features(_t(X[i:i + batch]).to(DEVICE)).cpu().numpy()
                           for i in range(0, len(X), batch)])


@torch.no_grad()
def predict(model, X, batch=128):
    model.eval()
    return np.concatenate([model(_t(X[i:i + batch]).to(DEVICE)).argmax(1).cpu().numpy()
                           for i in range(0, len(X), batch)])


def subject(sub, seed):
    es = build_epochs(subject=sub)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:3])
    ti, vi = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=seed)
    d = {"Xf": es.X[tr][ti], "yf": es.y[tr][ti],
         "Xv": es.X[tr][vi], "yv": es.y[tr][vi],
         "Xt": es.X[~tr], "yt": es.y[~tr],
         "Mt": es.imu_feats[~tr], "vt": valid[~tr], "grp": es.session[~tr]}
    del es
    return d


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arm, cfg in ARMS.items():
        for sd in SEEDS:
            key = f"{arm}|seed{sd}"
            if key in out and "acc" in out[key]:
                continue
            t0, A, R, states = time.time(), [], [], []
            try:
                for sub in SUBJECTS:
                    d = subject(sub, sd)
                    m = train_arm(cfg, d["Xf"], d["yf"], d["Xv"], d["yv"], seed=sd)
                    A.append(balanced_accuracy(d["yt"], predict(m, d["Xt"])))
                    Z = encode(m, d["Xt"])
                    R.append(FE.invariance_r2_cv(Z[d["vt"]], d["Mt"][d["vt"]],
                                                 d["grp"][d["vt"]])
                             if d["vt"].sum() > 40 else np.nan)
                    states.append(m.alignment_state())
                    del m, d, Z
                    torch.cuda.empty_cache(); gc.collect()
                out[key] = {"acc": float(np.mean(A)), "r2": float(np.nanmean(R)),
                            "align": states[0], "ea_equiv": None,
                            "secs": round(time.time() - t0, 1)}
                OUT.write_text(json.dumps(out, indent=2))
                print(f"  {key:22} acc {out[key]['acc']:.3f}  R2 {out[key]['r2']:+.3f}  "
                      f"align {states[0]}  {out[key]['secs']:.0f}s", flush=True)
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"}
                OUT.write_text(json.dumps(out, indent=2))
                print(f"  {key:22} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)

    ok = {k: v for k, v in out.items() if "acc" in v}
    print("\n" + "=" * 76)
    print("ASPD-NET -- does a learnable alignment beat fixed Euclidean Alignment?")
    print("=" * 76)
    print(f"{'arm':16}{'acc':>18}{'R2':>18}")
    for arm in ARMS:
        A = [v["acc"] for k, v in ok.items() if k.startswith(arm + "|")]
        R = [v["r2"] for k, v in ok.items() if k.startswith(arm + "|")]
        if not A:
            continue
        print(f"{arm:16}{np.mean(A):9.3f} +/-{np.std(A):5.3f}"
              f"{np.mean(R):+11.3f} +/-{np.std(R):5.3f}   (n={len(A)})")
    print(f"{'tangent_EA+LR':16}{0.776:9.3f} +/-{0.010:5.3f}{-0.265:+11.3f}"
          "            (reference)")
    print()
    al = [v["align"] for v in ok.values() if v.get("align")]
    if al:
        print("learned alignment coefficients (a=1,b=1 would mean it stayed at EA):")
        for k, v in ok.items():
            if k.startswith("aspd_learn") and v.get("align"):
                print(f"  {k:22} {v['align']}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
