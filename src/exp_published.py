"""Published architectures, re-run with the preprocessing defect removed.

The 18-backbone benchmark earlier in this project ranked published models on
per-window z-scored input, using a selection metric that was anti-correlated
with accuracy. Both of those are now known to be wrong, which makes that
leaderboard uninformative rather than merely noisy.

Per-window z-scoring forces every channel of every window to unit variance.
Demonstrated on a synthetic ERD (one channel's power halved in one class):

    per-window   class0 1.000  class1 1.000   ratio 1.000   <- ERD destroyed
    global       class0 1.584  class1 0.392   ratio 0.248   <- preserved (0.5^2)

Walk versus Stop in motor cortex IS a power modulation, so the input every
published model was benchmarked on had the task signature removed from it. Band
power recovered +0.134 balanced accuracy when the normalisation was changed,
which is larger than the spread across all 131 architecture variants tried.

So the architecture question gets asked again, on input that still contains the
signal, with the honest (label-conditional) leakage probe:

  models    peer-reviewed implementations from braindecode, trained as plain
            2-class classifiers -- no MID wrapper, no invariance penalties, so
            what is measured is the architecture and nothing else.
  schemes   perwindow (what the project has always used) vs global (per-channel
            statistics fitted on the FIT split only, the scheme the reference
            implementation from the cohort's own lab uses).

Reports balanced accuracy and conditional R^2 per (model, scheme), so the
comparison a reviewer would ask for -- does any published architecture beat a
simple baseline once the input is correct -- is answerable either way.

Writes results/published.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from calmnet_msa import imu_valid_mask
from abstain import balanced_accuracy
from train import set_seed, DEVICE
from exp_globalnorm import normalise
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "published.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS = 3, 60
# EEGNet is excluded on this machine only: importing it pulls in h5py, whose
# DLL is blocked by a Windows Application Control policy ("DLL load failed
# while importing h5s"). Environment, not architecture -- restore it wherever
# h5py loads.
MODELS = ["ShallowFBCSPNet", "Deep4Net", "EEGNeX", "EEGITNet",
          "BDTCN", "EEGConformer", "ATCNet"]
SCHEMES = ["perwindow", "global"]


def build(name, n_chan, n_time):
    import braindecode.models as B
    cls = getattr(B, name)
    kw = {"n_chans": n_chan, "n_outputs": 2, "n_times": n_time, "sfreq": 100.0}
    if name == "FBLightConvNet":
        kw["win_len"] = 100
    return cls(**kw)


def load(sub, win, seed):
    es = build_epochs(subject=sub, win=win, step=0.5, zscore=False)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, ci = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=seed)
    f = lambda a: a[tr][ti]
    c = lambda a: a[tr][ci]
    return {"Xf": f(es.X), "yf": f(es.y), "Xc": c(es.X), "yc": c(es.y),
            "Xt": es.X[~tr], "yt": es.y[~tr], "st": es.session[~tr],
            "imu_t": es.imu_feats[~tr], "vt": valid[~tr]}


def train_eval(name, d, scheme, seed=0):
    Xf, Xc, Xt = normalise(scheme, d["Xf"], d["Xc"], d["Xt"])
    set_seed(seed)
    model = build(name, Xf.shape[1], Xf.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    cnt = np.bincount(d["yf"], minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)),
                     dtype=torch.float32, device=DEVICE)
    dl = DataLoader(TensorDataset(torch.as_tensor(Xf), torch.as_tensor(d["yf"])),
                    batch_size=64, shuffle=True)
    tXc = torch.as_tensor(Xc).to(DEVICE)
    best, best_sc, best_ep = None, -1e9, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = model(tXc)
        if not torch.isfinite(lg).all():
            continue
        sc = balanced_accuracy(d["yc"], lg.argmax(1).cpu().numpy())
        if sc > best_sc:
            best_sc, best_ep = sc, ep
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= 20:
            break
    if best is None:
        raise RuntimeError("never produced a finite validation logit")
    model.load_state_dict(best)
    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            P.append(model(torch.as_tensor(Xt[i:i + 256]).to(DEVICE)).cpu().numpy())
    pred = np.concatenate(P).argmax(1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vt = d["vt"]
    r2 = (FE.invariance_r2_conditional(pred[vt, None].astype(float),
                                       d["imu_t"][vt], d["yt"][vt], d["st"][vt])
          if vt.sum() > 40 else float("nan"))
    return balanced_accuracy(d["yt"], pred), r2


def main():
    win = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    seed = int(os.environ.get("SEED", "0"))
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    D = {}
    for sub in SUBJECTS:
        try:
            D[sub] = load(sub, win, seed)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
    print("%d subjects, win=%s, seed=%d\n" % (len(D), win, seed), flush=True)

    for name in MODELS:
        for scheme in SCHEMES:
            key = "w%s|%s|%s|s%d" % (win, name, scheme, seed)
            if key in out:
                continue
            t0, A, R = time.time(), [], []
            for sub, d in D.items():
                try:
                    a, r = train_eval(name, d, scheme, seed)
                except Exception as e:
                    print("    [fail] %s/%s/%s: %s: %s"
                          % (name, scheme, sub, type(e).__name__, str(e)[:90]),
                          flush=True)
                    continue
                A.append(a); R.append(r)
            if not A:
                continue
            out[key] = {"win": win, "model": name, "scheme": scheme, "seed": seed,
                        "acc": float(np.mean(A)), "acc_sd": float(np.std(A)),
                        "cond_r2": float(np.nanmean(R)), "n_sub": len(A)}
            print("  %-16s %-10s acc %.3f+-%.3f  cond_r2 %+.3f  (%.0fs)"
                  % (name, scheme, out[key]["acc"], out[key]["acc_sd"],
                     out[key]["cond_r2"], time.time() - t0), flush=True)
            OUT.write_text(json.dumps(out, indent=1))

    print("\n%-16s %10s %10s %10s" % ("model", "perwindow", "global", "delta"),
          flush=True)
    print("-" * 50, flush=True)
    for name in MODELS:
        a = out.get("w%s|%s|perwindow|s%d" % (win, name, seed), {}).get("acc")
        b = out.get("w%s|%s|global|s%d" % (win, name, seed), {}).get("acc")
        if a is not None and b is not None:
            print("%-16s %10.3f %10.3f %+10.3f" % (name, a, b, b - a), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
