"""Published decoders under three amplitude normalisations (Table 4) and with the
amplitude side channel (Table 3). Cohort A, single seed.

    python exp_published.py                                                # published.json
    SCHEMES=global PUB_OUT=amp_ablation.json AMP=on python exp_published.py
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as FE
from common import (DEVICE, balanced_accuracy, grouped_split, imu_valid_mask,
                    normalise, set_seed)
from dataio import build_epochs, list_sessions

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = RESULTS / os.environ.get("PUB_OUT", "published.json")
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS = 3, 60
MODELS = os.environ.get("MODELS", "ShallowFBCSPNet,Deep4Net,EEGNeX,EEGITNet,BDTCN,"
                                  "EEGConformer,ATCNet").split(",")
SCHEMES = os.environ.get("SCHEMES", "perwindow,global,none").split(",")
AMP = os.environ.get("AMP", "off")                  # off | on | shuffle


def build(name, n_chan, n_time):
    if AMP != "off":
        from amp_channel import wrap
        return wrap(name, n_chan, n_time, 2, mode=AMP)
    import braindecode.models as B
    kw = {"n_chans": n_chan, "n_outputs": 2, "n_times": n_time, "sfreq": 100.0}
    if name == "FBLightConvNet":
        kw["win_len"] = 100
    return getattr(B, name)(**kw)


def load(sub, win, seed):
    es = build_epochs(subject=sub, win=win, step=0.5, zscore=False)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, ci = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=seed)
    f, c = (lambda a: a[tr][ti]), (lambda a: a[tr][ci])
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
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
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
                opt.zero_grad(set_to_none=True)
                continue
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
        raise RuntimeError("no finite validation output")
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(torch.as_tensor(Xt[i:i + 256]).to(DEVICE)).cpu().numpy()
                               for i in range(0, len(Xt), 256)]).argmax(1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vt = d["vt"]
    r2 = (FE.invariance_r2_conditional(pred[vt, None].astype(float), d["imu_t"][vt],
                                       d["yt"][vt], d["st"][vt])
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
    for name in MODELS:
        for scheme in SCHEMES:
            key = "w%s|%s|%s|s%d|amp-%s" % (win, name, scheme, seed, AMP)
            if key in out:
                continue
            t0, A, R = time.time(), [], []
            for sub, d in D.items():
                try:
                    a, r = train_eval(name, d, scheme, seed)
                except Exception as e:
                    print("    [fail] %s/%s/%s: %s" % (name, scheme, sub, e), flush=True)
                    continue
                A.append(a)
                R.append(r)
            if not A:
                continue
            out[key] = {"win": win, "model": name, "scheme": scheme, "seed": seed,
                        "acc": float(np.mean(A)), "acc_sd": float(np.std(A)),
                        "cond_r2": float(np.nanmean(R)), "n_sub": len(A)}
            print("  %-16s %-10s acc %.3f  (%.0fs)"
                  % (name, scheme, out[key]["acc"], time.time() - t0), flush=True)
            OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
