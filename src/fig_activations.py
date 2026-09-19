"""What the model actually computes, tapped from a trained network.

The architecture figure is a schematic. This one is measured: a model is trained
on one participant of cohort A with the reported settings, then forward hooks
record the real activations at the four points the schematic names, on held-out
windows the model never saw.

  tangent map        the 1830-dimensional tangent vector, shown folded back
                     into the 60x60 symmetric matrix it came from
  project            the branch's 128-dimensional output
  frame embedding    the stem's 128-dimensional output
  fuse               the 128-dimensional fused representation

Nothing here is simulated. Weights come from training, activations from
held-out test windows, and the class labels are the recorded exoskeleton state.

    python src/fig_activations.py

Writes results/fig_activations.pdf and .png.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
try:
    import fontTools.varLib  # noqa: F401
    matplotlib.rcParams["pdf.fonttype"] = 42
except Exception:
    matplotlib.rcParams["pdf.fonttype"] = 3
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as S                                            # noqa: E402
from figstyle import FULL, OURS, VERM, GREY                     # noqa: E402
from exp_calmnetx import load, make_split, normalise, EPOCHS    # noqa: E402
from abstain import balanced_accuracy                           # noqa: E402
from driftnet import build_driftnet                             # noqa: E402
from train import set_seed, DEVICE                              # noqa: E402
from atcnet_plus import selective_loss                          # noqa: E402

S.use()
SUB = os.environ.get("FA_SUB", "sub-01")


def train_one(d, seed=0):
    """Train the reported configuration on one participant."""
    Xf, Xc, Xt = normalise("global", d["Xf"], d["Xc"], d["Xt"])
    tXf, tyf, ixf, _ = make_split(Xf, d["yf"], d["sf"], d["tf"], d["gf"], 1)
    tXc, tyc, ixc, _ = make_split(Xc, d["yc"], d["sc"], d["tc"], d["gc"], 1)
    tXt, tyt, ixt, _ = make_split(Xt, d["yt"], d["st"], d["tt"], d["gt"], 1)

    set_seed(seed)
    model = build_driftnet(Xf.shape[1], Xf.shape[2], 2, use_align=False,
                           use_ctx=False, use_gate=False,
                           use_tangent=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=EPOCHS * max(1, len(ixf) // 32 + 1))
    cnt = np.bincount(d["yf"], minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32,
                     device=DEVICE)
    dl = DataLoader(TensorDataset(ixf, tyf), batch_size=32, shuffle=True)

    best, best_sc = None, -1e9
    for ep in range(EPOCHS):
        model.train()
        for bi, by in dl:
            xb = tXf[bi.reshape(-1)].to(DEVICE).reshape(
                bi.shape[0], bi.shape[1], tXf.shape[1], tXf.shape[2])
            opt.zero_grad()
            loss, _ = selective_loss(model(xb), by.to(DEVICE), weight=w)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(ixc), 64):
                b = ixc[i:i + 64]
                xb = tXc[b.reshape(-1)].to(DEVICE).reshape(
                    b.shape[0], b.shape[1], tXc.shape[1], tXc.shape[2])
                lg.append(model(xb)["logits"].cpu().numpy())
            sc = balanced_accuracy(d["yc"], np.concatenate(lg).argmax(1))
        if sc > best_sc:
            best_sc = sc
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model, tXt, ixt, d["yt"]


@torch.no_grad()
def tap(model, tXt, ixt, batch=64):
    """Forward hooks at the four points the schematic names."""
    store = {}
    hs = []
    hs.append(model.tangent.proj.register_forward_pre_hook(
        lambda m, i: store.setdefault("tangent", []).append(
            i[0].detach().cpu().numpy())))
    hs.append(model.tangent.proj.register_forward_hook(
        lambda m, i, o: store.setdefault("project", []).append(
            o.detach().cpu().numpy())))
    hs.append(model.fuse.register_forward_pre_hook(
        lambda m, i: store.setdefault("concat", []).append(
            i[0].detach().cpu().numpy())))
    hs.append(model.fuse.register_forward_hook(
        lambda m, i, o: store.setdefault("fuse", []).append(
            o.detach().cpu().numpy())))
    model.eval()
    pred = []
    for i in range(0, len(ixt), batch):
        b = ixt[i:i + batch]
        xb = tXt[b.reshape(-1)].to(DEVICE).reshape(
            b.shape[0], b.shape[1], tXt.shape[1], tXt.shape[2])
        pred.append(model(xb)["logits"].cpu().numpy())
    for h in hs:
        h.remove()
    out = {k: np.concatenate(v) for k, v in store.items()}
    out["stem"] = out["concat"][:, :128]
    return out, np.concatenate(pred)


def unvec(v, C=60):
    """Fold a tangent vector back into the symmetric matrix it came from."""
    iu = np.triu_indices(C)
    M = np.zeros((C, C))
    w = np.where(iu[0] == iu[1], 1.0, 1.0 / np.sqrt(2.0))
    M[iu] = v * w
    return M + M.T - np.diag(np.diag(M))


def main():
    t0 = time.time()
    d = load(SUB, 0)
    model, tXt, ixt, y = train_one(d)
    A, logits = tap(model, tXt, ixt)
    acc = balanced_accuracy(y, logits.argmax(1))
    print("  %s trained, held-out balanced accuracy %.3f (%.0fs)"
          % (SUB, acc, time.time() - t0))

    stop, walk = y == 0, y == 1
    T = A["tangent"]
    Ms, Mw = unvec(T[stop].mean(0)), unvec(T[walk].mean(0))
    D = Mw - Ms

    fig = plt.figure(figsize=(FULL, 3.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.55,
                          wspace=0.46)

    # ---- (a,b,c) the tangent map, folded back to 60 x 60
    for k, (M, ttl) in enumerate(((Ms, "Stop"), (Mw, "Walk"),
                                  (D, "Walk $-$ Stop"))):
        ax = fig.add_subplot(gs[0, k])
        lim = np.abs(D).max() if k == 2 else max(np.abs(Ms).max(),
                                                 np.abs(Mw).max())
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest")
        ax.set_title(ttl, fontsize=7.4, pad=4)
        ax.set_xticks([]), ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5.4)
        S.panel(ax, "abc"[k], x=-0.10, y=1.20)
    fig.text(0.055, 0.540, "tangent map, folded to $60 \\times 60$ "
             "(mean over held-out windows)", fontsize=6.6, color=GREY)

    # ---- (d) separability through the four taps
    ax = fig.add_subplot(gs[0, 3])
    names = ["tangent", "project", "stem", "fuse"]
    lbl = ["tangent\nmap", "project", "frame\nembed", "fuse"]
    aucs = []
    for n in names:
        Z = A[n]
        m = (Z[walk].mean(0) - Z[stop].mean(0))
        s = np.sqrt(0.5 * (Z[walk].var(0) + Z[stop].var(0))) + 1e-9
        aucs.append(float(np.abs(m / s).mean()))
    ax.bar(range(4), aucs, color=[GREY, OURS, GREY, OURS], width=0.62)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lbl, fontsize=6.0)
    # label on the right, clear of the neighbouring colour bar
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("mean $|d|$ per unit", fontsize=6.8)
    ax.tick_params(axis="y", labelsize=6.0)
    for s_ in ("top", "left"):
        ax.spines[s_].set_visible(False)
    S.panel(ax, "d", x=-0.16, y=1.20)

    # ---- (e,f,g,h) each tap in two dimensions
    from sklearn.decomposition import PCA
    for k, (n, ttl) in enumerate(zip(names, lbl)):
        ax = fig.add_subplot(gs[1, k])
        Z = PCA(n_components=2, random_state=0).fit_transform(A[n])
        ax.scatter(Z[stop, 0], Z[stop, 1], s=4, color=GREY, alpha=0.55,
                   linewidths=0, label="Stop")
        ax.scatter(Z[walk, 0], Z[walk, 1], s=4, color=VERM, alpha=0.55,
                   linewidths=0, label="Walk")
        ax.set_title(ttl.replace("\n", " "), fontsize=7.0, pad=3)
        ax.set_xticks([]), ax.set_yticks([])
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        S.panel(ax, "efgh"[k], x=-0.10, y=1.16)
        if k == 0:
            ax.legend(fontsize=5.8, frameon=False, loc="upper left",
                      handletextpad=0.2, borderpad=0.1)

    S.save(fig, "fig_activations")
    for n, a in zip(names, aucs):
        print("    %-10s dim %-5d mean |d| %.3f" % (n, A[n].shape[1], a))


if __name__ == "__main__":
    main()
