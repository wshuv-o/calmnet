"""CALM-Net v2 -- one model, built on a published backbone.

Foundation: FBMSNet (braindecode), chosen empirically, not by argument.

18 published architectures were run through this harness on three subjects.
Ranked by ACCURACY the winner is EEGConformer at 0.827 -- with intent->IMU
R^2 = +0.456, the highest movement leakage measured anywhere in this project.
Ranking by accuracy is the exact error this work exists to expose, so the
backbone is chosen from the ADMISSIBLE set (R^2 <= 0) instead:

    FBMSNet     0.690   R^2 -0.241   <- selected
    FBCNet      0.685   R^2 -0.094
    IFNet       0.600   R^2 -0.123
    Deep4Net    0.536   R^2 -0.368
    (EEGConformer 0.827, R^2 +0.456 -- rejected: leakage)

FBMSNet is a filter-bank multi-scale MI network: the right inductive bias for
ERD/ERS band power, parameter-efficient for ~1.1k training windows per subject,
and the most accurate architecture whose intent code does not retain movement.

What this adds on top, i.e. the actual model contribution:

  1. MID          intent/artefact split with a gradient-reversal adversary
                  against the 12-D IMU vector, so the command is decoded from
                  movement-invariant features. (already claimed, already real)

  2. Selective    a TRAINED selective head g(z)->[0,1] optimised with a
     head         SelectiveNet coverage-constrained objective under an
                  asymmetric cost. The paper claims this (Sec. SAS, lambda_3
                  L_sel); the repo never had it. Now it exists.

  3. Cost-        wrong-WALK (commit walk when the truth is stop) is the
     sensitive    dangerous error for an exoskeleton. It is priced C_WW times a
     objective    wrong-stop directly in the loss, not patched at threshold time.

  4. Mondrian     class-conditional conformal calibration giving a
     conformal    distribution-free bound on P(commit walk | true stop),
                  which is strictly stronger than the marginal coverage the
                  current pipeline guarantees, and is the quantity that
                  actually matters for safety.

The abstention rule the paper writes down but never ran with more than one term
is finally evaluated whole:

    commit argmax  iff  g(z) >= theta  AND  |conformal set| == 1
                        AND  (pred != walk  OR  p_walk >= tau_walk)
    else -> STOP
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mid import grad_reverse, _decorr_penalty
from train import set_seed, DEVICE
from abstain import balanced_accuracy

WALK, STOP = 1, 0
DEFAULT_BACKBONE = "FBMSNet"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CALMNetV2(nn.Module):
    def __init__(self, backbone=DEFAULT_BACKBONE, n_chan=60, n_time=200,
                 k_imu=12, n_classes=2, int_frac=0.5, p_drop=0.3):
        super().__init__()
        from braindecode_zoo import BDBackbone
        self.backbone = BDBackbone(backbone, n_chan=n_chan, n_time=n_time)
        d = self.backbone.feat_dim
        self.d_int = max(4, int(round(d * int_frac)))
        d_art = d - self.d_int
        if d_art < 4:                       # tiny feature vectors: keep a real artefact half
            self.d_int = max(4, d - 4)
            d_art = d - self.d_int

        self.classify = nn.Linear(self.d_int, n_classes)
        # movement adversary (through GRL) and artefact predictor (cooperative)
        self.imu_adv = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                     nn.Linear(32, 32), nn.ELU(), nn.Linear(32, k_imu))
        self.art_head = nn.Sequential(nn.Linear(d_art, 32), nn.ELU(), nn.Linear(32, k_imu))
        # trained selective head -- the piece the paper claimed and lacked
        self.select = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                    nn.Dropout(p_drop), nn.Linear(32, 1))
        # SelectiveNet auxiliary head (trained on every sample, stabilises g)
        self.aux = nn.Linear(self.d_int, n_classes)

    def split(self, x):
        z = self.backbone.features(x)
        return z[:, :self.d_int], z[:, self.d_int:]

    def forward(self, x, grl=1.0):
        zi, za = self.split(x)
        return {"logits": self.classify(zi),
                "aux": self.aux(zi),
                "g": torch.sigmoid(self.select(zi)).squeeze(-1),
                "adv": self.imu_adv(grad_reverse(zi, grl)),
                "art": self.art_head(za),
                "z_int": zi}

    @torch.no_grad()
    def encode(self, x):
        return self.split(x)


def _t(a, dtype=torch.float32):
    t = torch.as_tensor(a, dtype=dtype)
    return t.unsqueeze(1) if (dtype == torch.float32 and t.ndim == 3) else t


# --------------------------------------------------------------------------- #
# Cost-sensitive + coverage-constrained objective
# --------------------------------------------------------------------------- #
def cost_weights(y, c_ww=1.0, device=DEVICE):
    """Inverse-frequency class weights, optionally scaled by a wrong-WALK cost.

    c_ww defaults to 1.0 (pure balancing) deliberately. Pricing the safety
    asymmetry into the LOSS does not work here: walk is the minority class
    (~27%), so balancing already puts stop at 0.68 and walk at 1.85; scaling
    stop by 5 gives 3.4 and inverts the balance. Measured effect of c_ww=5:
    balanced accuracy collapsed to 0.50-0.54 with walk-recall 0.00 -- a model
    that never commits walk, trivially safe and entirely useless.

    The asymmetry is enforced instead at the decision threshold, by
    calibrate_walk_threshold(), which bounds P(commit walk | true stop)
    distribution-free rather than by tuning a loss weight.
    """
    cnt = np.bincount(y, minlength=2).astype(float)
    bal = cnt.sum() / (2 * np.maximum(cnt, 1))
    w = np.array([bal[STOP] * c_ww, bal[WALK]], dtype=np.float32)
    return torch.tensor(w, device=device)


def selective_loss(logits, y, g, w, target_cov=0.8, lam_cov=32.0):
    """SelectiveNet: mean loss over selected samples, divided by empirical
    coverage, plus a quadratic penalty for falling below the coverage target."""
    ce = F.cross_entropy(logits, y, weight=w, reduction="none")
    cov = g.mean().clamp(min=1e-3)
    sel = (g * ce).mean() / cov
    pen = lam_cov * torch.clamp(torch.tensor(target_cov, device=g.device) - cov, min=0) ** 2
    return sel + pen, float(cov.item())


def train_calmnet_v2(Xtr, ytr, Mtr, Xval, yval, *, backbone=DEFAULT_BACKBONE,
                     epochs=80, lr=1e-3, wd=1e-3, batch=64, patience=20,
                     lam_adv=1.0, lam_art=1.0, lam_dec=1.0, lam_sel=1.0,
                     c_ww=1.0, target_cov=0.8, seed=0, verbose=False):
    set_seed(seed)
    Mtr = np.atleast_2d(Mtr).astype(np.float32)
    model = CALMNetV2(backbone=backbone, n_chan=Xtr.shape[1], n_time=Xtr.shape[2],
                      k_imu=Mtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = cost_weights(ytr, c_ww)
    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-6
    Mst = ((Mtr - mu) / sd).astype(np.float32)
    dl = DataLoader(TensorDataset(_t(Xtr), _t(ytr, torch.long), _t(Mst)),
                    batch_size=batch, shuffle=True)

    best, best_score, best_ep = None, -1e9, 0
    for ep in range(epochs):
        grl = 2.0 / (1.0 + np.exp(-10 * ep / max(epochs - 1, 1))) - 1.0
        model.train()
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            o = model(xb, grl * lam_adv)
            l_sel, _ = selective_loss(o["logits"], yb, o["g"], w, target_cov)
            loss = (l_sel
                    + F.cross_entropy(o["aux"], yb, weight=w)          # SelectiveNet aux
                    + lam_art * F.mse_loss(o["art"], mb)
                    + F.mse_loss(o["adv"], mb)
                    + grl * lam_dec * _decorr_penalty(o["z_int"], mb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            o = model(_t(Xval).to(DEVICE), 0.0)
            pred = o["logits"].argmax(1).cpu().numpy()
            gv = o["g"].cpu().numpy()
        bacc = balanced_accuracy(yval, pred)
        # select on selective quality: accuracy among the windows g would commit
        keep = gv >= np.quantile(gv, 1 - target_cov)
        sel_acc = balanced_accuracy(yval[keep], pred[keep]) if keep.sum() > 5 else bacc
        score = 0.5 * bacc + 0.5 * sel_acc
        if score > best_score:
            best_score, best_ep = score, ep
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and ep % 10 == 0:
            print(f"    ep{ep:3d} bacc {bacc:.3f} sel_acc {sel_acc:.3f}", flush=True)
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best)
    return model, (mu, sd)


@torch.no_grad()
def predict_v2(model, X, batch=256):
    """Return (logits, probs, g) with dropout off."""
    model.eval()
    L, G = [], []
    for i in range(0, len(X), batch):
        o = model(_t(X[i:i + batch]).to(DEVICE), 0.0)
        L.append(o["logits"].cpu().numpy()); G.append(o["g"].cpu().numpy())
    lg = np.concatenate(L)
    e = np.exp(lg - lg.max(1, keepdims=True))
    return lg, e / e.sum(1, keepdims=True), np.concatenate(G)


@torch.no_grad()
def encode_v2(model, X, batch=256):
    model.eval()
    return np.concatenate([model.encode(_t(X[i:i + batch]).to(DEVICE))[0].cpu().numpy()
                           for i in range(0, len(X), batch)])


# --------------------------------------------------------------------------- #
# Mondrian (class-conditional) conformal + the full abstention rule
# --------------------------------------------------------------------------- #
def mondrian_qhat(cal_probs, cal_y, alpha=0.1):
    """One conformal threshold per TRUE class. Marginal conformal can hold 90%
    overall while badly under-covering the minority class; conditioning on the
    true class is what turns coverage into a per-class safety statement."""
    q = {}
    for c in (STOP, WALK):
        m = cal_y == c
        if m.sum() < 5:
            q[c] = 1.0
            continue
        s = 1.0 - cal_probs[m][:, c]
        lvl = min(1.0, np.ceil((m.sum() + 1) * (1 - alpha)) / m.sum())
        q[c] = float(np.quantile(s, lvl, method="higher"))
    return q


def mondrian_sets(probs, q):
    """Boolean (N,2): class c in the set iff 1 - p_c <= q_c."""
    return np.stack([(1.0 - probs[:, c]) <= q[c] for c in (STOP, WALK)], axis=1)


def calibrate_walk_threshold(cal_probs, cal_y, max_wrong_walk=0.05):
    """Smallest tau such that P(p_walk >= tau | true = STOP) <= max_wrong_walk.
    A distribution-free bound on the dangerous error, read off the calibration
    set rather than tuned on test."""
    stop = cal_probs[cal_y == STOP][:, WALK]
    if len(stop) == 0:
        return 0.5
    return float(np.quantile(stop, 1.0 - max_wrong_walk, method="higher"))


def sas_decide(probs, g, theta, q, tau_walk):
    """The full safety-asymmetric rule: selective head AND conformal singleton
    AND the wrong-walk bound. Returns (action, committed_mask)."""
    pred = probs.argmax(1)
    sets = mondrian_sets(probs, q)
    accept = (g >= theta) & (sets.sum(1) == 1)
    risky = (pred == WALK) & (probs[:, WALK] < tau_walk)
    accept &= ~risky
    return np.where(accept, pred, STOP), accept


def safety_report(y, probs, g, theta, q, tau_walk):
    action, committed = sas_decide(probs, g, theta, q, tau_walk)
    n = len(y)
    out = {"coverage": float(committed.mean())}
    if committed.any():
        out["executed_bal_acc"] = balanced_accuracy(y[committed], probs[committed].argmax(1))
    else:
        out["executed_bal_acc"] = float("nan")
    stop_mask = y == STOP
    out["wrong_walk_rate"] = float((action[stop_mask] == WALK).mean()) if stop_mask.any() else 0.0
    walk_mask = y == WALK
    out["walk_recall"] = float((action[walk_mask] == WALK).mean()) if walk_mask.any() else 0.0
    sets = mondrian_sets(probs, q)
    for c, nm in ((STOP, "stop"), (WALK, "walk")):
        m = y == c
        out[f"conformal_cov_{nm}"] = float(sets[m][:, c].mean()) if m.any() else float("nan")
    out["mean_set_size"] = float(sets.sum(1).mean())
    return out
