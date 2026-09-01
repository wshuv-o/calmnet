"""Multi-subject CALM-Net with double adversarial disentanglement.

Motivation (from results/extra_invariance.json): subjects whose intent code is
already movement-invariant (R^2 < 0) still gained large accuracy from more data
-- sub-02 0.516 -> 0.743 at R^2 -0.08, sub-05 0.657 -> 0.769 at R^2 -0.22. Those
subjects are sample-limited, not ceiling-limited. Each has only ~1.1k training
windows. Pooling all seven gives ~7.7k.

Pooling naively would let the decoder key on subject identity, so the intent code
is disentangled against TWO nuisances at once through separate gradient-reversal
heads:
    z_int  -/->  IMU vector   (movement, as in mid.py)
    z_int  -/->  subject id   (who is wearing the exoskeleton)
while per-subject FiLM parameters carry the residual idiosyncrasy that the shared
encoder is not allowed to represent.

Also fixes the zero-fill defect in dataio: sessions whose motion .tsv is absent
produce an all-zero IMU vector, which after standardisation becomes a constant
extreme value -- i.e. a session tag. Those windows are masked out of the movement
losses instead of being trained against.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from models import CALMNet
from mid import grad_reverse, _decorr_penalty
from train import set_seed, DEVICE
from abstain import balanced_accuracy
from dataio import build_epochs, list_sessions

SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3


# --------------------------------------------------------------------------- #
# Pooled data
# --------------------------------------------------------------------------- #
def imu_valid_mask(imu_feats, session):
    """False for windows whose IMU vector is identically zero for the WHOLE
    session -- that means the motion file was missing, not that the wearer was
    still. Per-window zeros inside an otherwise valid session are kept."""
    valid = np.ones(len(imu_feats), bool)
    for s in np.unique(session):
        m = session == s
        if not imu_feats[m].any():
            valid[m] = False
    return valid


def load_pooled(subjects=SUBJECTS, n_train=N_TRAIN):
    """Return pooled train arrays plus per-subject test sets and bookkeeping."""
    Xtr, ytr, Mtr, Str, Vtr, Gtr = [], [], [], [], [], []
    per_subject_test = {}
    seg_offset = 0
    for si, sub in enumerate(subjects):
        es = build_epochs(subject=sub)
        present = sorted(set(int(v) for v in np.unique(es.session)))
        sess = [s for s in list_sessions(sub) if s in present]
        train, test = sess[:n_train], sess[n_train:]
        valid = imu_valid_mask(es.imu_feats, es.session)

        tr = np.isin(es.session, train)
        Xtr.append(es.X[tr]); ytr.append(es.y[tr]); Mtr.append(es.imu_feats[tr])
        Str.append(np.full(int(tr.sum()), si)); Vtr.append(valid[tr])
        Gtr.append(es.segment[tr] + seg_offset)        # keep segment ids globally unique
        seg_offset += int(es.segment.max()) + 1

        per_subject_test[sub] = {
            "X": es.X[~tr], "y": es.y[~tr], "M": es.imu_feats[~tr],
            "session": es.session[~tr], "valid": valid[~tr], "subj": si,
            "train_sessions": train, "test_sessions": test,
        }
        print(f"  {sub}: train {int(tr.sum()):5d}  test {int((~tr).sum()):5d}  "
              f"IMU-invalid windows {int((~valid).sum()):5d}", flush=True)

    return {
        "X": np.concatenate(Xtr), "y": np.concatenate(ytr),
        "M": np.concatenate(Mtr), "subj": np.concatenate(Str),
        "valid": np.concatenate(Vtr), "segment": np.concatenate(Gtr),
        "test": per_subject_test, "subjects": list(subjects),
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MultiSubjectCALMNet(nn.Module):
    """CALMNet backbone + per-subject FiLM + intent/artefact split with two
    gradient-reversal adversaries (movement and subject identity)."""

    def __init__(self, n_chan=60, n_time=200, n_classes=2, k_imu=12,
                 n_subjects=7, p_drop=0.5):
        super().__init__()
        self.backbone = CALMNet(n_chan, n_time, n_classes=n_classes, p_drop=p_drop)
        d = self.backbone.feat_dim
        self.d_int = d // 2
        self.n_subjects = n_subjects
        # FiLM: per-subject affine on the ARTEFACT half only (see split())
        d_art = d - self.d_int
        self.film_gamma = nn.Embedding(n_subjects, d_art)
        self.film_beta = nn.Embedding(n_subjects, d_art)
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.weight)

        self.classify = nn.Linear(self.d_int, n_classes)
        self.art_head = nn.Sequential(nn.Linear(d - self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, k_imu))
        self.imu_adv = nn.Sequential(nn.Linear(self.d_int, 32), nn.ELU(),
                                     nn.Linear(32, 32), nn.ELU(),
                                     nn.Linear(32, k_imu))
        self.subj_adv = nn.Sequential(nn.Linear(self.d_int, 64), nn.ELU(),
                                      nn.Linear(64, 32), nn.ELU(),
                                      nn.Linear(32, n_subjects))
        # cooperative head: actively pulls subject identity into the artefact half
        self.subj_art = nn.Sequential(nn.Linear(d - self.d_int, 32), nn.ELU(),
                                      nn.Linear(32, n_subjects))

    def split(self, x, subj):
        """FiLM modulates ONLY the artefact half. Applying it to the whole vector
        injects subject identity into z_int, which the subject adversary is
        simultaneously trying to remove -- the two objectives cancel and training
        diverges (observed: loss 3.53 -> 4.73, best epoch 0). Here the artefact
        subspace absorbs subject idiosyncrasy and the intent subspace is scrubbed
        of it, so the two objectives point the same way."""
        z = self.backbone.features(x)
        z_int, z_art = z[:, :self.d_int], z[:, self.d_int:]
        z_art = z_art * self.film_gamma(subj) + self.film_beta(subj)
        return z_int, z_art

    def forward(self, x, subj, grl_imu=1.0, grl_subj=1.0):
        z_int, z_art = self.split(x, subj)
        return (self.classify(z_int),
                self.art_head(z_art),
                self.imu_adv(grad_reverse(z_int, grl_imu)),
                self.subj_adv(grad_reverse(z_int, grl_subj)),
                self.subj_art(z_art),          # pull subject identity INTO z_art
                z_int)

    @torch.no_grad()
    def encode(self, x, subj):
        return self.split(x, subj)


def _t(a, dtype=torch.float32):
    t = torch.as_tensor(a, dtype=dtype)
    return t.unsqueeze(1) if (dtype == torch.float32 and t.ndim == 3) else t


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_msa(data, fit_idx, val_idx, *, epochs=60, lr=1e-3, wd=1e-3, batch=128,
              lam_imu=1.0, lam_art=1.0, lam_subj=0.3, lam_dec=1.0, lam_sel=1.0,
              patience=25, clip=1.0, seed=0, verbose=True):
    """Train the pooled model. Movement losses are applied only to windows whose
    IMU vector is real; the subject adversary applies to every window."""
    set_seed(seed)
    X, y, M, S, V = data["X"], data["y"], data["M"], data["subj"], data["valid"]
    n_subj = len(data["subjects"])

    # standardise the IMU target using VALID fit windows only
    vf = V[fit_idx]
    mu = M[fit_idx][vf].mean(0)
    sd = M[fit_idx][vf].std(0) + 1e-6
    Mst = ((M - mu) / sd).astype(np.float32)

    model = MultiSubjectCALMNet(n_chan=X.shape[1], k_imu=M.shape[1],
                                n_subjects=n_subj).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    cnt = np.bincount(y[fit_idx], minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)),
                     dtype=torch.float32, device=DEVICE)

    ds = TensorDataset(_t(X[fit_idx]), _t(y[fit_idx], torch.long),
                       _t(Mst[fit_idx]), _t(S[fit_idx], torch.long),
                       _t(V[fit_idx].astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    best, best_score, best_ep = None, -1e9, 0
    for ep in range(epochs):
        p = ep / max(epochs - 1, 1)
        grl = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0          # DANN ramp
        model.train()
        last = 0.0
        for xb, yb, mb, sb, vb in dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            sb, vb = sb.to(DEVICE), vb.to(DEVICE)
            opt.zero_grad()
            logits, art, adv, sadv, sart, z_int = model(
                xb, sb, grl_imu=grl * lam_imu, grl_subj=grl * lam_subj)

            loss = F.cross_entropy(logits, yb, weight=w)
            loss = loss + lam_subj * F.cross_entropy(sadv, sb)   # adversary (via GRL)
            loss = loss + lam_subj * F.cross_entropy(sart, sb)   # cooperative, into z_art
            vm = vb.bool()
            if int(vm.sum()) > 1:
                loss = loss + lam_art * F.mse_loss(art[vm], mb[vm])
                loss = loss + F.mse_loss(adv[vm], mb[vm])
                loss = loss + lam_dec * grl * _decorr_penalty(z_int[vm], mb[vm])
            loss.backward()
            # two adversaries through a GRL make the gradient norm spiky; without
            # clipping the first run diverged (loss 3.53 -> 4.73, best epoch 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            last = float(loss.item())

        # Disentanglement-aware selection. Selecting on accuracy alone always
        # picks epoch 0, where the GRL ramp is ~0 and the model is a plain
        # leaky classifier (observed: best ep0, val bacc 0.689 -> 0.554 by ep10,
        # intent->IMU R2 +0.095). Penalising recoverable movement makes the
        # criterion prefer epochs that are actually invariant.
        pv = predict_msa(model, X[val_idx], S[val_idx])[1].argmax(1)
        yv, sv = y[val_idx], S[val_idx]
        accs = [balanced_accuracy(yv[sv == s], pv[sv == s])
                for s in range(n_subj) if (sv == s).any()]
        bacc = float(np.mean(accs))

        zv = encode_msa(model, X[val_idx], S[val_idx])
        vv = V[val_idx]
        if vv.sum() > 20:                       # probe on valid-IMU windows only
            h = np.arange(int(vv.sum())) % 10 < 7        # 70/30 probe split
            zq, mq = zv[vv], Mst[val_idx][vv]
            r2 = invariance_r2(zq[h], mq[h], zq[~h], mq[~h])
        else:
            r2 = 0.0
        score = bacc - lam_sel * max(0.0, r2)
        if score > best_score:
            best_score, best_ep = score, ep
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    ep{ep:3d}  loss {last:.3f}  bacc {bacc:.3f}  R2 {r2:+.3f}  score {score:.3f}",
                  flush=True)
        if ep - best_ep >= patience:
            print(f"    early stop ep{ep} (best ep{best_ep}, {best_score:.3f})", flush=True)
            break

    model.load_state_dict(best)
    return model, (mu, sd)


@torch.no_grad()
def predict_msa(model, X, subj, batch=256):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        lg = model(_t(X[i:i + batch]).to(DEVICE),
                   _t(subj[i:i + batch], torch.long).to(DEVICE),
                   grl_imu=0.0, grl_subj=0.0)[0]
        out.append(lg.cpu().numpy())
    lg = np.concatenate(out)
    e = np.exp(lg - lg.max(1, keepdims=True))
    return lg, e / e.sum(1, keepdims=True)


@torch.no_grad()
def encode_msa(model, X, subj, batch=256):
    model.eval()
    zi = []
    for i in range(0, len(X), batch):
        a, _ = model.encode(_t(X[i:i + batch]).to(DEVICE),
                            _t(subj[i:i + batch], torch.long).to(DEVICE))
        zi.append(a.cpu().numpy())
    return np.concatenate(zi)


def invariance_r2(z_fit, M_fit, z_te, M_te):
    """Linear recoverability of the IMU vector from the intent code.
    Negative => movement is not recoverable (what MID is supposed to achieve)."""
    est = Ridge(alpha=1.0).fit(z_fit, M_fit)
    return float(r2_score(M_te, est.predict(z_te), multioutput="variance_weighted"))


def subject_leakage(z_fit, s_fit, z_te, s_te):
    """Accuracy of recovering subject identity from the intent code.
    Chance = 1/n_subjects; higher means the pooled code is subject-tagged."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(z_fit, s_fit)
    return float((clf.predict(z_te) == s_te).mean())
