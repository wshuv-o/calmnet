"""The lever never pulled: EEG + kinematics FUSION.

Every decoder so far (ours and every baseline) is EEG-only, yet the exoskeleton carries
IMUs and this dataset is explicitly multimodal. For a real controller, fusing them is
the correct engineering, not cheating -- provided we report EEG-only and IMU-only
alongside, which we do.

  EEG branch (CALMNet features) + kinematic branch (MLP on the 12-D IMU descriptor)
  -> concat -> classifier

Reference (full data, same test set): EEG-only 0.835 | IMU-only 0.839
Question: does fusion beat BOTH (i.e. are the modalities complementary)?
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys
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
from models import CALMNet
from train import set_seed, DEVICE, class_weights
from abstain import balanced_accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRIALS = tuple(f"trial{i:02d}" for i in range(1, 13))
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, EPOCHS = 3, 110


class FusionNet(nn.Module):
    """EEG features + kinematic features -> joint decision."""
    def __init__(self, n_chan=60, n_time=200, k_imu=12, n_classes=2, p_drop=0.5, d_imu=32):
        super().__init__()
        self.eeg = CALMNet(n_chan, n_time, n_classes=n_classes, p_drop=p_drop)
        self.imu = nn.Sequential(nn.Linear(k_imu, d_imu), nn.LayerNorm(d_imu), nn.ELU(),
                                 nn.Dropout(p_drop), nn.Linear(d_imu, d_imu), nn.ELU())
        d = self.eeg.feat_dim + d_imu
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p_drop), nn.Linear(d, n_classes))

    def forward(self, x, m):
        return self.head(torch.cat([self.eeg.features(x), self.imu(m)], dim=1))


def _t(a):
    t = torch.as_tensor(a, dtype=torch.float32)
    return t.unsqueeze(1) if t.ndim == 3 else t


def train_fusion(X, y, M, Xv, yv, Mv, epochs=EPOCHS, lr=1e-3, wd=1e-3, batch=64, seed=0):
    set_seed(seed)
    mu, sd = M.mean(0), M.std(0) + 1e-6
    Ms, Mvs = (M - mu) / sd, (Mv - mu) / sd
    model = FusionNet(n_chan=X.shape[1], k_imu=M.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = class_weights(y)
    dl = DataLoader(TensorDataset(_t(X), torch.as_tensor(y), _t(Ms)), batch_size=batch, shuffle=True)
    best, best_s = None, -1
    for ep in range(epochs):
        model.train()
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, mb), yb, weight=w)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(_t(Xv).to(DEVICE), _t(Mvs).to(DEVICE)).argmax(1).cpu().numpy()
        s = balanced_accuracy(yv, p)
        if s > best_s:
            best_s, best = s, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    model._mu, model._sd = mu, sd
    return model


@torch.no_grad()
def predict_fusion(model, X, M, batch=256):
    model.eval(); Ms = (M - model._mu) / model._sd; out = []
    for i in range(0, len(X), batch):
        out.append(model(_t(X[i:i+batch]).to(DEVICE), _t(Ms[i:i+batch]).to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def run_subject(sub):
    es = build_epochs(subject=sub)
    sess = [s for s in list_sessions(sub) if s in set(int(x) for x in np.unique(es.session))]
    tr = es.by_sessions(sess[:N_TRAIN]); te = es.by_sessions(sess[N_TRAIN:])
    ti, vi = grouped_split(tr.segment, tr.y, frac=0.2, seed=0)
    ex = build_epochs(subject=sub, sessions=sess[:N_TRAIN], tasks=TRIALS)
    X = np.concatenate([tr.X[ti], ex.X]); y = np.concatenate([tr.y[ti], ex.y])
    M = np.concatenate([tr.imu_feats[ti], ex.imu_feats])

    m = train_fusion(X, y, M, tr.X[vi], tr.y[vi], tr.imu_feats[vi], seed=0)
    fus = balanced_accuracy(te.y, predict_fusion(m, te.X, te.imu_feats))
    # IMU-only reference on identical splits
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(M, y)
    imu = balanced_accuracy(te.y, clf.predict(te.imu_feats))
    print(f"  [{sub}] FUSION {fus:.3f} | IMU-only {imu:.3f}", flush=True)
    return {"subject": sub, "fusion": fus, "imu_only": imu}


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (EEG+IMU fusion) ########", flush=True)
        res[sub] = run_subject(sub)
        (RESULTS / "fusion.json").write_text(json.dumps(res, indent=2))
    g = lambda f: float(np.nanmean([res[s][f] for s in res]))
    print("\n============ MULTIMODAL FUSION (mean over 7) ============")
    print(f"  EEG-only (best arch, full data) : 0.836")
    print(f"  IMU-only (this run)             : {g('imu_only'):.3f}")
    print(f"  EEG + IMU FUSION                : {g('fusion'):.3f}   <- must beat BOTH")
    print(f"\n  fusion vs EEG-only: {g('fusion') - 0.836:+.3f} | vs IMU-only: {g('fusion') - g('imu_only'):+.3f}")
