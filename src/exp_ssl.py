"""IDEA 10 (modern) - Self-supervised contrastive pretraining (SimCLR/BENDR-style, the
paradigm behind 2021-2024 EEG foundation models). Pretrain an encoder with NT-Xent on
unlabeled windows + augmentations, then linear-probe walk/stop. Same invariance test.
"""
import os, sys, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from dataio import build_epochs
from train import set_seed
from abstain import balanced_accuracy
DEVICE = torch.device("cpu")   # GPU thermals inadequate on this machine (spikes to 86C); run SSL on CPU
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUBJECTS = sys.argv[1:] or [f"sub-0{i}" for i in range(1, 8)]


class Encoder(nn.Module):
    def __init__(self, C, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (1, 25), padding=(0, 12)), nn.BatchNorm2d(16), nn.ELU(),
            nn.Conv2d(16, 32, (C, 1)), nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.3),
            nn.Conv2d(32, 64, (1, 11), padding=(0, 5)), nn.BatchNorm2d(64), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(64, emb)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class Proj(nn.Module):
    def __init__(self, emb=128, out=64):
        super().__init__()
        self.p = nn.Sequential(nn.Linear(emb, emb), nn.ELU(), nn.Linear(emb, out))

    def forward(self, x):
        return F.normalize(self.p(x), dim=1)


def augment(x):
    B, _, C, T = x.shape
    x = torch.roll(x, shifts=int(torch.randint(-20, 20, (1,))), dims=3)   # time shift
    mask = (torch.rand(B, 1, C, 1, device=x.device) > 0.2).float()        # channel dropout 20%
    x = x * mask
    x = x * (0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=x.device))         # amplitude scale
    x = x + 0.1 * torch.randn_like(x)                                     # noise
    return x


def nt_xent(z1, z2, temp=0.2):
    N = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


import time
def pretrain(X, C, epochs=30, bs=96, seed=0):
    set_seed(seed)
    enc, proj = Encoder(C).to(DEVICE), Proj().to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
    Xt = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        enc.train(); proj.train()
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i + bs]].to(DEVICE)
            if len(xb) < 8:
                continue
            opt.zero_grad()
            z1, z2 = proj(enc(augment(xb))), proj(enc(augment(xb)))
            loss = nt_xent(z1, z2)
            loss.backward(); opt.step()
        time.sleep(0.2)                      # per-epoch micro-cooldown to cap GPU temp
    return enc


@torch.no_grad()
def embed(enc, X, bs=256):
    enc.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
    out = [enc(Xt[i:i + bs].to(DEVICE)).cpu().numpy() for i in range(0, len(Xt), bs)]
    return np.concatenate(out)


def run(subject):
    es = build_epochs(subject=subject, win=2.0, step=0.5, l_freq=8.0, h_freq=30.0, zscore=True)
    X, y, g = es.X, es.y, es.session.astype(int)
    motion = es.imu_feats[:, 1]
    sess = sorted(set(g)); tr = g <= sess[2]; te = ~tr
    if tr.sum() < 30 or te.sum() < 20 or len(set(y[te])) < 2:
        print(f"  [{subject}] bad split"); return None
    enc = pretrain(X[tr], C=X.shape[1])           # unlabeled contrastive pretrain on train
    E = embed(enc, X)
    def acc(Z):
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        c.fit(Z[tr], y[tr]); return balanced_accuracy(y[te], c.predict(Z[te]))
    raw = acc(E)
    res = Ridge(alpha=1.0).fit(motion[tr].reshape(-1, 1), E[tr])
    Einv = E - res.predict(motion.reshape(-1, 1))
    inv = acc(Einv)
    out = {"subject": subject, "ssl_raw": raw, "ssl_inv": inv}
    print(f"  [{subject}] SSL linear-probe raw={raw:.3f} | motion-removed={inv:.3f}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for sub in SUBJECTS:
        print(f"\n######## {sub} (contrastive SSL, GPU) ########", flush=True)
        try:
            r = run(sub)
            if r:
                res[sub] = r; (RESULTS / "ssl.json").write_text(json.dumps(res, indent=2))
        except Exception as e:
            print(f"  [{sub}] ERROR {e}")
        time.sleep(8)                        # cooldown between subjects
    if res:
        m = lambda k: float(np.nanmean([res[s][k] for s in res]))
        print("\n============ CONTRASTIVE SSL (mean) ============")
        print(f"  SSL raw            : {m('ssl_raw'):.3f}")
        print(f"  SSL motion-removed : {m('ssl_inv'):.3f}")
