import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (DEVICE, balanced_accuracy, fit_temperature, softmax_np,
                   weighted_ce)


class baseModel:
    """Training, calibration, prediction and saving for every model in the paper."""

    def __init__(self, net, train_cfg, batch, out_dir=None, device=DEVICE):
        self.net, self.cfg, self.batch, self.device = net.to(device), train_cfg, batch, device
        self.out_dir = Path(out_dir) if out_dir else None
        self.history = {"train_loss": [], "val_bacc": []}
        self.best_epoch, self.best_val, self.temperature = 0, float("nan"), 1.0

    def fit(self, Xf, yf, Xc, yc):
        """AdamW + OneCycle; keep the epoch with the best held-out balanced accuracy."""
        c, net = self.cfg, self.net
        Xf, yf = torch.as_tensor(Xf), torch.as_tensor(yf)
        opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=c["lr"], total_steps=c["epochs"] * max(1, len(yf) // self.batch + 1))
        cnt = np.bincount(yf.numpy(), minlength=2).astype(float)
        w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32,
                         device=self.device)
        dl = DataLoader(TensorDataset(torch.arange(len(yf)), yf), batch_size=self.batch,
                        shuffle=True)

        best, best_sc, best_ep = None, -1e9, 0
        for ep in range(c["epochs"]):
            net.train()
            losses = []
            for bi, by in dl:
                xb, by = Xf[bi].to(self.device), by.to(self.device)
                opt.zero_grad()
                loss = weighted_ce(net(xb), by, w)
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), c["clip"])
                opt.step()
                sched.step()
                losses.append(loss.item())
            sc = balanced_accuracy(yc, self.predict_logits(Xc).argmax(1))
            self.history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
            self.history["val_bacc"].append(float(sc))
            if sc > best_sc:
                best_sc, best_ep = sc, ep
                best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if ep - best_ep >= c["patience"]:
                break
        net.load_state_dict(best)                        # also restores the reference
        self.best_epoch, self.best_val = best_ep, float(best_sc)
        return self.history

    def calibrate(self, Xc, yc):
        """Temperature scaling on the held-out split (one pass; it updates the reference)."""
        self.cal_logits = self.predict_logits(Xc)
        self.temperature = float(np.clip(fit_temperature(self.cal_logits, yc), 0.5, 5.0))
        return self.temperature

    @torch.no_grad()
    def predict_logits(self, X):
        """Decode in array order, which must be recording order."""
        self.net.eval()
        X = torch.as_tensor(X)
        return np.concatenate([self.net(X[i:i + self.batch].to(self.device)).float().cpu().numpy()
                               for i in range(0, len(X), self.batch)])

    def predict_proba(self, X):
        return softmax_np(self.predict_logits(X), self.temperature)

    def save(self, name, results=None, y_true=None, y_pred=None, classes=("0", "1")):
        if self.out_dir is None:
            return
        d = self.out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "temperature": self.temperature,
                    "best_epoch": self.best_epoch}, d / "model.pth")
        (d / "history.json").write_text(json.dumps(
            {**self.history, "best_epoch": self.best_epoch, "best_val_bacc": self.best_val,
             "temperature": self.temperature}, indent=1))
        if results is not None:
            (d / "test_results.json").write_text(json.dumps(results, indent=1))
        self.plot_history(d / "loss-acc.png")
        if y_true is not None:
            self.plot_confusion(y_true, y_pred, classes, d / "confusion_matrix.png")

    def plot_history(self, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ep = np.arange(1, len(self.history["train_loss"]) + 1)
        ax.plot(ep, self.history["train_loss"], color="#0072B2", label="training loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax2 = ax.twinx()
        ax2.plot(ep, self.history["val_bacc"], color="#D55E00", label="held-out bal. acc.")
        ax2.axvline(self.best_epoch + 1, color="grey", lw=0.8, ls="--")
        ax2.set_ylabel("held-out balanced accuracy")
        fig.legend(loc="upper center", ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_confusion(y_true, y_pred, classes, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
        fig, ax = plt.subplots(figsize=(3, 2.6))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, "%.2f" % cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
