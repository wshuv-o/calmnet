"""Is the motion canceller cancelling artefact, or decoding from the IMU?

Under the corrected (label-conditional) probe the MotionReferenceCanceller
jumps to ~0.905 balanced accuracy from a bare-model 0.811, and shows NO excess
movement information. That looks like a clean win, and it is exactly the kind of
result that deserves one more control before it is believed.

The worry is specific. The layer computes

    y = x - g * h(m)

so the motion reference is not merely consulted, it is ALGEBRAICALLY INJECTED
into the signal the classifier then reads -- with a minus sign, but a linear
classifier does not care about signs. If h(m) happens to track Walk vs Stop
(and it trivially can: the legs move during walking), the classifier can read
the label straight out of the injected term and never touch the EEG. Accuracy
would be excellent and the decoder would be worthless, because it would be an
IMU classifier wearing an EEG costume.

The label-conditional probe CANNOT catch this. It removes exactly the
label-explained component of motion, which is precisely the component such a
shortcut would be exploiting. The fix for one confound is blind to this one.

So: break the pairing at inference and see what survives.

  true      motion reference as recorded                  (reference condition)
  shuffled  motion rows permuted across windows -- same marginal distribution,
            no correspondence to the EEG window it accompanies
  zeroed    motion set to 0, which makes the canceller exactly the identity
            (y = x - g*h(0), and h is bias-free, so h(0) = 0)

Predictions, stated before running:

  If the layer genuinely cancels artefact, `zeroed` should fall back TOWARD the
  bare model (~0.811) -- cancellation is simply switched off -- and `shuffled`
  should land near it or slightly worse, since it subtracts an irrelevant
  waveform.

  If the layer is decoding from the IMU, both should collapse BELOW the bare
  model, because the classifier has learned to lean on an input that is now
  uninformative or actively misleading.

The gap between `true` and `zeroed` is the honest size of the canceller's
contribution.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from exp_ablate import (load_all, train_arch, predict_arch, BARE, ALL_MODS)
from calibrate import fit_temperature, softmax_np
from abstain import balanced_accuracy

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "cancel_control.json"


def perturb(M, kind, seed=0):
    if kind == "true":
        return M
    if kind == "zeroed":
        return np.zeros_like(M)
    if kind == "shuffled":
        rng = np.random.default_rng(seed)
        return M[rng.permutation(len(M))]
    raise ValueError(kind)


KINDS = ("true", "shuffled", "zeroed")


def run(mods, D, label):
    """Train ONCE per subject, then evaluate the same model under every
    perturbation.

    The perturbations differ only at inference, so retraining for each one costs
    3x the GPU time to produce bit-identical models (train_arch calls set_seed,
    so the runs are deterministic). Training once also makes the comparison
    exact rather than merely reproducible: the three numbers come from the same
    weights, so any difference between them is the perturbation and nothing
    else.
    """
    accs = {k: [] for k in KINDS}
    for sub, d in D.items():
        model, _ = train_arch(mods, d["Xf"], d["Mf"], d["yf"],
                              d["Xv"], d["Mv"], d["yv"], seed=0)
        lgv, _ = predict_arch(model, d["Xv"], d["Mv"])
        T = float(np.clip(fit_temperature(lgv, d["yv"]), 0.5, 5.0))
        for kind in KINDS:
            lg, _ = predict_arch(model, d["Xt"], perturb(d["Mt"], kind))
            p = softmax_np(lg, T)
            if not np.isfinite(p).all():
                p = np.full_like(p, 0.5)
            accs[kind].append(balanced_accuracy(d["yt"], p.argmax(1)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    res = {k: float(np.mean(v)) for k, v in accs.items()}
    for k in KINDS:
        print("  %-10s %-9s acc %.3f" % (label, k, res[k]), flush=True)
    return res


def main():
    print("loading ...", flush=True)
    D = load_all()
    print("%d subjects\n" % len(D), flush=True)
    out = {}
    # bare has no canceller, so the motion reference should be irrelevant to it.
    # If bare moves when motion is perturbed, the harness itself is wrong and the
    # cancel numbers mean nothing -- this is the experiment's own sanity check.
    out["bare"] = run(dict(BARE), D, "bare")
    out["cancel"] = run({**BARE, "cancel": True}, D, "+cancel")
    # spec_gate ALSO consumes motion at inference: SpectralLeakageGate.forward
    # takes the motion envelope and builds a multiplicative per-band gate from
    # it, so its output is motion-dependent by construction. A multiplicative
    # route is subtler than the canceller's additive one, but it is still a route
    # -- and its accuracy gain (+0.008) is too small to be worth trusting on
    # assumption. The loss-term modules (adv/decorr/hsic/art) use motion only as
    # a training TARGET and receive none at inference, so they need no control.
    out["spec_gate"] = run({**BARE, "spec_gate": True}, D, "+spec_gate")
    OUT.write_text(json.dumps(out, indent=1))

    b = out["bare"]
    print("\n" + "=" * 70)
    print("%-12s %8s %9s %8s | %s" % ("config", "true", "shuffled", "zeroed", "verdict"))
    print("-" * 70)
    for name, r in out.items():
        if name == "bare":
            v = ("harness OK: invariant to motion"
                 if max(abs(r[k] - r["true"]) for k in KINDS) < 0.005
                 else "HARNESS BROKEN: bare should not move")
        elif r["shuffled"] < b["true"] - 0.02 or r["zeroed"] < b["true"] - 0.02:
            v = "DECODES FROM IMU -- reject"
        elif r["true"] - max(r["shuffled"], r["zeroed"]) > 0.02:
            v = "partly motion-driven"
        else:
            v = "EEG-only (gain, if any, is real)"
        print("%-12s %8.3f %9.3f %8.3f | %s"
              % (name, r["true"], r["shuffled"], r["zeroed"], v))
    print("-" * 70)
    print("bare reference accuracy: %.3f" % b["true"])
    print("A module is only credible if perturbing the motion reference leaves it")
    print("at or above the bare model -- anything that falls below bare was")
    print("leaning on the IMU rather than the EEG.")


if __name__ == "__main__":
    main()
