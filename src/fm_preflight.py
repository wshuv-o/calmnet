"""Pre-flight test for a frozen EEG foundation model branch.

The question
------------
Every intervention tried so far recovers errors the decoder already makes, and
E1 showed why that cannot work here: the gain from a causal HMM filter scales
as -0.418 x baseline (r = -0.984), so at a baseline of 0.88 it is negative. A
frozen encoder pretrained on thousands of hours is the one candidate that adds
information from OUTSIDE this dataset, so the baseline-dependence argument does
not bind on it.

Before building a branch, three cheap measurements decide whether it is worth
it. No training of our model, no GPU queue, one forward pass per window.

  1. LINEAR PROBE. Logistic regression on frozen features, trained on the same
     fitting sessions and scored on the same held-out sessions as pipeline C,
     with the same balanced_accuracy. Reference points in the same harness:
     dn_noctx 0.8682 (3 seeds), dn_stem 0.8824 (2 seeds). If the probe lands
     far below those, a branch cannot add +0.03 to a model already there.

  2. SESSION PROBE. How well the features predict which session a window came
     from. Our whole contribution is a drift-correction layer, so a frozen
     encoder that encodes recording day more strongly than our own features do
     would import the problem the paper exists to solve. Reported beside the
     same probe on log band power so the comparison is like for like.

  3. SUBJECT/SESSION-AXIS ERASURE. Lin et al. 2026 (arXiv 2606.06647) find the
     frozen subject-variance fraction of LaBraM, CBraMod and REVE runs at
     13-89x a random null in 12 of 12 model-dataset pairs, and that erasing
     that linear axis improves label decoding by +6 to +12 pp where the label
     varies within subject. Walk against Stop varies within participant, so
     that is our cell. Here the nuisance axis is the SESSION, since each
     decoder is fit per participant and subject identity is constant.

A control matters as much as the probe: the same logistic regression on log
band power, which is what our own stem sees. It bounds how much of any result
is the probe rather than the representation.

    FM_MODEL=labram python src/fm_preflight.py

Writes results/fm_preflight_<model>.json.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.signal import butter, filtfilt, resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abstain import balanced_accuracy                       # noqa: E402
from exp_calmnetx import load                                # noqa: E402

FM = os.environ.get("FM_MODEL", "labram")
REPO = {"labram": "braindecode/labram-pretrained",
        "cbramod": "braindecode/cbramod-pretrained",
        "eegpt": "braindecode/eegpt-pretrained"}[FM]
CLS = {"labram": "Labram", "cbramod": "CBraMod", "eegpt": "EEGPT"}[FM]
# LaBraM patches at 200 samples = 1 s, so the input has to be at 200 Hz. Our
# cache is 100 Hz, so this is an upsample: it adds no information and is done
# only to meet the patch convention the pretrained weights were trained under.
FM_SFREQ = 200.0
SUBS = [s for s in os.environ.get("FM_SUBS", "").split(",") if s] or \
       ["sub-0%d" % i for i in range(1, 8)]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "fm_preflight_%s.json" % FM)


def build_fm(ch_names, n_times):
    """Pretrained encoder at OUR montage and window length.

    from_pretrained cannot be used directly. The published config pins the
    model's 128-name channel vocabulary, and the checkpoint's temporal
    embedding is sized for 15 s at 200 Hz (16 patches) while a 4 s window needs
    5, so strict loading fails on shape. Building at our shape and slicing the
    over-long embedding is the standard treatment for a position embedding.

    The slicing is verified rather than trusted: the fraction of parameters
    actually restored is asserted, because a silent shape mismatch would leave
    part of the network at its random initialisation and the probe would then
    measure noise while looking like a real result. Only the two
    classification-head tensors are expected to be missing, and they are
    discarded anyway since the features are tapped at the head's input.
    """
    import braindecode.models as M
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cls = getattr(M, CLS)
    model = cls(n_outputs=2, n_chans=len(ch_names), n_times=n_times,
                sfreq=FM_SFREQ, chs_info=[{"ch_name": c} for c in ch_names])
    have = model.state_dict()
    ck = load_file(hf_hub_download(REPO, "model.safetensors"))
    use, sliced = {}, []
    for k, v in ck.items():
        if k in have and have[k].shape != v.shape:
            tgt = have[k].shape
            if len(tgt) == len(v.shape) and all(t <= s for t, s in
                                               zip(tgt, v.shape)):
                use[k] = v[tuple(slice(0, t) for t in tgt)]
                sliced.append((k, tuple(v.shape), tuple(tgt)))
            else:
                raise RuntimeError("cannot reconcile %s: %s vs %s"
                                   % (k, tuple(v.shape), tuple(tgt)))
        else:
            use[k] = v
    missing, unexpected = model.load_state_dict(use, strict=False)
    restored = sum(have[k].numel() for k in use if k in have)
    total = sum(p.numel() for p in model.parameters())
    frac = restored / total
    for k, a, b in sliced:
        print("  sliced %s %s -> %s" % (k, a, b), flush=True)
    print("  restored {:,} of {:,} parameters ({:.2f}%), missing {}".format(
        restored, total, 100 * frac, list(missing)), flush=True)
    if frac < 0.99:
        raise RuntimeError("only %.1f%% of the encoder loaded; the probe would "
                           "measure a partly random network" % (100 * frac))
    if unexpected:
        raise RuntimeError("unexpected keys: %s" % list(unexpected)[:8])
    return model.to(DEV).eval(), frac


def feature_tap(model):
    """Hook the input of the last Linear layer: the pooled representation the
    classification head reads, which is what a branch would consume."""
    import torch.nn as nn
    lin = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    name, mod = lin[-1]
    store = {}
    mod.register_forward_pre_hook(
        lambda _m, inp: store.__setitem__("f", inp[0].detach()))
    return name, store


@torch.no_grad()
def extract(model, store, X, ch_names, batch=32):
    """Frozen features for every window, in recording order."""
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch]).float().to(DEV)
        # ch_names is required per batch: the model looks each channel up in
        # its own vocabulary rather than assuming a fixed montage order.
        model(xb, ch_names=ch_names)
        f = store["f"]
        out.append(f.reshape(len(f), -1).float().cpu().numpy())
    return np.concatenate(out)


def to_fm_input(X, sfreq_in=100.0):
    """Resample to the model's rate and standardise per channel.

    Standardisation uses each split's own statistics per channel, which is what
    the alignment-free baselines in this project already do (exp_globalnorm),
    so the probe is not handed a normalisation the decoders do not get.
    """
    n = int(round(X.shape[-1] * FM_SFREQ / sfreq_in))
    # Resample in float32 chunks. The whole split at float64 is 2.2 GB for the
    # largest participant (6201 x 60 x 800) and blew up on sub-06; the model
    # casts to float32 on entry anyway, so the wider dtype bought nothing.
    out = np.empty((len(X), X.shape[1], n), dtype=np.float32)
    for i in range(0, len(X), 512):
        out[i:i + 512] = resample(X[i:i + 512].astype(np.float32), n,
                                  axis=-1).astype(np.float32)
    mu = out.mean(axis=(0, 2), keepdims=True)
    sd = out.std(axis=(0, 2), keepdims=True) + 1e-8
    out -= mu
    out /= sd
    return out


def band_power(X, sfreq=100.0, bands=((4, 8), (8, 13), (13, 30), (30, 45))):
    """Log band power per channel: what our own stem sees. The control."""
    out = []
    for lo, hi in bands:
        b, a = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band")
        Xf = filtfilt(b, a, X.astype(np.float64), axis=-1)
        out.append(np.log(Xf.var(axis=-1) + 1e-12))
    return np.concatenate(out, axis=1)


def probe(Ff, yf, Ft, yt, seed=0):
    """Logistic regression, fitting split to test split, balanced accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ff)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=seed)
    clf.fit(sc.transform(Ff), yf)
    return float(balanced_accuracy(yt, clf.predict(sc.transform(Ft))))


def multiclass_probe(F, g, seed=0):
    """How well the features predict which session a window came from.

    Cross-validated across windows, so it measures whether the information is
    present at all, not whether it generalises to a new session.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    u = np.unique(g)
    if len(u) < 2:
        return float("nan"), float("nan")
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000,
                                            class_weight="balanced",
                                            random_state=seed))
    sc = cross_val_score(pipe, F, g, cv=3, scoring="balanced_accuracy")
    return float(sc.mean()), float(1.0 / len(u))


def erase_axes(Ff, gf, Ft, k=None):
    """Project out the session axis, estimated on the FITTING split only.

    The directions come from the fitting sessions' class-conditional-free means,
    so nothing about the held-out sessions is used to build the projector. k
    defaults to the number of fitting sessions minus one, which spans their
    mean differences.
    """
    u = np.unique(gf)
    if len(u) < 2:
        return Ff, Ft
    mus = np.stack([Ff[gf == s].mean(0) for s in u])
    D = mus - mus.mean(0, keepdims=True)
    k = (len(u) - 1) if k is None else k
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    B = Vt[:max(1, min(k, Vt.shape[0]))]                  # (k, d)
    P = np.eye(Ff.shape[1]) - B.T @ B                      # orthogonal project
    return Ff @ P, Ft @ P


def main():
    print("frozen %s (%s) on %s, %d participants" % (FM, REPO, DEV, len(SUBS)),
          flush=True)
    res = {"model": FM, "repo": REPO, "cohort": "A", "sfreq_in": 100.0,
           "sfreq_fm": FM_SFREQ, "per_subject": {},
           "_reference": {"dn_noctx_3seed": 0.8682, "dn_stem_2seed": 0.8824,
                          "_note": "same harness, same splits, pipeline C"},
           "_caveats": [
               "Input is upsampled 100 -> 200 Hz to meet the model's 1 s patch "
               "convention. No information is added by that step.",
               "A linear probe is a lower bound on what a trained branch could "
               "extract, and an upper bound on what a frozen branch adds for "
               "free.",
               "LEAKAGE IS NOT AUDITED. ds007788 is public on OpenNeuro and "
               "this model's pretraining corpus has not been enumerated here. "
               "If our test sessions are in it, every number below is void.",
           ]}
    model, store, tap = None, None, None
    for sub in SUBS:
        t0 = time.time()
        try:
            d = load(sub, 0)
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
            continue
        ch = list(__import__("dataio").build_epochs(
            subject=sub, win=4.0, step=0.5, zscore=False).ch_names)
        Xf, Xt = to_fm_input(d["Xf"]), to_fm_input(d["Xt"])
        if model is None:
            model, frac = build_fm(ch, Xf.shape[-1])
            tap, store = feature_tap(model)
            res["encoder_params_restored"] = frac
            res["feature_tap"] = tap
            print("  feature tap: %s" % tap, flush=True)
        Ff = extract(model, store, Xf, ch)
        Ft = extract(model, store, Xt, ch)

        row = {"n_fit": int(len(Ff)), "n_test": int(len(Ft)),
               "feat_dim": int(Ff.shape[1])}
        row["probe_fm"] = probe(Ff, d["yf"], Ft, d["yt"])
        Pf, Pt = band_power(d["Xf"]), band_power(d["Xt"])
        row["probe_bandpower"] = probe(Pf, d["yf"], Pt, d["yt"])
        Ef, Et = erase_axes(Ff, d["sf"], Ft)
        row["probe_fm_erased"] = probe(Ef, d["yf"], Et, d["yt"])
        row["session_probe_fm"], row["session_chance"] = \
            multiclass_probe(Ff, d["sf"])
        row["session_probe_bandpower"], _ = multiclass_probe(Pf, d["sf"])
        res["per_subject"][sub] = row
        print("  %-8s probe FM %.3f | erased %.3f | bandpower %.3f || "
              "session FM %.3f vs bp %.3f (chance %.3f)  [%.0fs]"
              % (sub, row["probe_fm"], row["probe_fm_erased"],
                 row["probe_bandpower"], row["session_probe_fm"],
                 row["session_probe_bandpower"], row["session_chance"],
                 time.time() - t0), flush=True)
        io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))

    r = res["per_subject"]
    if r:
        print()
        for k in ("probe_fm", "probe_fm_erased", "probe_bandpower",
                  "session_probe_fm", "session_probe_bandpower"):
            v = [x[k] for x in r.values() if not np.isnan(x.get(k, np.nan))]
            if v:
                res.setdefault("summary", {})[k] = {
                    "mean": float(np.mean(v)),
                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    "n": len(v)}
                print("  %-26s %.4f  (n=%d)" % (k, np.mean(v), len(v)))
        io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("\nwrote %s" % os.path.relpath(OUT), flush=True)


if __name__ == "__main__":
    main()
