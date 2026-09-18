"""ATCNet+ : add-one module ablation on the best published decoder, at full data.

ATCNet is the strongest architecture measured here once the training set is large
enough (0.881 at 4881 fit windows vs 0.828 at 905). It is the BACKBONE, used
unmodified; each module is switchable so it has to earn its place.

  base       published ATCNet + linear head            <- the number to beat
  rate       + temporal hyperparameters matched to 100 Hz
  rate+ctx   + causal cross-epoch attention (15 s span)
  rate+gate  + trained abstention head
  full       all three

WHY THESE THREE

  rate  ATCNet's hyperparameters are in SAMPLES and tuned at 250 Hz. At 100 Hz
        the same counts stretch every receptive field 2.5x and push the input
        under braindecode's minimum, which silently rewrites the model:
        n_windows 5 -> 3 (its windowed attention ensemble), tcn_kernel 4 -> 2.
        It reaches 0.881 with 40% of that ensemble deleted.

  ctx   ATCNet attends over sliding windows INSIDE one 4 s epoch. The structure
        in this task is at 12-32 s -- self-transition 0.98/0.96 at a 0.5 s step,
        and a fixed label-space dwell prior cut spurious activations ~3x on both
        cohorts across 4 seeds. PowerAttnNet's attention failed precisely because
        it sat at the within-epoch timescale (deleting it was worth +0.024). This
        attends ACROSS epochs: K=12 at stride 2 spans 15 s. Subsampled because
        consecutive epochs overlap 87.5%, causal because a worn device has no
        future. The comparison is sharp: LEARNED persistence against a
        hand-specified scalar prior that already works.

  gate  SelectiveNet-style abstention trained under a coverage constraint, so
        declining to act is learned with the decision rather than thresholded
        onto a finished classifier.

FIVE REPORTED OUTCOMES, not one: balanced accuracy, ECE, accuracy at 90%
coverage (ranked by the trained gate), false activations per minute of standing,
detection latency, and conditional movement R^2. A decoder that starts and stops
a person's legs is not described by accuracy.

Writes results/atcplus.json.
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from calmnet_msa import imu_valid_mask
from abstain import balanced_accuracy
from calibrate import expected_calibration_error, softmax_np, fit_temperature
from train import set_seed, DEVICE
from exp_temporal import (streams, onset_metrics, forward_filter,
                          transition_matrix)
from exp_globalnorm import normalise
from atcnet_plus import build_atcnet_plus, selective_loss
from driftnet import build_driftnet
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / os.environ.get("CX_OUT", "atcplus.json")
# add-one ablation on the published ATCNet backbone. `base` is the
# unmodified model plus a linear head -- the number to beat (0.881).
ARMS = {
    "base":      dict(rate=False, use_ctx=False, use_gate=False),
    "rate":      dict(rate=True,  use_ctx=False, use_gate=False),
    "rate+ctx":  dict(rate=True,  use_ctx=True,  use_gate=False),
    "rate+gate": dict(rate=True,  use_ctx=False, use_gate=True),
    "full":      dict(rate=True,  use_ctx=True,  use_gate=True),
    # Stock ATCNet kernels + our two modules. The rate-matched kernels HURT
    # (0.9079 vs 0.9129 stock), so the honest test of whether context and gating
    # add anything is to bolt them onto the unmodified backbone.
    "atc+ours":  dict(rate=False, use_ctx=True,  use_gate=True),
}
# DriftNet ablation. `full` is the proposed architecture; each `no_*` arm
# removes exactly one component so it has to earn its place. `no_align` is the
# one that matters -- it isolates the in-network session adaptation, which is
# the part not taken from any prior architecture.
DRIFT_ARMS = {
    "dn_full":     dict(use_align=True,  use_ctx=True,  use_gate=True),
    "dn_noalign":  dict(use_align=False, use_ctx=True,  use_gate=True),
    "dn_noctx":    dict(use_align=True,  use_ctx=False, use_gate=True),
    "dn_nogate":   dict(use_align=True,  use_ctx=True,  use_gate=False),
    "dn_stem":     dict(use_align=False, use_ctx=False, use_gate=False),
    # Second-order branch. Each is its own control's arm plus ONE module, so
    # the ablation is that module in or out and nothing else moves: dn_tan
    # against dn_noctx, dn_stem_tan against dn_stem. Neither changes the
    # training recipe, so the published baselines stay comparable.
    "dn_tan":      dict(use_align=True,  use_ctx=False, use_gate=True,
                        use_tangent=True),
    "dn_stem_tan": dict(use_align=False, use_ctx=False, use_gate=False,
                        use_tangent=True),
    # Cohort B's reference arm is dn_gate (alignment off, which is what the
    # rate condition requires there), so the module ablation on that cohort
    # needs its own control pair rather than borrowing cohort A's.
    "dn_gate_tan": dict(use_align=False, use_ctx=False, use_gate=True,
                        use_tangent=True),
    # Selection head without alignment or context. Paired with dn_noctx it
    # isolates alignment as the single differing factor, which the pre-registered
    # third-cohort test (paper/PREREGISTRATION_cohort3.md) requires.
    "dn_gate":     dict(use_align=False, use_ctx=False, use_gate=True),
    # Alignment alone. Against dn_noctx (align + gate) it measures what the
    # selection head adds to accuracy, separately from its failed ranking role.
    "dn_align":    dict(use_align=True, use_ctx=False, use_gate=False),
}
MODEL = os.environ.get("CX_MODEL", "atcplus")
if MODEL == "driftnet":
    ARMS = DRIFT_ARMS
# CX_MODEL=bd:EEGNeX+ShallowFBCSPNet runs published braindecode decoders in this
# pipeline, each under the wrapper the stock-ATCNet `base` arm uses, so they are
# comparable with it and with DriftNet (pipelines are never compared across).
elif MODEL.startswith("bd:"):
    ARMS = {n: dict(use_ctx=False, use_gate=False, bd=n) for n in MODEL[3:].split("+")}
# CX_ARMS restricts the run to named arms. Used when a question only needs a
# subset -- e.g. "does slow adaptation hold on cohort A" is answered by
# dn_full vs dn_noalign alone, and the other three arms are already measured
# at the default momentum.
_sel = [a for a in os.environ.get("CX_ARMS", "").split(",") if a]
if _sel:
    ARMS = {k: v for k, v in ARMS.items() if k in _sel}
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN, WIN, STEP = 3, 4.0, 0.5
K_CTX, S_CTX = 8, 3
EPOCHS = int(os.environ.get("CX_EPOCHS", "30"))
SIZE = os.environ.get("CX_SIZE", "base")
TRIALS = tuple("trial%02d" % i for i in range(1, 13))
COHORT = os.environ.get("CX_COHORT", "ds007788")


def load_mobi(sub, seed):
    """Second cohort (Luu et al. treadmill BCI) in the same dict shape.

    External validation: independent lab, independent subjects, goniometers
    instead of IMU, treadmill walk/stand instead of exoskeleton walk/stop,
    64 channels instead of 60, and the class imbalance REVERSED -- Stop is ~12%
    here and the majority on ds007788. An architecture that only worked on
    whichever class happens to dominate would fail here.

    Windows are 2 s (this cohort's cached protocol) rather than the 4 s used on
    ds007788; the context module's span scales with the window, so K is kept and
    the covered duration is stated rather than matched.

    Split convention (fit trial 1, test trials 2-3) matches exp_mobi.py and
    exp_calmnet3.py so the numbers stay comparable with what is already reported.
    """
    from dataio_mobi import build_subject
    es = build_subject(sub, win=2.0, step=0.5, zscore=False)
    if es is None or len(es) < 100:
        raise RuntimeError("no MoBI data for %s" % sub)
    fit, test = es.by_trials([1]), es.by_trials([2, 3])
    ti, ci = grouped_split(fit.segment, fit.y, frac=0.3, seed=seed)
    const = lambda n: np.array(["mobi"] * n, object)
    return {"Xf": fit.X[ti], "yf": fit.y[ti], "sf": fit.trial[ti],
            "tf": const(len(ti)), "gf": fit.segment[ti],
            "Xc": fit.X[ci], "yc": fit.y[ci], "sc": fit.trial[ci],
            "tc": const(len(ci)), "gc": fit.segment[ci],
            "Xt": test.X, "yt": test.y, "st": test.trial,
            "tt": const(len(test)), "gt": test.segment,
            "imu_t": test.motion, "vt": np.ones(len(test), bool)}


def load_eegbci(sub, seed):
    """Third cohort (EEGMMIDB), in the same dict shape.

    Defined by paper/PREREGISTRATION_cohort3.md, committed before any model
    touched this cohort: motor execution, rest against movement, fit runs
    3/5/7/9, test runs 11/13. Blocks serve as segments, so the grouped hold-out
    keeps each class block whole. No IMU exists, so the leakage probe is
    skipped (vt all False) rather than filled with a meaningless value.
    """
    from dataio_eegbci import build_subject
    p = build_subject(sub)
    Xf, yf, rf, gf = p["fit"]
    Xt, yt, rt, gt = p["test"]
    ti, ci = grouped_split(gf, yf, frac=0.3, seed=seed)
    ti, ci = np.sort(ti), np.sort(ci)      # keep time order for streams()
    const = lambda n: np.array(["exec"] * n, object)
    return {"Xf": Xf[ti], "yf": yf[ti], "sf": rf[ti], "tf": const(len(ti)),
            "gf": gf[ti],
            "Xc": Xf[ci], "yc": yf[ci], "sc": rf[ci], "tc": const(len(ci)),
            "gc": gf[ci],
            "Xt": Xt, "yt": yt, "st": rt, "tt": const(len(yt)), "gt": gt,
            "imu_t": np.zeros((len(yt), 1), np.float32),
            "vt": np.zeros(len(yt), bool)}


def context_index(strm, n, k=K_CTX, s=S_CTX):
    """For every epoch, the indices of its k-1 predecessors at stride s.

    Positions near the start of a stream are edge-padded by repeating the
    earliest available epoch, so every sample has a full-length context and the
    batch stays rectangular. Contexts never cross a stream boundary, which would
    splice together epochs from different recordings.
    """
    idx = np.zeros((n, k), dtype=np.int64)
    pos = np.full(n, -1, dtype=np.int64)
    for st in strm:
        for j, e in enumerate(st):
            pos[e] = j
    for st in strm:
        arr = np.asarray(st)
        for j, e in enumerate(arr):
            back = j - np.arange(k - 1, -1, -1) * s
            back = np.clip(back, 0, None)
            idx[e] = arr[back]
    return idx


# Artefact control. CX_ICA=1 removes ICLabel-classified artefact components
# before band-passing (dataio._ica_clean). CX_FULL=0 restricts fitting to the
# training task, dropping the extra trial recordings; the ICA control uses it for
# both of its arms, so the two differ only in the cleaning step.
ICA = os.environ.get("CX_ICA", "0") == "1"
# E1: causal HMM forward filter over the decoder output. Off by default so
# the unsmoothed arm still runs and the two are comparable in one file.
SMOOTH = os.environ.get("CX_SMOOTH", "0") == "1"
FULL = os.environ.get("CX_FULL", "1") == "1"


def load(sub, seed, full=None):
    full = FULL if full is None else full
    es = build_epochs(subject=sub, win=WIN, step=STEP, zscore=False, ica=ICA)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, ci = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=seed)
    d = {"Xf": es.X[tr][ti], "yf": es.y[tr][ti],
         "sf": es.session[tr][ti], "tf": es.task[tr][ti], "gf": es.segment[tr][ti],
         "Xc": es.X[tr][ci], "yc": es.y[tr][ci],
         "sc": es.session[tr][ci], "tc": es.task[tr][ci], "gc": es.segment[tr][ci],
         "Xt": es.X[~tr], "yt": es.y[~tr], "st": es.session[~tr],
         "tt": es.task[~tr], "gt": es.segment[~tr],
         "imu_t": es.imu_feats[~tr], "vt": valid[~tr]}
    if full:
        ex = build_epochs(subject=sub, sessions=sess[:N_TRAIN], tasks=TRIALS,
                          win=WIN, step=STEP, zscore=False, ica=ICA)
        d["Xf"] = np.concatenate([d["Xf"], ex.X])
        d["yf"] = np.concatenate([d["yf"], ex.y])
        d["sf"] = np.concatenate([d["sf"], ex.session])
        d["tf"] = np.concatenate([d["tf"], ex.task])
        d["gf"] = np.concatenate([d["gf"], ex.segment])
    return d


def make_split(X, y, s, t, g, k):
    strm = streams(s, t, g)
    idx = context_index(strm, len(y), k=k)
    return torch.as_tensor(X), torch.as_tensor(y), torch.as_tensor(idx), strm


@torch.no_grad()
def infer(model, X, idx, batch=32):
    model.eval()
    L, G = [], []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        xb = X[b.reshape(-1)].to(DEVICE).reshape(b.shape[0], b.shape[1],
                                                 X.shape[1], X.shape[2])
        o = model(xb)
        L.append(o["logits"].float().cpu().numpy())
        G.append(o["gate"].float().cpu().numpy())
    return np.concatenate(L), np.concatenate(G)


def run(d, arm, seed):
    cfg = ARMS[arm]
    use_context = cfg["use_ctx"]
    k = K_CTX if use_context else 1
    Xf, Xc, Xt = normalise("global", d["Xf"], d["Xc"], d["Xt"])
    tXf, tyf, ixf, strm_f = make_split(Xf, d["yf"], d["sf"], d["tf"],
                                       d["gf"], k)
    tXc, tyc, ixc, _ = make_split(Xc, d["yc"], d["sc"], d["tc"], d["gc"], k)
    tXt, tyt, ixt, strm_t = make_split(Xt, d["yt"], d["st"], d["tt"], d["gt"], k)

    set_seed(seed)
    if MODEL == "driftnet":
        model = build_driftnet(Xf.shape[1], Xf.shape[2], 2, size=SIZE,
                               **cfg).to(DEVICE)
    elif "bd" in cfg:
        from atcnet_plus import build_bd_plus
        model = build_bd_plus(cfg["bd"], Xf.shape[1], Xf.shape[2], 2).to(DEVICE)
    else:
        model = build_atcnet_plus(Xf.shape[1], Xf.shape[2], 2, **cfg).to(DEVICE)
    # A frozen tangent reference is estimated from the fitting split before any
    # training, so the base point never sees a held-out session and never moves
    # with the test stream.
    tan = getattr(model, "tangent", None)
    if tan is not None and getattr(tan, "ref_mode", "running") == "frozen":
        tan.freeze_reference(tXf.to(DEVICE) if tXf.numel() < 4e8 else tXf)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=EPOCHS * max(1, len(ixf) // 32 + 1))
    cnt = np.bincount(d["yf"], minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32,
                     device=DEVICE)
    dl = DataLoader(TensorDataset(ixf, tyf), batch_size=32, shuffle=True)

    best, best_sc, best_ep = None, -1e9, 0
    for ep in range(EPOCHS):
        model.train()
        for bi, by in dl:
            xb = tXf[bi.reshape(-1)].to(DEVICE).reshape(bi.shape[0], bi.shape[1],
                                                        tXf.shape[1], tXf.shape[2])
            by = by.to(DEVICE)
            opt.zero_grad()
            loss, _ = selective_loss(model(xb), by, weight=w)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        lg, _ = infer(model, tXc, ixc)
        sc = balanced_accuracy(d["yc"], lg.argmax(1))
        if sc > best_sc:
            best_sc, best_ep = sc, ep
            best = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        if ep - best_ep >= 12:
            break
    model.load_state_dict(best)

    lgc, _ = infer(model, tXc, ixc)
    T = float(np.clip(fit_temperature(lgc, d["yc"]), 0.5, 5.0))
    lg, gate = infer(model, tXt, ixt)
    p = softmax_np(lg, T)
    pred = p.argmax(1)
    p_raw = p
    if SMOOTH:
        # Windows are decoded independently, yet Walk and Stop persist for
        # 12-32 s. A causal forward filter over the calibrated posteriors adds
        # the dwell structure back without seeing any future window, which is
        # the only variant an exoskeleton could run online.
        #
        # The transition matrix and the class prior come from the FITTING
        # labels only. Estimating them from the test stream would let the
        # filter read the labels it is being scored against.
        A = transition_matrix(d["yf"], strm_f, n_cls=2)
        pi = np.bincount(d["yf"], minlength=2).astype(float)
        pi = pi / pi.sum()
        # The filtered posterior replaces the raw one everywhere downstream, so
        # accuracy, calibration and the onset metrics all describe the SAME
        # predictor. Scoring accuracy on the filtered output while reporting
        # calibration from the raw output would describe two different systems
        # in one row. p_raw is kept so the pair can still be compared.
        p = p.copy()
        for st in strm_t:
            st = np.asarray(st)
            p[st] = forward_filter(p_raw[st], A, pi)
        pred = p.argmax(1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vt = d["vt"]
    r2 = (FE.invariance_r2_conditional(pred[vt, None].astype(float),
                                       d["imu_t"][vt], d["yt"][vt], d["st"][vt])
          if vt.sum() > 40 else float("nan"))
    dep = onset_metrics(d["yt"], pred, strm_t)
    # accuracy at 90% coverage, ranked by the TRAINED gate rather than a
    # post-hoc confidence threshold
    kk = max(1, int(0.9 * len(gate)))
    sel = np.argsort(-gate)[:kk]
    res = {"acc": balanced_accuracy(d["yt"], pred),
           "ece": expected_calibration_error(p, d["yt"]),
           "acc_at_90": balanced_accuracy(d["yt"][sel], pred[sel]),
           "cond_r2": r2, **dep}
    if SMOOTH:
        # The unfiltered pair from the same trained model, so the gain from
        # filtering is a within-run difference and not a comparison across two
        # separately trained networks.
        res["acc_unsmoothed"] = balanced_accuracy(d["yt"], p_raw.argmax(1))
        res["ece_unsmoothed"] = expected_calibration_error(p_raw, d["yt"])
    return res


def main():
    seeds = [int(s) for s in os.environ.get("CX_SEEDS", "0").split(",")]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    if COHORT.startswith("mobi"):
        from dataio_mobi import subjects as _ms
        subs, loader = _ms(), load_mobi
    elif COHORT == "eegbci":
        from dataio_eegbci import subjects as _es
        subs, loader = _es(), load_eegbci
    else:
        subs, loader = SUBJECTS, load
    D = {}
    for sub in subs:
        try:
            D[sub] = loader(sub, seeds[0])
        except Exception as e:
            print("  [skip] %s: %s: %s" % (sub, type(e).__name__, e), flush=True)
    n = int(np.mean([len(d["yf"]) for d in D.values()])) if D else 0
    print("%d subjects, ~%d fit windows each, size=%s, K=%d stride=%d (%.1f s span)"
          % (len(D), n, SIZE, K_CTX, S_CTX, WIN + (K_CTX - 1) * STEP * S_CTX),
          flush=True)

    for name in ARMS:
        for seed in seeds:
            key = "%s|s%d" % (name, seed)
            if key in out:
                continue
            t0, rows, subs_done = time.time(), [], []
            for sub, d in D.items():
                try:
                    if seed != seeds[0]:
                        d = loader(sub, seed)
                    t_sub = time.time()
                    rows.append(run(d, name, seed))
                    subs_done.append(sub)
                    # Per-subject progress. Without this an arm is silent for
                    # hours between start and finish, and liveness can only be
                    # checked by attaching a profiler to the process.
                    print("    %s/%s seed=%d  acc %.3f  (%.0fs, %d/%d)"
                          % (sub, name, seed, rows[-1]["acc"],
                             time.time() - t_sub, len(rows), len(D)), flush=True)
                except Exception as e:
                    print("    [fail] %s/%s: %s: %s"
                          % (sub, name, type(e).__name__, str(e)[:100]), flush=True)
            if not rows:
                continue
            agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
            agg["acc_sd"] = float(np.std([r["acc"] for r in rows]))
            # Keep the per-participant rows: cohort means alone cannot support
            # paired tests across arms, and n=7/8 makes the pairing essential.
            agg["per_subject"] = {s: {k: float(v) for k, v in r.items()}
                                  for s, r in zip(subs_done, rows)}
            out[key] = agg
            print("  %-10s seed=%d  acc %.3f+-%.3f  ece %.3f  acc@90 %.3f  "
                  "cond_r2 %+.3f  onsets/min %.2f  lat %.1fs  (%.0fs)"
                  % (name, seed, agg["acc"], agg["acc_sd"], agg["ece"],
                     agg["acc_at_90"], agg["cond_r2"],
                     agg["false_onsets_per_min"], agg["latency_s"],
                     time.time() - t0), flush=True)
            OUT.write_text(json.dumps(out, indent=1))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
