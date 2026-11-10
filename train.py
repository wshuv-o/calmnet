"""Train and test CALM-Net, the stem or a published decoder on one cohort.

    python train.py --config config/cohortA.yaml
    python train.py --config config/cohortC.yaml --causal-filter --causal-update
    python train.py --config config/cohortC.yaml --arms stem_tangent --batch 4 --momentum 1.0

Writes results/<out>.json and, per run, results/runs/<out>/<arm>_s<seed>_<subject>/
(model.pth, loss-acc.png, confusion_matrix.png, test_results.json).
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import yaml

from data import LOADERS
from model import BraindecodeDecoder, CALMNet
from model.baseModel import baseModel
from utils import (DEVICE, balanced_accuracy, conditional_r2,
                   expected_calibration_error, forward_filter, normalise_global,
                   onset_metrics, set_seed, softmax_np, streams,
                   transition_matrix)

RESULTS = Path(__file__).resolve().parent / "results"

ARMS = {"stem": dict(use_tangent=False),
        "stem_tangent": dict(use_tangent=True),
        "stem_wide": dict(use_tangent=False, n_filt=148, d_model=256)}   # capacity control


def build_model(arm, n_chan, n_time, cfg, args):
    if arm.startswith("bd:"):
        return BraindecodeDecoder(arm[3:], n_chan, n_time)
    m = cfg["model"]
    kw = dict(n_filt=m["n_filt"], d_model=m["d_model"], dropout=m["dropout"],
              shrink=m["shrink"], momentum=args.momentum or m["momentum"],
              ref_mode=args.ref_mode, causal_update=args.causal_update)
    kw.update(ARMS[arm])
    return CALMNet(n_chan, 2, **kw)


def sid(strm, n):
    s = np.full(n, -1, int)
    for i, st in enumerate(strm):
        s[st] = i
    return s


def run(d, arm, seed, cfg, args, run_dir):
    batch = args.batch or cfg["train"]["batch"]
    Xf, Xc, Xt = normalise_global(d["Xf"], d["Xc"], d["Xt"])
    strm_f = streams(d["sf"], d["tf"], d["gf"])
    strm_c = streams(d["sc"], d["tc"], d["gc"])
    strm_t = streams(d["st"], d["tt"], d["gt"])

    set_seed(seed)
    net = build_model(arm, Xf.shape[1], Xf.shape[2], cfg, args)
    tan = getattr(net, "tangent", None)
    if tan is not None and tan.ref_mode == "frozen":
        tan.to(DEVICE).freeze_reference(torch.as_tensor(Xf).to(DEVICE))
    trainer = baseModel(net, cfg["train"], batch, out_dir=run_dir)
    trainer.fit(Xf, d["yf"], Xc, d["yc"])
    T = trainer.calibrate(Xc, d["yc"])

    if args.test_init != "carried" and tan is not None and tan.ref_mode == "running":
        if args.test_init == "fitmean":
            tan.freeze_reference(torch.as_tensor(Xf).to(DEVICE))
        else:
            tan.primed.fill_(0)
    p = p_raw = trainer.predict_proba(Xt)
    if args.smooth:
        A = transition_matrix(d["yf"], strm_f)
        pi = np.bincount(d["yf"], minlength=2).astype(float)
        pi /= pi.sum()
        p = p_raw.copy()
        for st in strm_t:
            p[st] = forward_filter(p_raw[st], A, pi)
    pred = p.argmax(1)

    vt = d["vt"]
    res = {"acc": balanced_accuracy(d["yt"], pred),
           "ece": expected_calibration_error(p, d["yt"]),
           "cond_r2": (conditional_r2(pred[vt, None].astype(float), d["imu_t"][vt],
                                      d["yt"][vt], d["st"][vt])
                       if vt.sum() > 40 else float("nan")),
           **onset_metrics(d["yt"], pred, strm_t),
           "val_acc": trainer.best_val}
    if args.smooth:
        res["acc_unsmoothed"] = balanced_accuracy(d["yt"], p_raw.argmax(1))
        res["ece_unsmoothed"] = expected_calibration_error(p_raw, d["yt"])
    if run_dir is not None:
        trainer.save(f"{arm.replace(':', '_')}_s{seed}_{d['sub']}", results=res,
                     y_true=d["yt"], y_pred=pred, classes=tuple(cfg["data"]["classes_display"]))
    if args.dump:
        pi_f = np.bincount(d["yf"], minlength=2).astype(float)
        res["_dump"] = dict(p=p_raw, y=d["yt"], sid=sid(strm_t, len(d["yt"])),
                            pc=softmax_np(trainer.cal_logits, T), yc=d["yc"],
                            cid=sid(strm_c, len(d["yc"])), gc=d["gc"],
                            A=transition_matrix(d["yf"], strm_f), pi=pi_f / pi_f.sum())
    del trainer, net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arms", default="stem,stem_tangent",
                    help="stem, stem_tangent, stem_wide or bd:<braindecode model>")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=None, help="windows per reference update (N)")
    ap.add_argument("--momentum", type=float, default=None, help="reference update rate (m)")
    ap.add_argument("--ref-mode", default="running", choices=["running", "frozen", "none"])
    ap.add_argument("--causal-update", action="store_true")
    ap.add_argument("--causal-filter", action="store_true")
    ap.add_argument("--test-init", default="carried", choices=["carried", "fitmean", "firstbatch"])
    ap.add_argument("--smooth", action="store_true", help="causal output filter")
    ap.add_argument("--ica", action="store_true", help="cohort A only")
    ap.add_argument("--no-extra-trials", action="store_true", help="cohort A only")
    ap.add_argument("--dump", action="store_true", help="save per-window posteriors")
    ap.add_argument("--no-save", action="store_true", help="skip model.pth and plots")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    args.out = args.out or f"{cfg['cohort']}_{'_'.join(args.arms.split(','))}.json"
    out_path = RESULTS / args.out
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    list_subjects, loader = LOADERS[cfg["cohort"]]
    load = lambda sub, seed: dict(loader(cfg, sub, seed, causal=args.causal_filter, ica=args.ica,
                                         extra_trials=not args.no_extra_trials), sub=sub)

    D = {}
    for sub in list_subjects(cfg):
        try:
            D[sub] = load(sub, seeds[0])
        except Exception as e:
            print("  [skip] %s: %s" % (sub, e), flush=True)
    print("cohort %s: %d participants, %s" % (cfg["cohort"], len(D), DEVICE), flush=True)

    for arm in args.arms.split(","):
        for seed in seeds:
            key = "%s|s%d" % (arm, seed)
            if key in out:
                continue
            t0, rows, done = time.time(), [], []
            for sub, d in D.items():
                if seed != seeds[0]:
                    d = load(sub, seed)
                r = run(d, arm, seed, cfg, args,
                        None if args.no_save else RESULTS / "runs" / out_path.stem)
                dump = r.pop("_dump", None)
                if dump is not None:
                    dd = RESULTS / "dump" / out_path.stem
                    dd.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dd / f"{arm}_s{seed}_{sub}.npz", **dump)
                rows.append(r)
                done.append(sub)
                print("    %s %s seed=%d acc %.3f" % (sub, arm, seed, r["acc"]), flush=True)
            agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
            agg["acc_sd"] = float(np.std([r["acc"] for r in rows]))
            agg["per_subject"] = {s: {k: float(v) for k, v in r.items()}
                                  for s, r in zip(done, rows)}
            out[key] = agg
            print("  %s seed=%d  acc %.3f  ece %.3f  FA/min %.2f  (%.0f s)"
                  % (arm, seed, agg["acc"], agg["ece"], agg["false_onsets_per_min"],
                     time.time() - t0), flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
