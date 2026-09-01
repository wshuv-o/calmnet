"""Representation sweep: does feature engineering move the honest number?

The architecture sweep varied the model and held the representation fixed. This
does the opposite -- fixed simple classifier, varied representation -- so the
two axes can be compared on equal terms.

Grid: preprocessing x feature extractor.

  preprocessing   raw | CAR | surface Laplacian | ASR | ASR+Laplacian
  features        log band-power | FBCSP | Riemannian tangent (+Euclidean
                  Alignment) | PLV (phase) | tangent+PLV

Classifier is regularised logistic regression throughout -- deliberately weak,
so differences are attributable to the representation rather than to model
capacity. Everything is CPU-only and runs while the GPU is busy elsewhere.

Scored exactly like the architectures: balanced accuracy AND intent->motion R^2.
A representation can buy accuracy by admitting more movement just as an
architecture can, and ASR in particular is expected to trade one for the other.

Writes results/features.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions, DATA_ROOT
from splits import grouped_split
from abstain import balanced_accuracy
from calmnet_msa import imu_valid_mask
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "features.json"
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3

PREPS = ["raw", "car", "laplacian", "asr", "asr+laplacian"]
FEATS = ["bandpower", "fbcsp", "tangent_ea", "plv", "tangent+plv"]


def electrode_pos(ch_names, subject="sub-01"):
    import pandas as pd
    tsv = DATA_ROOT / subject / "ses-01" / "eeg" / f"{subject}_ses-01_electrodes.tsv"
    df = pd.read_csv(tsv, sep="\t")
    pos = {str(r["name"]): np.array([r["x"], r["y"], r["z"]], float)
           for _, r in df.iterrows()}
    return np.stack([pos.get(c, np.zeros(3)) for c in ch_names])


def load(sub):
    es = build_epochs(subject=sub)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, vi = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=0)
    return {"Xf": es.X[tr][ti], "yf": es.y[tr][ti], "Mf": es.imu_feats[tr][ti],
            "vf": valid[tr][ti], "Xt": es.X[~tr], "yt": es.y[~tr],
            "Mt": es.imu_feats[~tr], "vt": valid[~tr],
            "motf": es.motion[tr][ti], "ch": es.ch_names}


def preprocess(kind, Xf, Xt, pos):
    if kind == "raw":
        return Xf, Xt
    if kind == "car":
        return FE.car(Xf), FE.car(Xt)
    if kind == "laplacian":
        return FE.laplacian(Xf, pos), FE.laplacian(Xt, pos)
    if kind.startswith("asr"):
        # calibrate on the quietest third of the FIT windows only
        q = np.quantile(Xf.std(axis=(1, 2)), 0.33)
        cal = Xf[Xf.std(axis=(1, 2)) <= q]
        if len(cal) < 5:
            cal = Xf[:50]
        asr = FE.fit_asr(cal)
        a, b = FE.apply_asr(asr, Xf), FE.apply_asr(asr, Xt)
        if kind.endswith("laplacian"):
            a, b = FE.laplacian(a, pos), FE.laplacian(b, pos)
        return a, b
    raise ValueError(kind)


def extract(kind, Xf, yf, Xt):
    if kind == "bandpower":
        return FE.log_var(FE.bandpass(Xf, 8, 30)), FE.log_var(FE.bandpass(Xt, 8, 30))
    if kind == "fbcsp":
        return FE.fbcsp(Xf, yf, Xt)
    if kind == "tangent_ea":
        Cf, Ct = FE.covariances(Xf), FE.covariances(Xt)
        return FE.tangent(FE.euclidean_align(Cf)), FE.tangent(FE.euclidean_align(Ct))
    if kind == "plv":
        return FE.plv_features(Xf), FE.plv_features(Xt)
    if kind == "tangent+plv":
        Cf, Ct = FE.covariances(Xf), FE.covariances(Xt)
        return (np.concatenate([FE.tangent(FE.euclidean_align(Cf)), FE.plv_features(Xf)], 1),
                np.concatenate([FE.tangent(FE.euclidean_align(Ct)), FE.plv_features(Xt)], 1))
    raise ValueError(kind)


def run_cell(prep, feat, D, pos_by_sub):
    accs, r2s = [], []
    for sub, d in D.items():
        Xf, Xt = preprocess(prep, d["Xf"], d["Xt"], pos_by_sub[sub])
        Ff, Ft = extract(feat, Xf, d["yf"], Xt)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=0.1,
                                               class_weight="balanced"))
        clf.fit(Ff, d["yf"])
        accs.append(balanced_accuracy(d["yt"], clf.predict(Ft)))
        # invariance probe on the FEATURES themselves
        vf, vt = d["vf"], d["vt"]
        if vf.sum() > 20 and vt.sum() > 20:
            mu, sd = d["Mf"][vf].mean(0), d["Mf"][vf].std(0) + 1e-6
            est = Ridge(alpha=1.0).fit(Ff[vf], (d["Mf"][vf] - mu) / sd)
            r2s.append(r2_score((d["Mt"][vt] - mu) / sd, est.predict(Ft[vt]),
                                multioutput="variance_weighted"))
    acc, r2 = float(np.mean(accs)), float(np.nanmean(r2s)) if r2s else float("nan")
    return {"acc": acc, "r2": r2, "score": acc - max(0.0, r2),
            "n_leak": int(sum(1 for r in r2s if r > 0))}


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("loading subjects ...", flush=True)
    D, pos = {}, {}
    for s in SUBJECTS:
        D[s] = load(s)
        pos[s] = electrode_pos(D[s]["ch"], s)
    print(f"{len(D)} subjects\n", flush=True)

    for prep in PREPS:
        for feat in FEATS:
            key = f"{prep}|{feat}"
            if key in out and "acc" in out[key]:
                continue
            t0 = time.time()
            try:
                r = run_cell(prep, feat, D, pos)
                r["secs"] = round(time.time() - t0, 1)
                out[key] = r
                print(f"  {key:26} acc {r['acc']:.3f}  R2 {r['r2']:+.3f}  "
                      f"score {r['score']:.3f}  leak {r['n_leak']}/7  {r['secs']:.0f}s",
                      flush=True)
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {key:26} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
            OUT.write_text(json.dumps(out, indent=2))

    ok = {k: v for k, v in out.items() if "acc" in v}
    if not ok:
        return
    print("\n" + "=" * 78)
    print("REPRESENTATION SWEEP  (fixed logistic regression; only features vary)")
    print("=" * 78)
    print(f"{'preprocessing | features':28}{'acc':>8}{'R2':>9}{'score':>9}{'leak':>7}")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["score"]):
        print(f"{k:28}{v['acc']:8.3f}{v['r2']:+9.3f}{v['score']:9.3f}{v['n_leak']:5d}/7")
    best = max(ok, key=lambda k: ok[k]["score"])
    adm = {k: v for k, v in ok.items() if v["r2"] <= 0}
    print("-" * 78)
    print(f"best by honest score : {best}  ({ok[best]['acc']:.3f}, R2 {ok[best]['r2']:+.3f})")
    if adm:
        ba = max(adm, key=lambda k: adm[k]["acc"])
        print(f"best with R2 <= 0    : {ba}  ({adm[ba]['acc']:.3f}, R2 {adm[ba]['r2']:+.3f})")
    else:
        print("best with R2 <= 0    : none")
    print("\nreference: deep composed architecture = 0.695 +/- 0.024 at R2 +0.073")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
