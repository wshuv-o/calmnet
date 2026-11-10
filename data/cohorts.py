"""Loaders for cohorts A-E. Each returns fit (f), held-out (c) and test (t) splits
with X, y, s (recording), t (task) and g (segment). Data root: $CALMNET_DATA or ./datasets.
"""
import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np

from .preprocess import (bandpass_ba_zero_phase, bandpass_causal,
                         bandpass_zero_phase, hold_resample)
from .splits import grouped_split

DATA_ROOT = Path(os.environ.get("CALMNET_DATA",
                                Path(__file__).resolve().parent.parent / "datasets"))
CACHE = DATA_ROOT / "cache"


def _pack(fit, ti, ci, test, s_key, t_fn, imu_t=None, vt=None):
    n_t = len(test["y"])
    return {"Xf": fit["X"][ti], "yf": fit["y"][ti], "sf": fit[s_key][ti],
            "tf": t_fn(fit, ti), "gf": fit["segment"][ti],
            "Xc": fit["X"][ci], "yc": fit["y"][ci], "sc": fit[s_key][ci],
            "tc": t_fn(fit, ci), "gc": fit["segment"][ci],
            "Xt": test["X"], "yt": test["y"], "st": test[s_key],
            "tt": t_fn(test, np.arange(n_t)), "gt": test["segment"],
            "imu_t": np.zeros((n_t, 1), np.float32) if imu_t is None else imu_t,
            "vt": np.zeros(n_t, bool) if vt is None else vt}


def _subset(es, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(es["y"]) else v)
            for k, v in es.items()}


# ---------------------------------------------------------------- cohort A --
A_STATES = {"x81": 1, "x0": 0}
A_TRIALS = tuple("trial%02d" % i for i in range(1, 13))
A_ICA_DROP = {"eye blink", "muscle artifact", "heart beat", "line noise",
              "channel noise"}
_EOG_BETAS = {}


def a_subjects(cfg):
    return ["sub-%02d" % i for i in range(1, 8)]


def a_sessions(root, sub):
    return sorted(int(p.name.split("-")[1]) for p in (root / sub).glob("ses-*")
                  if p.is_dir())


def _a_ica_clean(raw, tag, log_path):
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    fit = raw.copy().filter(1.0, 45.0, picks="eeg", verbose="ERROR")
    n = min(30, len(mne.pick_types(raw.info, eeg=True)) - 1)
    ica = ICA(n_components=n, method="picard",
              fit_params=dict(ortho=False, extended=True),
              max_iter="auto", random_state=0, verbose="ERROR")
    ica.fit(fit, picks="eeg", verbose="ERROR")
    labels = list(label_components(fit, ica, method="iclabel")["labels"])
    ica.exclude = [i for i, l in enumerate(labels) if l in A_ICA_DROP]
    ica.apply(raw, verbose="ERROR")
    new = not log_path.exists()
    with open(log_path, "a", encoding="utf-8") as fh:
        if new:
            fh.write("recording,n_components,n_removed,removed_labels\n")
        fh.write("%s,%d,%d,%s\n" % (tag, n, len(ica.exclude),
                                    ";".join(labels[i] for i in ica.exclude)))
    return raw


def _a_causal_eog_betas(root, sub, band):
    """Causal chain: EOG coefficients from the first-session training recording only."""
    import mne
    if sub not in _EOG_BETAS:
        ses = a_sessions(root, sub)[0]
        ref = (root / sub / f"ses-{ses:02d}" / "eeg"
               / f"{sub}_ses-{ses:02d}_task-training_eeg.edf")
        r = mne.io.read_raw_edf(ref, preload=True, verbose="ERROR")
        r.set_channel_types({c: "eog" for c in r.ch_names if "EOG" in c.upper()})
        r.filter(band[0], band[1], picks=["eeg", "eog"], method="fir",
                 phase="minimum", fir_design="firwin", pad="edge", verbose="ERROR")
        _, b = mne.preprocessing.regress_artifact(
            r, picks="eeg", picks_artifact="eog", copy=True, verbose="ERROR")
        _EOG_BETAS[sub] = b
    return _EOG_BETAS[sub]


def _a_raw(root, sub, edf, band, causal, ica, log_path):
    import mne
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]
    raw.set_channel_types({c: "eog" for c in eog})
    eog_regress = True
    if ica:
        raw = _a_ica_clean(raw, edf.stem, log_path)
        eog_regress = False
    if causal:
        raw.filter(band[0], band[1], picks=["eeg", "eog"], method="fir",
                   phase="minimum", fir_design="firwin", pad="edge", verbose="ERROR")
    else:
        raw.filter(band[0], band[1], picks=["eeg", "eog"], method="fir",
                   phase="zero", fir_design="firwin", verbose="ERROR")
    if eog_regress and eog:
        try:
            betas = _a_causal_eog_betas(root, sub, band) if causal else None
            raw, _ = mne.preprocessing.regress_artifact(
                raw, picks="eeg", picks_artifact="eog", betas=betas, copy=True,
                verbose="ERROR")
        except Exception as e:
            if causal:
                print(f"  [eog] {edf.name}: causal regression failed ({e})", flush=True)
    raw.pick("eeg")
    return raw


def _a_imu(root, sub, ses, task, tracksys):
    import pandas as pd
    mdir = root / sub / f"ses-{ses:02d}" / "motion"
    stem = f"{sub}_ses-{ses:02d}_task-{task}_tracksys-{tracksys}_motion"
    tsv, js = mdir / (stem + ".tsv"), mdir / (stem + ".json")
    if not tsv.exists():
        return None
    try:
        data = pd.read_csv(tsv, sep="\t")

        def mag(pattern):
            cols = [c for c in data.columns if re.search(pattern, str(c), re.I)]
            if not cols:
                return None
            v = data[cols].astype(float).to_numpy()
            v = v - v.mean(axis=0, keepdims=True)
            return np.sqrt((v ** 2).sum(axis=1))
        acc, gyr = mag(r"acc"), mag(r"gyro")
        if acc is None:
            return None
        if gyr is None:
            gyr = np.zeros_like(acc)
        sf = float(json.loads(js.read_text()).get("SamplingFrequency", 100.0)) \
            if js.exists() else 100.0
        return acc, gyr, sf
    except Exception:
        return None


def _a_imu_feats(imu, a, b, sf):
    if imu is None:
        return [0.0] * 6
    acc, gyr, msf = imu
    ma, mb = int(a / sf * msf), int(b / sf * msf)
    out = []
    for series in (acc, gyr):
        s = series[ma:mb]
        out += [float(np.mean(s)), float(np.std(s)), float(np.max(s))] if len(s) else [0.0] * 3
    return out


def a_epochs(cfg, sub, sessions, tasks, causal=False, ica=False):
    import pandas as pd
    dc = cfg["data"]
    root = DATA_ROOT / dc["dir"]
    band, win, step, margin = tuple(dc["band"]), dc["window_s"], dc["step_s"], dc["margin_s"]
    tag = (f"{sub}_s{'-'.join(map(str, sessions))}_{'-'.join(tasks)}_w{win}_st{step}"
           f"_{float(band[0])}-{float(band[1])}_z0_v2imu")
    tag += "_ica" if ica else ""
    tag += "_causal" if causal else ""
    cf = CACHE / "A" / f"{tag}.npz"
    if cf.exists():
        with np.load(cf, allow_pickle=True) as d:
            return {k: d[k] for k in ("X", "y", "session", "task", "segment", "imu_feats")}

    X, y, ses_l, task_l, seg_l, feats = [], [], [], [], [], []
    seg_base = 0
    for ses in sessions:
        eeg_dir = root / sub / f"ses-{ses:02d}" / "eeg"
        for task in tasks:
            edf = eeg_dir / f"{sub}_ses-{ses:02d}_task-{task}_eeg.edf"
            evt = eeg_dir / f"{sub}_ses-{ses:02d}_task-{task}_acq-rexstate_events.tsv"
            if not (edf.exists() and evt.exists()):
                continue
            ev = pd.read_csv(evt, sep="\t")
            seg = [(float(r["onset"]), float(r["duration"]), A_STATES[str(r["trial_type"])])
                   for _, r in ev.iterrows()
                   if str(r["trial_type"]) in A_STATES and float(r["duration"]) > 0]
            if not seg:
                continue
            raw = _a_raw(root, sub, edf, band, causal, ica,
                         CACHE.parent / "ica_components.csv")
            data = raw.get_data()
            sf = round(raw.info["sfreq"])
            wlen, wstep, mrg = round(win * sf), round(step * sf), round(margin * sf)
            head = _a_imu(root, sub, ses, task, "head")
            exo = _a_imu(root, sub, ses, task, "exo")
            n0 = len(y)
            for gi, (onset, dur, lab) in enumerate(seg):
                s0 = int(onset * sf) + mrg
                s1 = min(int((onset + dur) * sf), data.shape[1])
                for a in range(s0, s1 - wlen + 1, wstep):
                    w = data[:, a:a + wlen]
                    if w.shape[1] != wlen:
                        continue
                    X.append(w); y.append(lab); seg_l.append(seg_base + gi)
                    feats.append(_a_imu_feats(head, a, a + wlen, sf)
                                 + _a_imu_feats(exo, a, a + wlen, sf))
            seg_base += len(seg)
            ses_l += [ses] * (len(y) - n0)
            task_l += [task] * (len(y) - n0)
    es = {"X": np.asarray(X, np.float32), "y": np.asarray(y, np.int64),
          "session": np.asarray(ses_l), "task": np.asarray(task_l, dtype=object),
          "segment": np.asarray(seg_l, np.int64),
          "imu_feats": np.asarray(feats, np.float32)}
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cf, **es)
    return es


def _imu_valid(imu_feats, session):
    valid = np.ones(len(imu_feats), bool)
    for s in np.unique(session):
        m = session == s
        if not imu_feats[m].any():
            valid[m] = False
    return valid


def load_a(cfg, sub, seed, causal=False, ica=False, extra_trials=True):
    """Fit sessions 1-3 (+ trial recordings), test sessions 4-9."""
    root = DATA_ROOT / cfg["data"]["dir"]
    n_fit = cfg["split"]["n_fit_sessions"]
    es = a_epochs(cfg, sub, a_sessions(root, sub), ("training",), causal, ica)
    valid = _imu_valid(es["imu_feats"], es["session"])
    present = sorted(set(int(v) for v in np.unique(es["session"])))
    sess = [s for s in a_sessions(root, sub) if s in present]
    tr = np.isin(es["session"], sess[:n_fit])
    fit, test = _subset(es, tr), _subset(es, ~tr)
    ti, ci = grouped_split(fit["segment"], fit["y"], frac=cfg["split"]["heldout_frac"],
                           seed=seed)
    d = _pack(fit, ti, ci, test, "session", lambda e, i: e["task"][i],
              imu_t=test["imu_feats"], vt=valid[~tr])
    if extra_trials:
        ex = a_epochs(cfg, sub, sess[:n_fit], A_TRIALS, causal, ica)
        for k, src in (("Xf", "X"), ("yf", "y"), ("sf", "session"), ("tf", "task"),
                       ("gf", "segment")):
            d[k] = np.concatenate([d[k], ex[src]])
    return d


# ---------------------------------------------------------------- cohort B --
def b_subjects(cfg):
    root = DATA_ROOT / cfg["data"]["dir"]
    if not root.is_dir():
        return []
    return sorted({p.name.split("-")[0] for p in root.glob("SL*-T*") if p.is_dir()})


def _b_walk_bounds(trial_dir):
    """Walking onset / offset (conductor codes 8 / 10)."""
    on = off = None
    cond = trial_dir / "conductor.txt"
    if cond.exists():
        ev = []
        for l in cond.read_text(errors="replace").splitlines()[2:]:
            q = l.split()
            if len(q) >= 2:
                try:
                    ev.append((float(q[0]), int(float(q[1]))))
                except ValueError:
                    pass
        t8 = [t for t, c in ev if c == 8]
        t10 = [t for t, c in ev if c == 10]
        if t8:
            on = min(t8)
        cand = [t for t in t10 if on is None or t > on + 60]
        if cand:
            off = max(cand)
    return on, off


def b_epochs(cfg, sub, causal=False):
    """Stand / walk / stand windows, grouped in 20 s chunks for the held-out split."""
    dc = cfg["data"]
    root = DATA_ROOT / dc["dir"]
    sf, band = float(dc["sfreq"]), tuple(dc["band"])
    win, step, margin, chunk_s = dc["window_s"], dc["step_s"], dc["margin_s"], dc["chunk_s"]
    tag = (f"{sub}_w{win}_s{step}_{float(band[0])}-{float(band[1])}_c{int(chunk_s)}_ts_z0"
           + ("_causal" if causal else ""))
    cf = CACHE / "B" / f"{tag}.npz"
    if cf.exists():
        with np.load(cf, allow_pickle=True) as d:
            return {k: d[k] for k in ("X", "y", "trial", "segment")}
    wlen, wstep, mrg, chunk = int(win * sf), int(step * sf), int(margin * sf), int(chunk_s * sf)
    X, y, tr, seg = [], [], [], []
    segid = 0
    for t_idx in (1, 2, 3):
        d = root / f"{sub}-T{t_idx:02d}"
        if not (d / "eeg.txt").exists():
            continue
        E = np.loadtxt(d / "eeg.txt", skiprows=1, dtype=np.float32)
        J = np.loadtxt(d / "joints.txt", skiprows=2, dtype=np.float32)
        te, eeg = E[:, 0], E[:, 1:].T
        n = min(eeg.shape[1], len(J))
        eeg, te = eeg[:, :n], te[:n]
        on, off = _b_walk_bounds(d)
        if on is None or off is None:
            continue
        eeg = (bandpass_causal(eeg, sf, band) if causal
               else bandpass_zero_phase(eeg, sf, band)).astype(np.float32)
        for b0, b1, lab in ((0.0, on, 0), (on, off, 1), (off, te[-1], 0)):
            a0, a1 = int(b0 * sf) + mrg, int(b1 * sf) - mrg
            if a1 - a0 < wlen:
                continue
            for a in range(a0, a1 - wlen + 1, wstep):
                w = eeg[:, a:a + wlen]
                if w.shape[1] != wlen:
                    continue
                c0 = (a - a0) // chunk
                if (a - a0 + wlen - 1) // chunk != c0:
                    continue
                X.append(w); y.append(lab); tr.append(t_idx); seg.append(segid + int(c0))
            segid += int((a1 - a0) // chunk) + 1
    if not X:
        return None
    X = np.asarray(X, dtype=np.float32)
    X = (X / (float(X.std()) + 1e-30)).astype(np.float32)
    es = {"X": X, "y": np.asarray(y, np.int64), "trial": np.asarray(tr, np.int64),
          "segment": np.asarray(seg, np.int64)}
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cf, **es)
    return es


def load_b(cfg, sub, seed, causal=False, **_):
    """Fit trial 1, test trials 2-3."""
    es = b_epochs(cfg, sub, causal)
    if es is None or len(es["y"]) < 100:
        raise RuntimeError("no MoBI data for %s" % sub)
    fit = _subset(es, np.isin(es["trial"], cfg["split"]["fit_trials"]))
    test = _subset(es, np.isin(es["trial"], cfg["split"]["test_trials"]))
    ti, ci = grouped_split(fit["segment"], fit["y"], frac=cfg["split"]["heldout_frac"],
                           seed=seed)
    return _pack(fit, ti, ci, test, "trial",
                 lambda e, i: np.array(["mobi"] * len(i), object))


# ---------------------------------------------------------------- cohort C --
def c_subjects(cfg):
    root = DATA_ROOT / cfg["data"]["dir"]
    sp = cfg["split"]
    return ["S%03d" % i for i in range(1, cfg["data"]["n_subjects"] + 1)
            if all((root / ("S%03d" % i) / ("S%03dR%02d.edf" % (i, r))).exists()
                   for r in sp["fit_runs"] + sp["test_runs"])]


def _c_run(cfg, sub, run, causal):
    """Windows wholly inside one rest (T0) or movement (T1/T2) annotation."""
    import mne
    dc = cfg["data"]
    sf, band = float(dc["sfreq"]), tuple(dc["band"])
    raw = mne.io.read_raw_edf(DATA_ROOT / dc["dir"] / sub / ("%sR%02d.edf" % (sub, run)),
                              preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    if causal:
        fs_in = raw.info["sfreq"]
        data = hold_resample(bandpass_causal(raw.get_data(), fs_in, band),
                             fs_in, sf).astype(np.float32)
    else:
        raw.resample(sf, verbose="ERROR")
        data = bandpass_zero_phase(raw.get_data(), sf, band).astype(np.float32)
    wlen, wstep = int(dc["window_s"] * sf), int(dc["step_s"] * sf)
    X, y, blk = [], [], []
    for bi, a in enumerate(raw.annotations):
        desc = str(a["description"])
        if desc not in ("T0", "T1", "T2"):
            continue
        s0 = int(round(a["onset"] * sf))
        s1 = int(round((a["onset"] + a["duration"]) * sf))
        for st in range(s0, s1 - wlen + 1, wstep):
            if st + wlen > data.shape[1]:
                break
            X.append(data[:, st:st + wlen]); y.append(0 if desc == "T0" else 1); blk.append(bi)
    return (np.stack(X) if X else np.empty((0, data.shape[0], wlen), np.float32),
            np.asarray(y, np.int64), np.asarray(blk, np.int64))


def load_c(cfg, sub, seed, causal=False, **_):
    """Fit runs 3/5/7/9, test runs 11/13."""
    parts, base = {}, 0
    for split, runs in (("fit", cfg["split"]["fit_runs"]), ("test", cfg["split"]["test_runs"])):
        Xs, ys, rs, gs = [], [], [], []
        for r in runs:
            X, y, blk = _c_run(cfg, sub, r, causal)
            if len(y) == 0:
                continue
            Xs.append(X); ys.append(y); rs.append(np.full(len(y), r, np.int64))
            gs.append(blk + base)
            base += int(blk.max()) + 1
        parts[split] = {"X": np.concatenate(Xs), "y": np.concatenate(ys),
                        "run": np.concatenate(rs), "segment": np.concatenate(gs)}
    fit, test = parts["fit"], parts["test"]
    ti, ci = grouped_split(fit["segment"], fit["y"], frac=cfg["split"]["heldout_frac"],
                           seed=seed)
    ti, ci = np.sort(ti), np.sort(ci)
    return _pack(fit, ti, ci, test, "run",
                 lambda e, i: np.array(["exec"] * len(i), object))


# ---------------------------------------------------------------- cohort D --
def d_subjects(cfg):
    return [str(s) for s in range(1, 10)]


def load_d(cfg, sub, seed, causal=False, **_):
    """Fit session 1, test session 2, one 4 s window per trial (moabb download)."""
    import mne
    from moabb.datasets import BNCI2014_001
    dc = cfg["data"]
    sf, band, win = float(dc["sfreq"]), tuple(dc["band"]), dc["window_s"]
    classes = tuple(dc["classes"])
    data = BNCI2014_001().get_data(subjects=[int(sub)])[int(sub)]
    X, y, sess, seg = [], [], [], []
    for si, (_, runs) in enumerate(sorted(data.items())):
        for _, raw in sorted(runs.items()):
            r = raw.copy().pick(mne.pick_types(raw.info, eeg=True, eog=False))
            if causal:
                fs_in = r.info["sfreq"]
                x = hold_resample(bandpass_causal(r.get_data(), fs_in, band), fs_in, sf)
                out = mne.io.RawArray(x, mne.create_info(r.ch_names, sf, "eeg"),
                                      first_samp=0, verbose=False)
                ann = r.annotations
                out.set_annotations(mne.Annotations(ann.onset - r.first_time,
                                                    ann.duration, ann.description))
                r = out
            else:
                r.filter(band[0], band[1], method="iir",
                         iir_params=dict(order=4, ftype="butter"), verbose=False)
                r.resample(sf, verbose=False)
            ev, eid = mne.events_from_annotations(r, verbose=False)
            keep = {k: v for k, v in eid.items() if k in classes}
            if not keep:
                continue
            ep = mne.Epochs(r, ev, event_id=keep, tmin=0.0, tmax=win - 1.0 / sf,
                            baseline=None, preload=True, verbose=False,
                            on_missing="ignore")
            if len(ep) == 0:
                continue
            inv = {v: k for k, v in keep.items()}
            ys = np.array([0 if inv[v] == classes[0] else 1 for v in ep.events[:, -1]], int)
            X.append(ep.get_data(copy=True).astype(np.float32)); y.append(ys)
            sess.append(np.full(len(ys), si, int))
            seg.append(np.arange(len(ys)) + 1000 * len(seg))
    if not X:
        raise RuntimeError("no trials for subject %s" % sub)
    es = {"X": np.concatenate(X)[:, :, :int(round(win * sf))], "y": np.concatenate(y),
          "session": np.concatenate(sess), "segment": np.concatenate(seg)}
    fit, test = _subset(es, es["session"] == 0), _subset(es, es["session"] == 1)
    if len(fit["y"]) < 40 or len(test["y"]) < 40:
        raise RuntimeError("too few trials for %s" % sub)
    ti, ci = grouped_split(fit["segment"], fit["y"], frac=cfg["split"]["heldout_frac"],
                           seed=seed)
    ti, ci = np.sort(ti), np.sort(ci)
    return _pack(fit, ti, ci, test, "session",
                 lambda e, i: np.array(["mi"] * len(i), object))


# ---------------------------------------------------------------- cohort E --
E_CODES = {402: 0, 404: 1}
_ZIP = {}


def _e_zip(cfg):
    path = os.environ.get("DECODED_ZIP", str(DATA_ROOT / cfg["data"]["dir"] / "dataset.zip"))
    if path not in _ZIP:
        _ZIP[path] = zipfile.ZipFile(path)
    return _ZIP[path]


def _e_runs(cfg, sub):
    """Condition-3 runs sorted by run number. subject_01 has two dates with the
    same run numbers, so its dates interleave; kept as in the reported runs."""
    p = cfg["data"]["protocol"]
    pat = re.compile(r"%s/%s/CSV/(?:\d{8}/)?%s_cond_03_run_(\d+)_EEG\.csv$" % (p, sub, sub))
    hits = [(int(m.group(1)), n) for n in _e_zip(cfg).namelist() if (m := pat.match(n))]
    return [n for _, n in sorted(hits)]


def e_subjects(cfg):
    pat = re.compile(r"%s/(subject_\d+)/" % cfg["data"]["protocol"])
    names = {m.group(1) for n in _e_zip(cfg).namelist() if (m := pat.match(n))}
    return sorted(s for s in names if len(_e_runs(cfg, s)) >= 8)


def _e_read_run(cfg, name, causal):
    dc = cfg["data"]
    sf_in, sf, band = float(dc["sfreq_in"]), float(dc["sfreq"]), tuple(dc["band"])
    lines = _e_zip(cfg).read(name).decode("utf-8", "replace").strip().split("\n")
    head = lines[0].strip().split(";")
    cols = {c: i for i, c in enumerate(head)}
    eeg_names = [c for c in head if c not in ("time", "task", "HR", "HL", "VU", "VD")]
    if "task" not in cols or not eeg_names:
        return None, None
    idx, ti = [cols[c] for c in eeg_names], cols["task"]
    data = np.empty((len(lines) - 1, len(idx) + 1), np.float64)
    keep = 0
    for ln in lines[1:]:
        p = ln.split(";")
        if len(p) <= ti:
            continue
        try:
            data[keep, :-1] = [float(p[i]) for i in idx]
            data[keep, -1] = float(p[ti] or 0)
        except ValueError:
            continue
        keep += 1
    if keep < int(5 * sf_in):
        return None, None
    X, task = data[:keep, :-1].T, data[:keep, -1]
    X = bandpass_causal(X, sf_in, band) if causal else bandpass_ba_zero_phase(X, sf_in, band)
    step = int(round(sf_in / sf))
    return X[:, ::step].astype(np.float32), task[::step].astype(int)


def _e_windows(X, task, n, hop):
    xs, ys, segs = [], [], []
    change = np.flatnonzero(np.diff(task) != 0) + 1
    bounds = np.concatenate([[0], change, [len(task)]])
    for si in range(len(bounds) - 1):
        a, b = bounds[si], bounds[si + 1]
        code = int(task[a])
        if code not in E_CODES or (b - a) < n:
            continue
        for s in range(a, b - n + 1, hop):
            xs.append(X[:, s:s + n]); ys.append(E_CODES[code]); segs.append(si)
    if not xs:
        return None, None, None
    return np.stack(xs), np.array(ys, int), np.array(segs, int)


def load_e(cfg, sub, seed, causal=False, **_):
    """First half of the runs fit, second half test."""
    dc = cfg["data"]
    n, hop = int(round(dc["window_s"] * dc["sfreq"])), max(1, int(round(dc["step_s"] * dc["sfreq"])))
    X, y, sess, seg = [], [], [], []
    base = 0
    for ri, run in enumerate(_e_runs(cfg, sub)):
        sig, tk = _e_read_run(cfg, run, causal)
        if sig is None:
            continue
        xs, ys, sg = _e_windows(sig, tk, n, hop)
        if xs is None:
            continue
        X.append(xs); y.append(ys); sess.append(np.full(len(ys), ri, int))
        seg.append(sg + base)
        base = int(sg.max()) + 1 + base
    if not X:
        raise RuntimeError("no usable runs for %s" % sub)
    es = {"X": np.concatenate(X), "y": np.concatenate(y),
          "session": np.concatenate(sess), "segment": np.concatenate(seg)}
    nr = int(es["session"].max()) + 1
    fit = _subset(es, np.isin(es["session"], list(range(nr))[:nr // 2]))
    test = _subset(es, np.isin(es["session"], list(range(nr))[nr // 2:]))
    if len(fit["y"]) < 60 or len(test["y"]) < 60:
        raise RuntimeError("too few windows for %s" % sub)
    ti, ci = grouped_split(fit["segment"], fit["y"], frac=cfg["split"]["heldout_frac"],
                           seed=seed)
    ti, ci = np.sort(ti), np.sort(ci)
    return _pack(fit, ti, ci, test, "session",
                 lambda e, i: np.array(["exo"] * len(i), object))


LOADERS = {"A": (a_subjects, load_a), "B": (b_subjects, load_b),
           "C": (c_subjects, load_c), "D": (d_subjects, load_d),
           "E": (e_subjects, load_e)}
