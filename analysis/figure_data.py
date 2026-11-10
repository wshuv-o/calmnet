"""Data behind the figures: result-file readers and the CPU-only computations they need."""
import itertools
import json
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml

from figstyle import RESULTS, ROOT

CACHE = RESULTS / "figure_data"
SEEDS = (0, 1, 2)
BATCH, MOM = 32, 0.2           # reported N and m, mu = N / m = 160 windows
COHORTS = ("C", "D", "E", "A", "B")

# submitted runs per cohort: branch file, stem file
OFFLINE = {"C": ("c_stem_tangent_3seed.json", "c_stem_tangent_3seed.json"),
           "D": ("d_bnci_3seed.json", "d_bnci_3seed.json"),
           "E": ("e_decoded_3seed.json", "e_decoded_3seed.json"),
           "A": ("a_tangent_3seed.json", "a_ablation_s12.json"),
           "B": ("b_tangent_3seed.json", "b_tangent_3seed.json")}
# memory sweep, submitted runs: cohort, N, m, branch file, stem file (matched on N)
SWEEP = [("C", 32, 0.2, "c_stem_tangent_3seed.json", "c_stem_tangent_3seed.json"),
         ("C", 8, 1.0, "c_boundary_mu8.json", "c_boundary_mu8.json"),
         ("C", 4, 0.25, "c_fixedN_N4_m0.25.json", "c_fixedN_N4_m0.25.json"),
         ("C", 4, 1.0, "c_boundary_mu4.json", "c_boundary_mu4.json"),
         ("C", 2, 0.125, "c_fixedN_N2_m0.125.json", "c_fixedN_N2_m0.125.json"),
         ("C", 2, 1.0, "c_boundary_mu2.json", "c_boundary_mu2.json"),
         ("A", 8, 1.0, "a_batch8_m1_3seed.json", "a_batch8_3seed.json"),
         ("A", 8, 0.2, "a_batch8_3seed.json", "a_batch8_3seed.json"),
         ("A", 32, 0.2, "a_tangent_3seed.json", "a_ablation_s12.json"),
         ("B", 32, 0.2, "b_tangent_3seed.json", "b_tangent_3seed.json"),
         ("B", 256, 0.2, "b_batch256_3seed.json", "b_batch256_3seed.json"),
         ("B", 32, 0.01, "b_tangent_m001_3seed.json", "b_tangent_3seed.json"),
         ("D", 32, 0.2, "d_bnci_3seed.json", "d_bnci_3seed.json"),
         ("E", 32, 0.2, "e_decoded_3seed.json", "e_decoded_3seed.json")]
# complete causal processing: cohort, mu, branch file; stem from CAUSAL_STEM
CAUSAL = [("C", 2, "c_fullcausal_mu2.json"), ("C", 4, "c_fullcausal_mu4.json"),
          ("C", 8, "c_fullcausal_mu8.json"), ("C", 32, "c_fullcausal_mu32.json"),
          ("C", 160, "c_fullcausal_3seed.json"), ("C", 640, "c_fullcausal_mu640.json"),
          ("D", 32, "d_fullcausal_mu32.json"), ("D", 160, "d_fullcausal_3seed.json"),
          ("D", 640, "d_fullcausal_mu640.json"),
          ("E", 8, "e_fullcausal_mu8.json"), ("E", 32, "e_fullcausal_mu32.json"),
          ("E", 160, "e_fullcausal_3seed.json"), ("E", 640, "e_fullcausal_mu640.json"),
          ("A", 8, "a_fullcausal_mu8.json"), ("A", 32, "a_fullcausal_mu32.json"),
          ("A", 160, "a_fullcausal_3seed.json"), ("A", 640, "a_fullcausal_mu640.json"),
          ("B", 160, "b_fullcausal_dump.json"), ("B", 3200, "b_fullcausal_mu3200.json")]
CAUSAL_STEM = {"C": "c_fullcausal_3seed.json", "D": "d_fullcausal_3seed.json",
               "E": "e_fullcausal_3seed.json", "A": "a_fullcausal_3seed.json",
               "B": "b_fullcausal_dump.json"}
# offline filters, causal reference update only
CAUSAL_UPDATE = {"C": "c_tangent_causal_3seed.json", "D": "d_tangent_causal_3seed.json",
                 "E": "e_tangent_causal_3seed.json", "A": "a_tangent_causal_3seed.json",
                 "B": "b_tangent_causal_3seed.json"}
# (N, m) settings measured for the contrast-survival figure
RATE_CONFIG = {"A": [(8, 1.0), (8, 0.2), (32, 0.2)],
               "B": [(32, 0.2), (256, 0.2), (32, 0.01)],
               "C": [(2, 1.0), (4, 1.0), (8, 1.0), (32, 0.2)],
               "D": [(32, 0.2)], "E": [(32, 0.2)]}
MOBI_EOG = ("FT9", "FT10", "TP9", "TP10")      # moved to the eyes, recorded as EOG


# ------------------------------------------------------------ result files --
def J(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def seedavg(d, arm, metric, subs=None):
    """Per-participant mean over seeds, in a fixed participant order."""
    subs = subs or sorted(d["%s|s0" % arm]["per_subject"])
    v = np.nanmean([[d["%s|s%d" % (arm, s)]["per_subject"][x][metric] for x in subs]
                    for s in SEEDS], axis=0)
    return subs, v


def per_seed(d, arm, subs, metric="acc"):
    """participants x seeds"""
    return np.array([[d["%s|s%d" % (arm, s)]["per_subject"][x][metric] for s in SEEDS]
                     for x in subs], float)


def delta(branch_file, stem_file, metric="acc"):
    """Branch minus stem per participant, seeds averaged."""
    subs, tan = seedavg(J(branch_file), "stem_tangent", metric)
    _, stem = seedavg(J(stem_file), "stem", metric, subs)
    return subs, tan - stem


def tau_blk():
    """Median single-class block of the test recordings, in windows."""
    ps = J("revision/protocol_stats.json")
    return {c: int(round(ps[c]["test"]["median"])) for c in COHORTS}


def sweep_delta():
    """Mean change in balanced accuracy per (cohort, mu) of the submitted sweep."""
    out = {}
    for coh, n, m, fb, fc in SWEEP:
        mu = int(round(n / m))
        if (coh, mu) not in out:              # C has two mu = 16 rows; not plotted on the surface
            out[(coh, mu)] = float(delta(fb, fc)[1].mean())
    return out


def supplement_deltas():
    """Per cohort: change with the reported and with the causal update order."""
    out = {}
    for coh in COHORTS:
        fb, fs_ = OFFLINE[coh]
        db, ds, dc = J(fb), J(fs_), J(CAUSAL_UPDATE[coh])
        subs = sorted(db["stem_tangent|s0"]["per_subject"])
        stem = per_seed(ds, "stem", subs).mean(1)
        br = per_seed(db, "stem_tangent", subs).mean(1)
        ca = per_seed(dc, "stem_tangent", subs).mean(1)
        # B and E were paired with a stem rerun in the same file
        ca_stem = per_seed(dc, "stem", subs).mean(1) if "stem|s0" in dc else stem
        out[coh] = (br - stem, ca - ca_stem)
    return out


# ------------------------------------------------------------- raw data --
def cfg(letter):
    with open(ROOT / "config" / ("cohort%s.yaml" % letter), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load(letter, sub, seed=0):
    """A cohort's splits with the trainer's per-channel scaling."""
    from data.cohorts import LOADERS
    from utils import normalise_global
    d = LOADERS[letter][1](cfg(letter), sub, seed)
    Xf, Xc, Xt = normalise_global(d["Xf"], d["Xc"], d["Xt"])
    return d, Xf, Xc, Xt


def subjects(letter):
    from data.cohorts import LOADERS
    return LOADERS[letter][0](cfg(letter))


def a_channels(sub="sub-01"):
    """EEG channel names of cohort A, in data order."""
    import mne
    from data.cohorts import DATA_ROOT, a_sessions
    root = DATA_ROOT / cfg("A")["data"]["dir"]
    ses = a_sessions(root, sub)[0]
    edf = root / sub / ("ses-%02d" % ses) / "eeg" / ("%s_ses-%02d_task-training_eeg.edf" % (sub, ses))
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    return [c for c in raw.ch_names if "EOG" not in c.upper()]


def b_channels():
    """Electrode name per data column of cohort B, from its impedance file."""
    from data.cohorts import DATA_ROOT
    p = DATA_ROOT / cfg("B")["data"]["dir"] / "SL01-T01" / "impedances-before.txt"
    names = re.findall(r"^#\s*(\d+)\s+(\S+)\s", p.read_text(encoding="latin-1"), re.M)
    out = [None] * max(int(i) for i, _ in names)
    for i, n in names:
        out[int(i) - 1] = n
    return out


# --------------------------------------------------- covariance geometry --
def covs(X, chunk=512):
    """Shrunk, trace-normalised window covariances, as the branch computes them."""
    from model import TangentBranch
    tb = TangentBranch(X.shape[1], 8)
    with torch.no_grad():
        return np.concatenate([tb._shrunk_cov(torch.as_tensor(X[i:i + chunk]).double()).numpy()
                               for i in range(0, len(X), chunk)])


def running_path(Cf, Ct, batch=BATCH, mom=MOM):
    """Reference in force for each test window (causal) and after each batch."""
    M = Cf.mean(0)
    ref_win, path = np.empty_like(Ct), [M.copy()]
    for i in range(0, len(Ct), batch):
        ref_win[i:i + batch] = M
        M = (1 - mom) * M + mom * Ct[i:i + batch].mean(0)
        path.append(M.copy())
    return ref_win, np.array(path)


def csp_pair(Cf, yf):
    """Walk- and Stop-dominant CSP filters of the fitting split, unit variance on its mean."""
    from scipy.linalg import eigh
    lam, V = eigh(Cf[yf == 1].mean(0), Cf[yf == 1].mean(0) + Cf[yf == 0].mean(0))
    W = np.stack([V[:, -1], V[:, 0]], 1)
    W = W / np.sqrt(np.einsum("ci,cd,di->i", W, Cf.mean(0), W))
    return W, lam[[-1, 0]]


def proj(C, W):
    return np.einsum("ci,...cd,dj->...ij", W, C, W)


def tangent_diag(C, ref):
    """Diagonal of log(ref^-1/2 C ref^-1/2), one row per window."""
    w, v = np.linalg.eigh(ref)
    ri = v @ np.diag(np.clip(w, 1e-6, None) ** -0.5) @ v.T
    ew, ev = np.linalg.eigh(ri @ C @ ri)
    return np.einsum("nij,nj,nij->ni", ev, np.log(np.clip(ew, 1e-6, None)), ev)


def maps_for(Cf, Ct, yf, yt, batch=BATCH, mom=MOM):
    """Session drift and Walk-Stop contrast per electrode, frozen and running reference."""
    M0 = Cf.mean(0)
    fit = tangent_diag(Cf, M0)
    sd = fit.std(0) + 1e-9
    tf = tangent_diag(Ct, M0)
    ref_win, _ = running_path(Cf, Ct, batch, mom)
    tr = np.empty_like(tf)
    for i in range(0, len(Ct), batch):
        tr[i:i + batch] = tangent_diag(Ct[i:i + batch], ref_win[i])
    bal = lambda t, y: 0.5 * (t[y == 1].mean(0) + t[y == 0].mean(0))
    out = {}
    for name, t in (("frozen", tf), ("running", tr)):
        out["drift_" + name] = (bal(t, yt) - bal(fit, yf)) / sd
        out["contrast_" + name] = (t[yt == 1].mean(0) - t[yt == 0].mean(0)) / sd
    return out


def predict_from_labels(y, mu):
    """Contrast left after an exponential average of memory mu follows this label sequence."""
    x = np.where(np.asarray(y) == 1, 1.0, -1.0)
    r, m = np.empty_like(x), x[0]
    for i, xi in enumerate(x):
        r[i] = m                               # reference before the window
        m += (xi - m) / mu
    d = x - r
    return 0.5 * (d[x > 0].mean() - d[x < 0].mean())


# ------------------------------------------------------------------ caches --
def cached(name, compute, recompute=False):
    f = CACHE / (name + ".npz")
    if f.exists() and not recompute:
        return dict(np.load(f, allow_pickle=True))
    out = compute()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return dict(np.load(f, allow_pickle=True))


def cone_data(recompute=False):
    """Test covariances and the running reference, projected by the CSP pair (A sub-01, B SL01)."""
    def compute():
        out = {}
        for tag, sub in (("A", "sub-01"), ("B", "SL01")):
            d, Xf, _, Xt = load(tag, sub)
            Cf, Ct = covs(Xf), covs(Xt)
            _, path = running_path(Cf, Ct)
            W, lam = csp_pair(Cf, d["yf"])
            yt = d["yt"]
            out.update({tag + "_pts": proj(Ct, W), tag + "_y": yt,
                        tag + "_path": proj(path, W), tag + "_fitmean": proj(Cf.mean(0), W),
                        tag + "_walkmean": proj(Ct[yt == 1].mean(0), W),
                        tag + "_stopmean": proj(Ct[yt == 0].mean(0), W), tag + "_lam": lam})
        return out
    return cached("fig_cone_data", compute, recompute)


def refmaps_data(recompute=False):
    """Drift and contrast maps averaged over every participant of A and B."""
    def compute():
        out = {}
        for tag in ("A", "B"):
            acc, subs = {}, subjects(tag)
            for sub in subs:
                d, Xf, _, Xt = load(tag, sub)
                for k, v in maps_for(covs(Xf), covs(Xt), d["yf"], d["yt"]).items():
                    acc.setdefault(k, []).append(v)
            out.update({"%s_%s" % (tag, k): np.mean(v, 0) for k, v in acc.items()})
            out[tag + "_n"] = len(subs)
        out["A_ch"] = np.array(a_channels(), dtype=object)
        out["B_ch"] = np.array(b_channels(), dtype=object)
        return out
    return cached("fig_refmaps_data", compute, recompute)


def rate_data(recompute=False):
    """Measured and label-predicted contrast survival at every RATE_CONFIG setting."""
    def compute():
        out = {}
        for tag, conf in RATE_CONFIG.items():
            acc = {}
            for sub in subjects(tag):
                try:
                    d, Xf, _, Xt = load(tag, sub)
                except Exception as e:
                    print("  [skip] %s %s: %s" % (tag, sub, e))
                    continue
                Cf, Ct = covs(Xf), covs(Xt)
                for n, m in conf:
                    mu = int(round(n / m))
                    mp = maps_for(Cf, Ct, d["yf"], d["yt"], n, m)
                    acc.setdefault("%s_meas_%d" % (tag, mu), []).append(
                        np.abs(mp["contrast_running"]).mean() / np.abs(mp["contrast_frozen"]).mean())
                    acc.setdefault("%s_pred_%d" % (tag, mu), []).append(
                        predict_from_labels(d["yt"], mu))
            out.update({k: np.array(v) for k, v in acc.items()})
        return out
    return cached("fig_rate3d_data", compute, recompute)


def topo_one(sub):
    """Reference power, M^-1/2 gain and Walk-Stop 8-30 Hz power (dB), fitting sessions."""
    from scipy.signal import butter, filtfilt
    from data.cohorts import DATA_ROOT, a_epochs, a_sessions
    from model import TangentBranch
    c = cfg("A")
    allses = a_sessions(DATA_ROOT / c["data"]["dir"], sub)
    es = a_epochs(c, sub, allses, ("training",))
    X = (es["X"] / (es["X"].std() + 1e-30)).astype(np.float32)
    present = set(int(v) for v in np.unique(es["session"]))
    sess = [s for s in allses if s in present][:c["split"]["n_fit_sessions"]]
    m = np.isin(es["session"], sess)
    X, y = X[m], es["y"][m]
    br = TangentBranch(X.shape[1], 128, shrink=c["model"]["shrink"]).eval()
    with torch.no_grad():
        br.freeze_reference(torch.as_tensor(X))
    M = br.run_cov.double().numpy()
    w, v = np.linalg.eigh(M)
    W = v @ np.diag(np.clip(w, 1e-6, None) ** -0.5) @ v.T
    # absolute band power: the reference diagonal is trace-normalised, hence relative
    nyq = c["data"]["sfreq"] / 2.0
    b_, a_ = butter(4, [c["data"]["band"][0] / nyq, c["data"]["band"][1] / nyq], btype="band")
    Xb = filtfilt(b_, a_, X.astype(np.float64), axis=-1)
    diff = 10.0 * np.log10((Xb[y == 1].var(-1).mean(0) + 1e-30) / (Xb[y == 0].var(-1).mean(0) + 1e-30))
    return dict(power=np.diag(M), gain=np.sqrt((W ** 2).sum(1)), diff=diff,
                n=int(len(y)), sessions=np.array(sess))


def topography_data(recompute=False, sub="sub-01"):
    def compute():
        r = topo_one(sub)
        ch = a_channels(sub)
        stats = {"subject": sub, "sessions": r["sessions"].tolist(), "n_windows": r["n"]}
        og, od = np.argsort(-r["gain"]), np.argsort(r["diff"])
        pick = lambda v, idx: [[ch[i], round(float(v[i]), 2)] for i in idx]
        stats.update(highest_gain=pick(r["gain"], og[:6]), lowest_gain=pick(r["gain"], og[-6:][::-1]),
                     largest_decrease=pick(r["diff"], od[:6]),
                     largest_increase=pick(r["diff"], od[-6:][::-1]))
        (RESULTS / "topography_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        return dict(r, ch=np.array(ch, dtype=object))
    return cached("fig_topography_data", compute, recompute)


def signflip(D):
    """Max-statistic sign-flip test of the per-channel mean across participants."""
    n = D.shape[0]
    flips = np.array(list(itertools.product([1, -1], repeat=n)))
    null_max = np.abs(flips @ D / n).max(1)
    obs = D.mean(0)
    return obs, np.array([(null_max >= abs(o)).mean() for o in obs])


def spatial_r(D, rng, n_perm=2000):
    """Mean pairwise map correlation against a within-map channel shuffle."""
    def mean_r(A):
        return np.corrcoef(A)[np.triu_indices(len(A), 1)].mean()
    obs = mean_r(D)
    null = np.array([mean_r(np.array([rng.permutation(r) for r in D])) for _ in range(n_perm)])
    return float(obs), float((null >= obs).mean())


def topography_group(recompute=False):
    """Group topography of cohort A, results/revision/topography_group.json."""
    path = RESULTS / "revision" / "topography_group.json"
    if path.exists() and not recompute:
        return J("revision/topography_group.json")
    subs, G, Dd, nwin = subjects("A"), [], [], {}
    names = a_channels(subs[0])
    for s in subs:
        assert a_channels(s) == names, "montage differs for %s" % s
        r = topo_one(s)
        G.append(r["gain"]); Dd.append(r["diff"]); nwin[s] = r["n"]
    G, Dd = np.array(G), np.array(Dd)
    mean_d, p_d = signflip(Dd)
    agree = (np.sign(Dd) == np.sign(mean_d)).sum(0)
    rng = np.random.default_rng(0)
    r_d, pr_d = spatial_r(Dd, rng)
    r_g, pr_g = spatial_r(G, rng)
    order = np.argsort(mean_d)
    res = {"_note": "Cohort A, fitting sessions: M^-1/2 gain and Walk-Stop 8-30 Hz power per "
                    "participant; sign agreement, max-statistic sign-flip p, spatial r.",
           "subjects": subs, "n_windows": nwin, "channels": names,
           "mean_diff_db": mean_d.tolist(), "signflip_p_maxstat": p_d.tolist(),
           "sign_agreement_of_7": agree.tolist(), "per_subject_diff_db": Dd.tolist(),
           "mean_gain": G.mean(0).tolist(),
           "spatial_r_diff": r_d, "spatial_r_diff_p": pr_d,
           "spatial_r_gain": r_g, "spatial_r_gain_p": pr_g,
           "significant_channels": [names[i] for i in np.flatnonzero(p_d < 0.05)],
           "largest_decrease": [[names[i], round(float(mean_d[i]), 2), int(agree[i])] for i in order[:8]],
           "largest_increase": [[names[i], round(float(mean_d[i]), 2), int(agree[i])]
                                for i in order[-8:][::-1]]}
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def activations_data(recompute=False, sub="sub-01", seed=0):
    """Train the reported model on one cohort A participant and tap four layers on its test windows."""
    def compute():
        from model import CALMNet
        from model.baseModel import baseModel
        from utils import set_seed
        c = cfg("A")
        d, Xf, Xc, Xt = load("A", sub, seed)
        set_seed(seed)
        mc = c["model"]
        net = CALMNet(Xf.shape[1], 2, n_filt=mc["n_filt"], d_model=mc["d_model"],
                      dropout=mc["dropout"], shrink=mc["shrink"], momentum=mc["momentum"])
        tr = baseModel(net, c["train"], c["train"]["batch"])
        tr.fit(Xf, d["yf"], Xc, d["yc"])
        store = {}
        grab = lambda k, t: store.setdefault(k, []).append(t.detach().float().cpu().numpy())
        hs = [net.tangent.proj.register_forward_pre_hook(lambda m, i: grab("tangent", i[0])),
              net.tangent.proj.register_forward_hook(lambda m, i, o: grab("project", o)),
              net.fuse.register_forward_pre_hook(lambda m, i: grab("concat", i[0])),
              net.fuse.register_forward_hook(lambda m, i, o: grab("fuse", o))]
        logits = tr.predict_logits(Xt)
        for h in hs:
            h.remove()
        out = {k: np.concatenate(v) for k, v in store.items()}
        out["stem"] = out["concat"][:, :mc["d_model"]]
        return dict(out, y=d["yt"], logits=logits)
    return cached("fig_activations_data", compute, recompute)
