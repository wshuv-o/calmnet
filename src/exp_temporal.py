"""Two levers this project never pulled: window LENGTH and TEMPORAL STRUCTURE.

Everything run so far -- 131 architecture variants, 18 published backbones, 5
preprocessings x 5 feature families, ~14 ablated modules, two cohorts -- shares
two fixed choices that were never treated as variables:

  1. Every epoch cache on disk is w2.0_st0.5. Window length was flagged as
     untested in features.py and never swept, even though ERD/ERS is a
     time-resolved phenomenon and the one time the representation WAS varied
     (the frequency-band series) it moved the honest number more than any
     architecture did.

  2. Every model treats windows as i.i.d. But Walk and Stop are *sustained
     states with dwell times* -- the rexstate segments are tens of seconds long
     and the window stream is sampled every 0.5 s. No model has been given
     access to that structure.

WHY TEMPORAL STRUCTURE IS THE RIGHT THING TO TRY HERE, SPECIFICALLY

The project's blocking result is that accuracy and movement leakage rise
together: every mechanism that bought accuracy did it by admitting more
movement. A label-space temporal prior is the one mechanism that *cannot* do
that, because it never touches the representation -- it only reweights a
decision using the dwell statistics of the label sequence. If it raises
accuracy, the gain is invariant almost by construction.

"Almost" is doing work in that sentence, and this experiment is built to attack
it. The obvious objection is that dwell structure IS movement structure: the
subject moves continuously while walking, so a prior that smooths toward
persistence might be laundering the confound rather than avoiding it.

MEASURING THAT REQUIRED FIXING THE MEASUREMENT

The project's existing leakage metric -- movement R^2 from the representation --
cannot answer the question, because Walk and Stop differ in how much the body
moves BY DEFINITION. The label predicts motion, so any representation that
decodes the label must predict motion, and raw R^2 rises with accuracy whether
or not the decoder is cheating. That was survivable while every architecture
scored within a point of every other; it breaks the moment something actually
improves accuracy. Verified on synthetic data: features encoding ONLY the label,
with motion driven ONLY by the label, score raw R^2 = +0.995 -- indistinguishable
from genuine artefact capture.

So four probes are reported, and the conditional pair is the one that carries
the argument:

  r2_feat        raw movement R^2 from the FEATURES
  r2_dec         raw movement R^2 from the EMITTED DECISION
  r2_cond_feat   movement R^2 from features, WITHIN class  <- the honest one
  r2_cond_dec    movement R^2 from decision, WITHIN class  <- the honest one

Conditioning centres the movement features within each class, removing the
component the label explains and leaving the residual motion that varies inside
Walk and inside Stop. A decoder that merely knows "walking bodies move" predicts
none of it (~0). One that has latched onto stride cadence or head bob still
does. On the synthetic case above the conditional probe returns -0.001 where the
raw probe returned +0.995.

ARMS

  none          i.i.d. windows                                 (baseline)
  forward       causal forward filtering                       (ONLINE / deployable)
  viterbi       HMM Viterbi decode                             (offline upper bound)
  stack         feature-space context stacking t-1,t,t+1       (leakage control)
  fwdshuf       forward filter on SHUFFLED streams             (null control)
  forward@0.5   forward filter with a uniform transition matrix(null control)
  forward@auto  tau calibrated to a false-onset budget         (the deployable rule)

Three controls, because a smoothing result is easy to fake. `stack` injects
temporal context in FEATURE space, so unlike the label-space arms it CAN import
movement -- separating "temporal context helps" from "label-space priors are
safe". `fwdshuf` runs identical arithmetic with window ORDER destroyed, so if
the gain is really temporal it must vanish. `forward@0.5` keeps the prior
arithmetic but removes all memory (verified: at tau=0.5 the filter reduces
exactly to prior-corrected emissions), isolating how much of the gain is simply
the class-prior division that the filter performs along the way.

Base representation is tangent-space + Euclidean Alignment with L2 logistic
regression: the pipeline that actually won the representation sweep. Held fixed
so differences are attributable to the two axes under test.

Writes results/temporal.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions
from splits import grouped_split
from abstain import balanced_accuracy
from calmnet_msa import imu_valid_mask
import features as FE

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "temporal.json"


SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]
N_TRAIN = 3
WINDOWS = [1.0, 2.0, 3.0, 4.0]
ARMS = ["none", "forward", "viterbi", "stack"]
# Split seed is the only stochastic element (plus the shuffle control's own
# generator). This project has previously been misled by single-seed
# leaderboards, so it is made settable rather than hard-coded at 0.
SEED = int(os.environ.get("TEMPORAL_SEED", "0"))
EPS = 1e-12

def save(path, key, value):
    """Merge-then-write, atomically.

    Two runs of this script share an output file whenever they cover different
    arms of the same cohort, and a read-modify-write held in memory from process
    start silently discards whatever the other process finished in the meantime.
    That already destroyed one completed sweep in this project. Re-reading
    immediately before writing keeps the window to milliseconds, and writing via
    a temp file + os.replace means a crash mid-write cannot leave truncated JSON
    that the next run would fail to parse.
    """
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text())
        except Exception:
            cur = {}
    cur[key] = value
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, indent=1))
    os.replace(tmp, path)
    return cur


# --------------------------------------------------------------------------- #
# Stream reconstruction
# --------------------------------------------------------------------------- #
def streams(es_session, es_task, es_segment):
    """Contiguous recording streams, in time order.

    build_epochs appends per session -> per task -> per segment -> per window, so
    array order IS time order inside a (session, task) recording. That is an
    assumption about another module's loader, so it is asserted rather than
    trusted: segment ids must be non-decreasing within every recording. If the
    loader ever reorders, the HMM would silently decode a shuffled sequence and
    still return a plausible-looking number -- the exact failure mode that is
    impossible to spot downstream.
    """
    out = []
    key = np.array(["%s|%s" % (s, t) for s, t in zip(es_session, es_task)])
    for k in dict.fromkeys(key.tolist()):          # preserve first-seen order
        idx = np.flatnonzero(key == k)
        seg = np.asarray(es_segment)[idx]
        assert np.all(np.diff(seg) >= 0), "stream %s is not in time order" % k
        out.append(idx)
    return out


def transition_matrix(y, strm, n_cls=2, prior=1.0):
    """Estimate P(y_t | y_t-1) from consecutive windows inside each stream."""
    A = np.full((n_cls, n_cls), prior, float)
    for idx in strm:
        ys = np.asarray(y)[idx]
        for a, b in zip(ys[:-1], ys[1:]):
            A[int(a), int(b)] += 1.0
    return A / A.sum(1, keepdims=True)


def transition_from_tau(tau, n_cls=2):
    """A dwell prior with ONE knob: the self-transition probability.

    Estimating the transition matrix from data sounds more principled than
    setting it, and on ds007788 it is -- there are dozens of state changes per
    session to count. On the MoBI cohort it is actively dangerous: each
    recording contains a single contiguous walk block, so the estimate comes out
    at P(stay walking) = 0.999 from essentially one transition. A prior that
    strong makes leaving the majority state nearly impossible and Viterbi
    degenerates to predicting Walk everywhere, which scores a tidy-looking
    accuracy at balanced-chance.

    So tau is exposed as a deliberate choice instead. Sweeping it traces the
    false-onset / latency curve directly: high tau buys a device that almost
    never lurches but responds late, low tau the reverse. That curve is the
    point -- a confidence threshold cannot produce it, because thresholding a
    per-window score has no notion of how long a state has already persisted.
    """
    off = (1.0 - tau) / (n_cls - 1)
    A = np.full((n_cls, n_cls), off, float)
    np.fill_diagonal(A, tau)
    return A


def parse_arm(arm):
    """`forward` -> (forward, None);  `forward@0.95` -> (forward, 0.95);
    `forward@auto` -> (forward, "auto") for budget calibration."""
    if "@" in arm:
        base, tau = arm.split("@", 1)
        return base, (tau if tau == "auto" else float(tau))
    return arm, None


TAU_GRID = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999)
ONSET_BUDGET = 0.5      # spurious walk commands per minute of standing


def calibrate_tau(P_cal, y_cal, strm_cal, pi, decode, budget=ONSET_BUDGET,
                  grid=TAU_GRID):
    """Pick the dwell prior from a SAFETY budget rather than by guessing.

    A fixed tau is arbitrary and a tau estimated from the label counts is
    degenerate on single-block recordings. The defensible version states the
    constraint that actually matters for a wearable exoskeleton -- "do not
    spuriously command a walk more than `budget` times per minute of standing"
    -- and then buys the cheapest prior that satisfies it.

    Cheapest means lowest latency: persistence is what suppresses false onsets,
    and it is also what delays real ones, so the two move together and the
    smallest adequate tau is optimal. Scanning the grid low-to-high and stopping
    at the first feasible point therefore minimises latency subject to the
    safety constraint, rather than maximising accuracy and reporting safety
    afterwards.

    Selection happens on the held-out calibration split, never on test -- the
    same discipline the conformal machinery elsewhere in this project uses, and
    the reason the reported test false-onset rate is an honest estimate rather
    than the number that was optimised.
    """
    for tau in sorted(grid):
        A = transition_from_tau(tau)
        pred = np.zeros(len(P_cal), int)
        for idx in strm_cal:
            pred[idx] = decode(P_cal[idx], A, pi)
        if onset_metrics(y_cal, pred, strm_cal)["false_onsets_per_min"] <= budget:
            return float(tau)
    return float(max(grid))


def _log_emission(P, pi):
    """Turn calibrated posteriors into emission log-likelihoods.

    P(x|y) is proportional to P(y|x)/P(y); dividing out the training class prior
    matters here because Stop outnumbers Walk roughly 2.5:1, so leaving the prior
    in would apply it twice -- once in the classifier, once in the HMM.
    """
    return np.log(np.clip(P, EPS, None)) - np.log(np.clip(pi, EPS, None))


def forward_filter(P, A, pi):
    """Causal filtering: alpha_t(j) = P(y_t=j | x_1..t). Deployable online --
    uses no future windows, which is the only variant a real exoskeleton could
    actually run."""
    E = np.exp(_log_emission(P, pi))
    a = pi * E[0]
    a = a / (a.sum() + EPS)
    out = [a]
    for t in range(1, len(P)):
        a = (a @ A) * E[t]
        a = a / (a.sum() + EPS)
        out.append(a)
    return np.stack(out)


def viterbi(P, A, pi):
    """MAP label sequence. Offline -- an upper bound on what temporal structure
    can buy, not a deployable decoder."""
    lE, lA, n = _log_emission(P, pi), np.log(np.clip(A, EPS, None)), len(P)
    K = P.shape[1]
    d = np.log(np.clip(pi, EPS, None)) + lE[0]
    bp = np.zeros((n, K), int)
    for t in range(1, n):
        m = d[:, None] + lA
        bp[t] = m.argmax(0)
        d = m.max(0) + lE[t]
    path = np.zeros(n, int)
    path[-1] = int(d.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def stack_context(F, strm, k=1):
    """Concatenate features from t-k..t+k, edge-padded inside each stream.

    Unlike the label-space arms this touches the REPRESENTATION, so it is the
    control that can leak.
    """
    out = np.zeros((len(F), F.shape[1] * (2 * k + 1)), F.dtype)
    for idx in strm:
        B = F[idx]
        cols = [B[np.clip(np.arange(len(B)) + o, 0, len(B) - 1)]
                for o in range(-k, k + 1)]
        out[idx] = np.concatenate(cols, 1)
    return out


# --------------------------------------------------------------------------- #
# Deployment metrics
# --------------------------------------------------------------------------- #
def onset_metrics(y_true, y_pred, strm, step=0.5):
    """What a wearer of the exoskeleton would actually experience.

    Per-window balanced accuracy is the wrong currency for a device that starts
    and stops a person's legs. Two things matter instead:

      false_onsets_per_min  how often the device spuriously COMMANDS a walk
                            while the wearer is standing still. A single
                            isolated false window is a lurch; the per-window
                            error rate hides how many distinct lurches there
                            are, because one long false run and fifty scattered
                            false windows can score identically.

      latency_s             how long after a genuine walk intent the device
                            actually moves. This is the PRICE of a persistence
                            prior and the reason this function exists: smoothing
                            suppresses false onsets by being reluctant to change
                            state, and that same reluctance delays real onsets.
                            Reporting the first without the second would be
                            advertising a free lunch that is not free.

    Together they are a trade-off curve, not a score. The dwell prior sets the
    operating point on it explicitly, which is the part a fixed confidence
    threshold cannot do.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    false_onsets, stop_windows, lat = 0, 0, []
    for idx in strm:
        yt, yp = y_true[idx], y_pred[idx]
        stop_windows += int((yt == 0).sum())
        # a false onset is a 0->1 transition in the PREDICTION while truly Stop
        rise = np.flatnonzero((yp[1:] == 1) & (yp[:-1] == 0)) + 1
        false_onsets += int(sum(1 for r in rise if yt[r] == 0))
        # detection latency at each genuine Stop->Walk transition
        true_rise = np.flatnonzero((yt[1:] == 1) & (yt[:-1] == 0)) + 1
        for r in true_rise:
            end = len(yt)
            nxt = true_rise[true_rise > r]
            stop_after = np.flatnonzero(yt[r:] == 0)
            if len(stop_after):
                end = r + int(stop_after[0])
            hit = np.flatnonzero(yp[r:end] == 1)
            lat.append(float(hit[0]) * step if len(hit) else float("nan"))
    stop_min = max(stop_windows * step / 60.0, 1e-9)
    miss = float(np.mean([np.isnan(v) for v in lat])) if lat else float("nan")
    return {
        "false_onsets_per_min": float(false_onsets / stop_min),
        "latency_s": float(np.nanmedian(lat)) if lat else float("nan"),
        "missed_onsets": miss,
        "n_true_onsets": int(len(lat)),
    }


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(sub, win):
    es = build_epochs(subject=sub, win=win, step=0.5)
    valid = imu_valid_mask(es.imu_feats, es.session)
    pres = sorted(set(int(v) for v in np.unique(es.session)))
    sess = [s for s in list_sessions(sub) if s in pres]
    tr = np.isin(es.session, sess[:N_TRAIN])
    ti, ci = grouped_split(es.segment[tr], es.y[tr], frac=0.3, seed=SEED)
    f = lambda a: a[tr][ti]
    c = lambda a: a[tr][ci]
    return {
        "Xf": f(es.X), "yf": f(es.y), "Mf": f(es.imu_feats), "vf": f(valid),
        "sf": f(es.session), "tf": f(es.task), "gf": f(es.segment),
        "Xc": c(es.X), "yc": c(es.y), "sc": c(es.session), "tc": c(es.task),
        "gc": c(es.segment),
        "Xt": es.X[~tr], "yt": es.y[~tr], "Mt": es.imu_feats[~tr],
        "vt": valid[~tr], "st": es.session[~tr], "tt": es.task[~tr],
        "gt": es.segment[~tr],
    }


def load_mobi(sub, win):
    """Second cohort (Luu et al. treadmill BCI), same dict shape as `load`.

    Worth running for more than the usual "does it replicate" reason. MoBI is
    where the project's wrong-walk safety bound is currently BREACHED (0.28-0.39
    against a 0.05 target), because Stop is only ~11% of that cohort -- the
    mirror image of ds007788, where Stop is the majority. A dwell prior that
    only worked on the majority class would be worthless, so a cohort with the
    imbalance flipped is the sharpest available test of it.

    Split convention (fit trial 1, test trials 2-3) matches exp_mobi.py and
    exp_calmnet3.py so numbers stay comparable with what is already reported.
    Trials, not chunks, are the probe grouping: chunks are 20 s neighbours and
    would let the probe exploit autocorrelation.
    """
    from dataio_mobi import build_subject
    es = build_subject(sub, win=win, step=0.5)
    if es is None or len(es) < 100:
        raise RuntimeError("no MoBI data for %s" % sub)
    fit, test = es.by_trials([1]), es.by_trials([2, 3])
    ti, ci = grouped_split(fit.segment, fit.y, frac=0.3, seed=SEED)
    const = lambda n: np.array(["mobi"] * n, object)
    return {
        "Xf": fit.X[ti], "yf": fit.y[ti], "Mf": fit.motion[ti],
        "vf": np.ones(len(ti), bool), "sf": fit.trial[ti],
        "tf": const(len(ti)), "gf": fit.segment[ti],
        "Xc": fit.X[ci], "yc": fit.y[ci], "sc": fit.trial[ci],
        "tc": const(len(ci)), "gc": fit.segment[ci],
        "Xt": test.X, "yt": test.y, "Mt": test.motion,
        "vt": np.ones(len(test), bool), "st": test.trial,
        "tt": const(len(test)), "gt": test.segment,
    }


def _tangent(X):
    """Covariance -> Euclidean Alignment -> log-Euclidean tangent vector.

    EA is fitted per set, which is the point of it: each set gets whitened by
    its OWN mean covariance, so no statistic crosses from test back into fit.
    """
    return FE.tangent(FE.euclidean_align(FE.covariances(X)))


# --------------------------------------------------------------------------- #
# One (window, arm) cell
# --------------------------------------------------------------------------- #
def embed(d):
    """Compute the tangent-space features ONCE per subject, not once per arm.

    Every arm shares the same representation -- only what happens to the
    decision afterwards differs -- so recomputing it per arm is pure waste. On
    ds007788 that waste is invisible (60 channels, ~900 windows); on MoBI it is
    the entire cost, because 5042 test windows of 64x64 covariances take 62 s to
    embed and a ten-arm sweep paid that eight times over. Hoisting it turns an
    85-minute cohort into a 15-minute one and changes no number.
    """
    if "Ff" not in d:
        d["Ff"], d["Ft"] = _tangent(d["Xf"]), _tangent(d["Xt"])
        if "Xc" in d:
            d["Fc"] = _tangent(d["Xc"])
    return d


def run_subject(d, arm):
    arm, tau = parse_arm(arm)
    embed(d)
    Ff, Ft = d["Ff"], d["Ft"]
    sf = streams(d["sf"], d["tf"], d["gf"])
    st = streams(d["st"], d["tt"], d["gt"])

    if arm == "stack":
        Ff, Ft = stack_context(Ff, sf), stack_context(Ft, st)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=0.1,
                                           class_weight="balanced"))
    clf.fit(Ff, d["yf"])
    P = clf.predict_proba(Ft)

    pi = np.array([(d["yf"] == c).mean() for c in (0, 1)], float)
    decode = {"forward": lambda p, A, pi_: forward_filter(p, A, pi_).argmax(1),
              "viterbi": viterbi}.get(arm)

    if tau == "auto":
        # tau chosen on the held-out calibration split, never on test
        Fc = d["Fc"]
        sc = streams(d["sc"], d["tc"], d["gc"])
        tau = calibrate_tau(clf.predict_proba(Fc), d["yc"], sc, pi, decode)
    A = transition_matrix(d["yf"], sf) if tau is None else transition_from_tau(tau)

    if arm in ("none", "stack"):
        post, pred = P, P.argmax(1)
    elif arm == "forward":
        post = np.zeros_like(P)
        for idx in st:
            post[idx] = forward_filter(P[idx], A, pi)
        pred = post.argmax(1)
    elif arm == "viterbi":
        post, pred = P.copy(), np.zeros(len(P), int)
        for idx in st:
            pred[idx] = viterbi(P[idx], A, pi)
    elif arm == "fwdshuf":
        # NULL CONTROL. Identical arithmetic to `forward`, but the window order
        # inside each stream is permuted first and the outputs are mapped back.
        # Every emission is untouched; only temporal ADJACENCY is destroyed. If
        # `forward` beats the baseline for the reason claimed, this must not --
        # and if it does, the gain was never about time.
        rng = np.random.default_rng(0)
        post = np.zeros_like(P)
        for idx in st:
            perm = rng.permutation(len(idx))
            back = np.empty_like(perm)
            back[perm] = np.arange(len(perm))
            post[idx] = forward_filter(P[idx][perm], A, pi)[back]
        pred = post.argmax(1)
    else:
        raise ValueError(arm)

    acc = balanced_accuracy(d["yt"], pred)
    # Both probes run on the TEST distribution, session-grouped, fixed width --
    # the corrected probe. r2_feat is representation-level, r2_post is what the
    # decoder actually emits.
    vt = d["vt"]
    r2f = r2p = r2d = r2cf = r2cd = float("nan")
    if vt.sum() > 40:
        r2f = FE.invariance_r2_cv(Ft[vt], d["Mt"][vt], d["st"][vt])
        r2p = FE.invariance_r2_cv(post[vt], d["Mt"][vt], d["st"][vt])
        # r2_dec: the probe on the HARD decision the arm actually emits.
        #
        # r2_post is not comparable across arms as it stands. Viterbi replaces
        # the predicted PATH but leaves the soft posterior untouched, so its
        # r2_post is by construction identical to the baseline's -- it is
        # literally the same array, and reading that as "Viterbi adds no
        # leakage" would be reading a tautology. The emitted label sequence is
        # the one object every arm genuinely produces and a device genuinely
        # acts on, so probing it is the only measure that means the same thing
        # in every row. Probe width does not distort the comparison: the null
        # is flat (-0.003) at widths 2, 64 and 1830.
        r2d = FE.invariance_r2_cv(pred[vt, None].astype(float),
                                  d["Mt"][vt], d["st"][vt])
        # The label-conditional probes: excess movement information, over and
        # above what Walk-vs-Stop already implies. These are the ones that can
        # be compared between arms at different accuracies.
        r2cf = FE.invariance_r2_conditional(Ft[vt], d["Mt"][vt], d["yt"][vt],
                                            d["st"][vt])
        r2cd = FE.invariance_r2_conditional(pred[vt, None].astype(float),
                                            d["Mt"][vt], d["yt"][vt], d["st"][vt])
    dep = onset_metrics(d["yt"], pred, st)
    return acc, r2f, r2p, r2d, r2cf, r2cd, A[1, 1], A[0, 0], dep


def main():
    wins = [float(w) for w in sys.argv[1].split(",")] if len(sys.argv) > 1 else WINDOWS
    cohort = sys.argv[2] if len(sys.argv) > 2 else "ds007788"
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ARMS
    suffix = "" if SEED == 0 else "_seed%d" % SEED
    out_path = (RESULTS / ("temporal%s.json" % suffix) if cohort == "ds007788"
                else RESULTS / ("temporal_%s%s.json" % (cohort, suffix)))
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    # Explicit dispatch: an unrecognised cohort name used to fall through to the
    # MoBI loader and run a whole sweep on the wrong dataset under the right
    # filename, which is the kind of mistake that is invisible in the output.
    if cohort.startswith("ds007788"):
        subs, loader = SUBJECTS, load
    elif cohort.startswith("mobi"):
        from dataio_mobi import subjects as mobi_subjects
        subs, loader = mobi_subjects(), load_mobi
    else:
        raise SystemExit("unknown cohort %r (expected ds007788* or mobi*)" % cohort)
    for win in wins:
        D = {}
        for sub in subs:
            try:
                D[sub] = loader(sub, win)
            except Exception as e:
                print("  [skip] %s win=%s: %s: %s" % (sub, win, type(e).__name__, e),
                      flush=True)
        if not D:
            continue
        n = int(np.mean([len(d["yf"]) for d in D.values()]))
        print("\n######## [%s] win=%ss  n_fit~%d/subject  %d subjects ########"
              % (cohort, win, n, len(D)), flush=True)
        for arm in arms:
            t0 = time.time()
            A, Rf, Rp, Rd, Rcf, Rcd, dw, ds, DEP = ([] for _ in range(9))
            for sub, d in D.items():
                try:
                    a, rf, rp, rd, rcf, rcd, w1, w0, dep = run_subject(d, arm)
                except Exception as e:
                    print("    [fail] %s/%s: %s: %s" % (sub, arm, type(e).__name__, e),
                          flush=True)
                    continue
                A.append(a); Rf.append(rf); Rp.append(rp); Rd.append(rd)
                Rcf.append(rcf); Rcd.append(rcd)
                dw.append(w1); ds.append(w0); DEP.append(dep)
            if not A:
                continue
            acc, r2f, r2p = float(np.mean(A)), float(np.nanmean(Rf)), float(np.nanmean(Rp))
            r2d = float(np.nanmean(Rd))
            r2cf, r2cd = float(np.nanmean(Rcf)), float(np.nanmean(Rcd))
            dep_m = {k: float(np.nanmean([d[k] for d in DEP])) for k in DEP[0]}
            key = "w%s|%s" % (win, arm)
            out[key] = {"cohort": cohort, "win": win, "arm": arm,
                        "acc": acc, "acc_std": float(np.std(A)),
                        "r2_feat": r2f, "r2_post": r2p, "r2_dec": r2d,
                        "r2_cond_feat": r2cf, "r2_cond_dec": r2cd,
                        "score": acc - max(0.0, r2cf), "n_fit": n,
                        "per_subject": {s: float(v) for s, v in zip(D, A)},
                        "self_walk": float(np.mean(dw)), "self_stop": float(np.mean(ds)),
                        "deploy": dep_m, "secs": round(time.time() - t0, 1)}
            print("  %-14s acc %.3f+-%.3f  r2_feat %+.3f  r2_dec %+.3f  "
                  "COND feat %+.3f dec %+.3f | onsets/min %.2f  lat %.1fs  missed %.2f  "
                  "tau %.3f (%.0fs)"
                  % (arm, acc, np.std(A), r2f, r2d, r2cf, r2cd,
                     dep_m["false_onsets_per_min"], dep_m["latency_s"],
                     dep_m["missed_onsets"], float(np.mean(dw)), time.time() - t0),
                  flush=True)
            out = save(out_path, key, out[key])
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
