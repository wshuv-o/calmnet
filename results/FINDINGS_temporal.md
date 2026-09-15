# Preprocessing, not architecture: what actually capped this project

---

## 00. ARCHITECTURES ARE NOT DISTINGUISHABLE ON THIS DATASET

The split seed -- which only changes WHICH segments go to fit versus calibration,
not how much data there is -- moves accuracy more than any architectural choice
does. Same subjects, same test sessions, same normalisation, 4 s windows:

| model | seed 0 | seed 1 | swing |
|---|---|---|---|
| ShallowFBCSPNet (2017 baseline) | 0.821 | **0.874** | 0.053 (n=2) |
| ATCNet | 0.828 | 0.868 | 0.040 (n=2) |
| PowerAttn-noattn | 0.836 | 0.868 | 0.032 (n=2) |
| PowerAttn-full (the merge) | 0.837 | 0.852 | 0.015 |
| PowerAttn-nopower | 0.837 | 0.839 | 0.002 |

At seed 0 the merge ranks **first**; at seed 1 it ranks **fourth** and the plain
2017 baseline wins.

### Calibrated version of that claim (the two-seed reading above overstates it)

A "swing" over two seeds is the range of two samples and estimates variance
badly. With three or more seeds the instability is real but roughly half what
two seeds suggested:

| pipeline | n | mean | seed SD |
|---|---|---|---|
| PowerAttn-nopower | 3 | 0.842 | 0.006 |
| bare CNN (perwindow) | 3 | 0.823 | 0.008 |
| PowerAttn-full | 3 | 0.840 | 0.009 |
| tangent + logreg | 4 | 0.778 | 0.009 |
| bare CNN (none) | 3 | 0.873 | 0.013 |
| bare CNN (global) | 3 | 0.862 | 0.023 |
| ShallowFBCSPNet | 2 | 0.847 | 0.027 |
| ATCNet | 2 | 0.848 | 0.020 |

Typical seed SD is **~0.010-0.015**, and the two alarming values are the two
with n=2. So the defensible statement is:

> Seed SD is ~0.01; with 3 seeds the standard error on a mean is ~0.006-0.013.
> Architecture differences of ~0.02 among deep models are marginal but NOT
> unresolvable -- they require 3+ seeds and cannot be read off a single run.

That is weaker than "architectures are indistinguishable", which an earlier
draft of this section claimed on n=2 evidence. It still invalidates every
single-seed ranking here, which is the part that matters, but it does not
license "measurement is hopeless".

**Consequence: no single-seed architecture ranking on this dataset carries
information.** That covers, retroactively:

- the 131-variant sweep and its "winners"
- the 18-backbone leaderboard
- the v3 arm comparison
- the 0.836 CNN headline reported earlier in this session, which reversed sign
  on seeds 1 and 2
- the motion canceller, which looked like the best module in the project

Measurement precision on 905 windows x 7 subjects is roughly +-0.05. Every
architectural effect anyone has proposed for this task is smaller than that. The
only routes to a trustworthy architecture comparison are more data per subject or
more subjects -- not a better module.

### The merge itself (PowerAttnNet = ShallowFBCSPNet power pathway x ATCNet attention)

Built on the two mechanisms this project measured: power is the signal (+0.134
from restoring amplitude) and the states have 12-32 s dwells. It keeps
ShallowFBCSPNet's `square -> pool -> log` estimator but emits a SEQUENCE instead
of collapsing the time axis, then attends over it -- the thing neither parent
does.

Pre-registered success condition: `full` must beat both its own ablations and
both parents. **It failed**, and with three seeds the failure is now
well-measured rather than noise-limited:

    PowerAttn-full     0.840 +- 0.009  (n=3)
    PowerAttn-nopower  0.842 +- 0.006  (n=3)   <- removing the power pathway: no cost
    PowerAttn-noattn   0.852 +- 0.016  (n=2)   <- removing attention: no cost, if anything better

The power pathway -- the entire reason for the merge -- contributes nothing, and
the error bars are now tight enough to say that rather than to shrug at it.
Removing attention does not hurt either. Both parents sit at ~0.847-0.848, so the
merge does not beat them.

Also tried and failed: an amplitude side-channel fused into all seven published
backbones. Mean effect -0.005 (4 up, 3 down), and a shuffle control that permutes
the amplitude vector across the batch matched the intact version exactly
(EEGNeX 0.841 both ways) -- so the apparent gains were capacity, not information.

---

---

## THE RESULT: amplitude normalisation, found by reading someone else's code

Comparing against `shonaka/EEG-neural-decoding` (Contreras-Vidal lab, who
recorded the MoBI cohort) surfaced one difference: they fit a single
StandardScaler on train and apply it to test. `dataio.build_epochs` does
`X.std(axis=2)` -- every channel of every window forced to unit variance
independently.

Bare CNN, 4 s windows, 3 seeds:

| scheme | accuracy | onsets/min | cond R² |
|---|---|---|---|
| perwindow (what the project always used) | 0.823 ± 0.008 | 1.34 | −0.082 |
| global (per-channel stats from the fit split) | 0.862 ± 0.023 | 1.20 | −0.075 |
| **none** (one global scalar; all relative amplitude kept) | **0.873 ± 0.013** | **0.97** | −0.076 |

Monotonic: the more amplitude information survives, the better the decoder gets,
on accuracy AND safety together, with leakage flat. **+0.050 from a
preprocessing line** -- larger than the spread across all 131 architecture
variants, 18 backbones and 14 ablated modules tried before it.

### What it does NOT mean

An earlier draft of this section said per-window z-scoring "deletes ERD". That
is too strong and the benchmark disproves it: ShallowFBCSPNet scores 0.824 on
per-window data, which would be impossible if the signal were gone.

What per-window normalisation removes is **marginal per-channel power**. The
correlation structure survives, and a spatial filter's output power depends on
correlations rather than on marginal variances -- so architectures that learn
spatial filters, or renormalise internally, route around the loss. The damage is
specific to features that read marginal power directly:

| feature | per-window | raw | Δ |
|---|---|---|---|
| log band-power (pure marginal power) | 0.625 | 0.760 | **+0.134** |
| tangent + EA (correlation-based) | 0.762 | 0.746 | −0.016 |
| correlation only | 0.738 | 0.734 | −0.004 |

### Published architectures, 4 s, seed 0

| model | perwindow | global | Δ |
|---|---|---|---|
| Deep4Net | 0.760 | 0.821 | **+0.061** |
| ShallowFBCSPNet | 0.824 | 0.821 | −0.003 |
| EEGNeX | 0.834 | 0.832 | −0.002 |
| EEGITNet | 0.833 | 0.791 | **−0.042** |
| BDTCN | 0.820 | 0.822 | +0.002 |

Model-dependent, not universal. **Provisional headline: the bare CNN with
minimal normalisation (0.873 ± 0.013, 3 seeds) beats every published
architecture tested (0.760–0.834, 1 seed each).** Provisional because the
published models have one seed and have not yet been run under `none`, which was
the winning scheme -- both queued.

### Two bugs found on the way

- `features.covariances` added an **absolute** 1e-5 ridge. Raw EEG variance is
  ~8e-11, so on unnormalised data the regulariser exceeded the signal by five
  orders of magnitude and every covariance came back as pure identity, silently.
  Now scale-relative; numerically unchanged for normalised input.
- `exp_published` defaults `mods.get(k, True)`, which would have switched
  excluded modules back ON in every cell of the clean ablation.

### Probe sensitivity (`exp_sensitivity.py`) -- the null is now a bound

Injecting the real motion reference, spatially mixed, at known power:

| injected | cond R² | raw R² | accuracy |
|---|---|---|---|
| 0% | −0.112 | +0.199 | 0.763 |
| 0.1% | −0.057 | +0.447 | 0.819 |
| **0.5%** | **+0.015** | +0.726 | 0.873 |
| 5% | +0.085 | +0.828 | 0.913 |
| 50% | +0.094 | +0.821 | 0.917 |

**Detection threshold 0.50% of signal power.** So the −0.11 on clean data bounds
real contamination below half a percent, rather than merely failing to find it.
Note 5% injected artefact buys +0.15 accuracy: contamination is extremely
exploitable, which is why every gain in this project needed a control.

---

---

## 0. THE MOTION CANCELLER IS AN IMU CLASSIFIER — exclude every arm containing it

Under the corrected probe `A_+cancel` scores **0.905** balanced accuracy with
conditional R² = −0.041, the best number anywhere in this project. It is not a
decoder. `exp_cancel_control.py`:

```
bare   : true 0.811  shuffled 0.811  zeroed 0.811     <- correctly invariant
+cancel: true 0.905  shuffled 0.529  zeroed 0.501     <- chance without the IMU
```

`MotionReferenceCanceller` computes `y = x - g*h(m)`, and `h` is bias-free, so
`h(0) = 0` and a **zeroed** reference makes the layer exactly the identity — the
classifier receives clean, unmodified EEG. On that EEG it scores **0.501**. It
never learned to use the brain signal at all; it learned to read Walk vs Stop
out of the motion term the canceller algebraically injects into `x`. Mismatched
(shuffled) motion gives 0.529, likewise chance.

The bare model is invariant to all three conditions (0.811 / 0.811 / 0.811),
which is the harness's own sanity check and it passes.

**The conditional probe cannot catch this.** An IMU→label shortcut lives
entirely inside the label-explained component of motion, which is precisely what
`invariance_r2_conditional` subtracts. Fixing the accuracy confound created a
blind spot for this one. Both controls are needed; neither substitutes.

**Implications**
- Every arm with `cancel` on is contaminated: `A_+cancel`, `B_full`, `B_-*`
  (all of which keep cancel except `B_-cancel`), and the `full` / `large_full`
  family in `calmnet3.json`.
- The earlier verdict "in-network motion cancellation fails"
  (`SESSION_NOTES.md:162`) was **correct**, though it was reached via the broken
  leakage metric rather than this control.
- `basis` is a sub-option of the canceller, so `A_+basis` == `A_bare` (0.811)
  when cancel is off, as observed.
- Any module that takes the motion reference as an input needs this control
  before its accuracy means anything.

---

Two results from this session. The first invalidates how the project has been
scoring itself for two days. The second is the first mechanism that improves the
decoder without the usual trade.

---

## 1. `score = balanced_accuracy - max(0, R²)` is anti-correlated with accuracy

Every ranking in this project -- the 131-variant sweep, the 18-backbone
selection, the module ablation in `exp_ablate.py`, the v3 arms -- ordered
candidates by

```
score = balanced accuracy - max(0, intent->motion R²)
```

The metric does not measure what it was built to measure.

Scope: this affects **ranking and selection**, not training. Epoch selection
inside `train_arch` uses plain validation balanced accuracy, so the optimiser
itself was never steered by the broken objective. (An earlier note here claimed
otherwise, from reading the variable name `best_score` rather than what fed it.)

**Why.** Walk and Stop differ in how much the body is moving *by definition*. So
the label itself predicts the IMU, and any decoder that tracks the label must
predict the IMU too. Raw R² rises with accuracy whether or not the decoder is
touching the artefact.

**Measured, on real data.** Take real ds007788 test epochs and real IMU
features. Generate predictions by flipping a controlled fraction of the *true
labels* -- these contain, by construction, zero information beyond the label and
cannot possibly be reading movement:

| prediction accuracy | raw R² | conditional R² |
|---|---|---|
| 0.504 | −0.004 | −0.011 |
| 0.697 | +0.103 | −0.012 |
| 0.795 | +0.257 | −0.011 |
| 0.891 | +0.471 | −0.011 |
| 0.954 | +0.685 | −0.011 |
| **1.000** | **+0.863** | −0.012 |

A **perfect decoder using no movement information scores R² = +0.863** — which
this project would have recorded as catastrophic leakage.

Pooled over sub-01/02/03 with zero-leak predictions:

```
slope of raw R² on accuracy   = +1.38
corr(accuracy, raw R²)        = +0.838
corr(accuracy, SCORE)         = -0.347      <-- NEGATIVE
```

Because the penalty grows ~1.4x faster than the accuracy term it is subtracted
from, **the score decreases as decoders get better**. Chance-level predictions
score ~0.50; perfect predictions score 0.14. The ranking was systematically
preferring worse decoders, and the epoch selector was stopping training at
whichever epoch best satisfied an anti-accuracy criterion.

**This retracts the headline.** Commit `3f43429` reports "accuracy and leakage
correlate at r = +0.958" across the v3 arms as an empirical finding about those
models. It is very largely a mathematical necessity of the metric: r = +0.838
arises with predictions that provably carry no movement information at all. The
correlation is not evidence that the architectures were cheating.

It also explains the two-day null result. "No architecture beat the ceiling" was
partly a statement about the scoring function, not about the architectures.

**The fix.** `features.invariance_r2_conditional` centres the movement features
within each class before probing, removing the component the label explains and
leaving residual motion that varies *inside* Walk and *inside* Stop. It asks the
question the project meant to ask: not "does this representation know about
movement" (unavoidable) but "does it know **more** about movement than the task
requires".

Validated both ways on synthetic data:

| | raw R² | conditional R² |
|---|---|---|
| features encoding only the label, motion driven only by the label | +0.995 | **−0.001** |
| features also carrying genuine within-class movement | +0.999 | **+0.998** |

And flat at −0.011 across the entire accuracy range on real data (table above).

**Consequence.** Every leakage number in `ablation.json`, `features.json`,
`backbone_selection.json` and `calmnet3.json` is confounded with accuracy and
should not be compared across arms of differing accuracy. Re-scoring needs the
features, so it means re-running, not re-reading.

---

## 2. A label-space dwell prior improves the decoder without the usual trade

The project's premise is that Walk/Stop are *sustained states*. Measured dwell
structure on ds007788: self-transition 0.984 (Stop) / 0.955 (Walk) at a 0.5 s
window step -- expected dwell 12–32 s, i.e. 25–65 consecutive correlated windows.
Every model built so far treats windows as i.i.d. and discards all of it.

Base pipeline held fixed (tangent space + Euclidean Alignment + L2 logistic
regression, the representation-sweep winner), 7 subjects, sessions 1-3 fit /
4-9 test.

### ds007788, 2 s windows, all arms and controls (COMPLETE)

| arm | acc | r2_dec | cond_dec | onsets/min | latency |
|---|---|---|---|---|---|
| viterbi@auto | 0.824 | +0.321 | −0.059 | 1.49 | 0.0 s |
| **forward@0.5** (no memory) | **0.818** | +0.319 | −0.056 | 3.00 | 0.0 s |
| viterbi | 0.814 | +0.317 | −0.061 | **0.87** | 0.2 s |
| forward@auto (τ=0.837) | 0.808 | +0.307 | −0.057 | 1.29 | 0.2 s |
| **forward** (τ=0.956) | 0.798 | +0.303 | −0.057 | **0.97** | 0.9 s |
| stack | 0.763 | +0.267 | −0.060 | 1.40 | 0.7 s |
| none | 0.763 | +0.274 | −0.057 | 1.87 | 0.6 s |
| **fwdshuf** (order destroyed) | **0.647** | +0.112 | −0.052 | 3.85 | 0.9 s |

`false_onsets_per_min` = spurious 0→1 *transitions* during true Stop, per minute
of standing: how often the device would lurch. Per-window error rate hides this,
because one long false run and fifty scattered false windows can score alike.
For reference, a published lower-limb BMI reports 1.45 false positives/min.

**What the controls establish — and what they take away.**

`forward@0.5` has no temporal memory whatsoever (uniform transition matrix,
verified to reduce exactly to prior-corrected emissions) and yet scores the
**second-highest accuracy of any arm, 0.818**. So the accuracy gain from
"temporal filtering" is *mostly the class-prior division the filter performs on
the way through*, not temporal structure. An earlier reading of this experiment
that credited temporal structure with the accuracy gain was wrong, and the
control is the only reason that is known.

What temporal structure uniquely delivers is **safety**, and the effect is
large: matched against `forward@0.5` at comparable accuracy (0.818 vs 0.798),
`forward` cuts spurious device activations from 3.00 to 0.97 per minute — a
**3.1× reduction** — for 0.02 accuracy and 0.9 s of latency. That is the real
contribution, and it is a contribution to the metric a patient actually feels.

`fwdshuf` collapses to **0.647**, well *below* the i.i.d. baseline, confirming
that window adjacency is doing genuine work and the filter is not merely
relabelling. `stack` gains nothing (0.763 = baseline), so feature-space temporal
context is not a substitute for a label-space prior.

### The confound, reproduced on real experimental arms

Across these eight arms:

```
corr(accuracy, r2_dec)      = +0.991      spread 0.210
corr(accuracy, r2_cond_dec) = -0.706      spread 0.010
```

The project's reported "+0.958 correlation between accuracy and leakage" is
reproduced here at +0.991 — by arms that differ only in post-hoc label
smoothing and cannot differ in what they encode. Conditional leakage is flat to
within 0.010 across a 0.18 accuracy range. The correlation is the metric.

### The representation was never leaking

Every arm scores **cond_feat ≈ −0.11**: the tangent+EA representation carries no
excess movement information beyond what Walk-vs-Stop implies. Checked that this
is not a dead metric — within-class movement variance is 13% (sub-01) to 54%
(sub-02) of total, and residual motion is genuinely predictable from other
residual motion dims (R² = +0.072 / +0.179), so the probe has real signal
available and returns negative anyway.

This undercuts the project's founding premise. The movement confound, as
measured honestly, is not present in this representation.

### Window length (never varied before; all 35 caches on disk were w2.0)

| window | none | forward | Δ |
|---|---|---|---|
| 1.0 s | 0.729 (3.79/min) | 0.775 (1.42/min) | +0.046 |
| 2.0 s | 0.763 (1.87/min) | 0.798 (0.97/min) | +0.035 |

The prior helps more at short windows, consistent with a long window already
performing crude temporal integration (step is fixed at 0.5 s, so 4 s windows
overlap 87%). Window length and temporal smoothing are therefore *not*
independent axes, which is itself worth stating.

### Replication on MoBI (8 subjects, imbalance FLIPPED — Stop is ~12% there)

| arm | acc | onsets/min |
|---|---|---|
| none | 0.675 | 6.67 |
| forward@0.5 (no memory) | 0.764 | 6.41 |
| **forward** (memory) | 0.766 | **2.09** |
| viterbi | 0.772 | **1.01** |
| viterbi@auto | 0.783 | 1.27 |
| fwdshuf | 0.550 | 6.05 |

Cleaner here than on ds007788. Temporal memory costs **nothing** in accuracy
(0.764 → 0.766) and still cuts spurious activations **3.1x**; Viterbi takes the
cohort from 6.67 to 1.01 per minute. This is the cohort where the project's
wrong-walk safety bound was most badly breached, and it is also the cohort where
the majority class is reversed — so the prior is not simply exploiting whichever
class happens to dominate.

`fwdshuf` collapses to 0.550 (near chance) with onsets unchanged at 6.05,
confirming again that adjacency, not arithmetic, is doing the work.

### Seed replication (ds007788, 4 seeds, paired within-seed)

| arm | Δacc vs baseline | same sign across seeds | Δonsets/min |
|---|---|---|---|
| forward | +0.024 ± 0.008 | yes | −1.07 |
| viterbi | +0.038 ± 0.010 | yes | −1.20 |
| forward@0.5 | **+0.040 ± 0.010** | yes | **+1.07** |
| fwdshuf | −0.113 ± 0.005 | yes | +2.16 |

Baseline 0.778 ± 0.009. The memoryless control gets the *larger* accuracy gain
while making safety *worse*: accuracy comes from the prior arithmetic, safety
comes from the memory, and they are separable.

### Budget calibration does not transfer across sessions

τ chosen on the held-out calibration split to meet 0.5 false onsets/min picks
τ = 0.837 (forward) and 0.764 (viterbi), and meets the budget *in calibration*.
On the held-out **test sessions** those arms run at 1.29 and 1.49 onsets/min —
2.6-3.0x over budget. The safety guarantee does not survive the session shift,
which is precisely the longitudinal drift this project exists to study.

This is a negative result with an obvious next move: the adaptive conformal
machinery already in `calibrate.py` (`adaptive_conformal`, verified correct
earlier) is built for exactly this — an online update that re-tightens the
threshold when the realised error rate drifts. Coupling the dwell prior to it,
rather than to a one-shot split-calibration quantile, is the untried step.

Note the plain estimated-τ arm (τ=0.956 from label counts) achieves 0.97
onsets/min without any budget machinery, beating both calibrated arms on safety.

### Online adaptive tau: fixes the drift, does not beat the simple arm

`adaptive_forward` in `exp_temporal.py` controls the dwell strength online from
a signal that exists at deployment. Adaptive conformal is the textbook answer to
the drift above, and `calibrate.adaptive_conformal` implements Gibbs-Candes
correctly, but it updates from realised coverage and therefore needs the true
label after each window -- which a worn exoskeleton never learns.

The decoder's own *switching rate* is observable without labels, and the training
labels supply the target (Stop->Walk transitions per window). Switching faster
than the wearer's own dwell statistics permit is very largely false alarms,
because genuine intent cannot exceed how fast a person actually starts and stops.
So: `tau <- tau + eta * (observed switching rate - target)` over a trailing
horizon. Causal, label-free, needs no calibration split.

Validated on a synthetic stream: tau climbs 0.900 -> 0.987 unaided, switching
rate 0.210 -> 0.010, false onsets 26.2 -> 1.44 per minute.

On real data:

| arm | ds007788 acc / onsets | MoBI acc / onsets |
|---|---|---|
| none | 0.763 / 1.87 | 0.675 / 6.67 |
| forward (tau estimated from labels) | 0.798 / **0.97** | 0.766 / **2.09** |
| forward@auto (budget-calibrated) | 0.808 / 1.29 | 0.779 / 2.59 |
| **adapt** (online, label-free) | 0.798 / 1.03 | 0.775 / 2.56 |

It reaches the hand-estimated arm's operating point on ds007788 (0.798 / 1.03 vs
0.798 / 0.97) and sits mid-pack on MoBI. It does repair the specific failure it
was built for -- `forward@auto`'s calibrated budget drifting to 1.29 on held-out
sessions -- but it does not beat plain `forward`.

**Honest verdict: a deployability result, not a performance one.** Its value is
that it needs no labels, no calibration split, and self-tunes under session
drift, while matching what label-based tau estimation achieves.

### Calibrating the prior instead of guessing it

Estimating the transition matrix is fine on ds007788 (dozens of transitions per
session) and actively dangerous on MoBI, where each recording holds one
contiguous walk block and the estimate comes out at P(stay walking) = 0.999 from
essentially one transition -- strong enough to collapse Viterbi onto the majority
class.

So τ (self-transition) is chosen from a **safety budget** on the held-out
calibration split, never on test: the smallest τ whose calibration false-onset
rate meets the budget. Smallest, because persistence is what suppresses false
onsets *and* what delays real ones -- so this minimises latency subject to the
safety constraint rather than maximising accuracy and reporting safety
afterwards. Verified on a synthetic stream: 23.5 → 0.5 false onsets/min for 1.0 s
of added latency, τ selected automatically.

### Controls

A smoothing result is easy to fake, so three:

- **`fwdshuf`** — identical arithmetic, window order permuted inside each stream
  and mapped back. Emissions untouched; only adjacency destroyed. If the gain is
  temporal it must vanish here.
- **`forward@0.5`** — uniform transition matrix. Verified numerically to reduce
  exactly to prior-corrected emissions (memoryless except the single init step,
  2/200 elements). Isolates how much of the gain is just the class-prior division
  the filter performs en route.
- **`stack`** — temporal context in feature space, which *can* import movement.

---

## 3. The module ablation, run correctly, says no module earns its place

With the conditional probe AND the two motion-input modules excluded
(`EXCLUDE_MODS=cancel,basis`), `results/ablation_clean.json`:

| module | add-one | gain | leave-out | gain | verdict |
|---|---|---|---|---|---|
| (bare) | 0.811 | | | | |
| spec_gate | 0.819 | +0.008 | 0.694 | −0.019 | KEEP |
| attn | 0.803 | −0.008 | 0.680 | −0.005 | drop |
| multiband | 0.831 | +0.020 | 0.678 | −0.003 | KEEP |
| adv | 0.783 | −0.029 | 0.687 | −0.012 | drop |
| decorr | 0.615 | −0.196 | 0.776 | −0.101 | drop |
| hsic | 0.799 | −0.012 | 0.710 | −0.035 | drop |
| art | 0.810 | −0.001 | 0.658 | +0.017 | KEEP |
| FULL | 0.675 | | | | |
| **STACKED** | **0.789** | | | | composed architecture |

**The composed architecture loses to the bare model (0.789 vs 0.811).** Every
module is individually marginal and they are jointly harmful. This is the
methodology working as intended and returning a negative.

Every disentanglement loss term (`adv`, `decorr`, `hsic`, `art`) hurts, which is
consistent with section 1: they were built to suppress a confound that the
honest probe says is not present. `decorr` is catastrophic (−0.196).

Note also `spec_gate` is invariant to the motion perturbation to three decimal
places (0.819 / 0.819 / 0.819), meaning its motion-conditioning path is inert.
It was designed to attenuate bands whose power envelope tracks the motion
envelope; it is not doing that. Its +0.008 is not the mechanism working.

**The best EEG-only decoder in this project is the bare CNN at 0.811.**

## 4. Window length (the other untried lever)

| window | none | forward | viterbi |
|---|---|---|---|
| 1.0 s | 0.729 / 3.79 | 0.775 / 1.42 | 0.788 / 1.05 |
| 2.0 s | 0.763 / 1.87 | 0.798 / 0.97 | 0.814 / 0.87 |
| 3.0 s | 0.778 / 1.01 | 0.805 / 0.61 | 0.827 / 0.73 |
| 4.0 s | 0.786 / 0.58 | 0.807 / **0.34** | 0.828 / 0.49 |

(accuracy / false onsets per minute.) Monotonic in window length on both axes,
and it composes with the prior rather than substituting for it: the 3 s baseline
(0.778, 1.01) already beats the 1 s prior-filtered arm (0.775, 1.42). Step is
fixed at 0.5 s throughout, so 4 s windows overlap 87% -- long windows are
themselves a crude temporal integrator, which is why the prior's marginal
benefit shrinks as windows lengthen (+0.046 at 1 s, +0.021 at 4 s).

## 5. The combination: dwell prior on the bare CNN (`exp_cnn_temporal.py`)

The prior operates on emitted posteriors and knows nothing about where they came
from, so it composes with any decoder. Everything above applied it to
tangent+logreg (0.778). Applied instead to the best EEG-only decoder:

| window | arm | acc | onsets/min | latency | cond R² |
|---|---|---|---|---|---|
| 2 s | none | 0.811 | 2.23 | 0.4 s | −0.051 |
| 2 s | **forward** (causal) | **0.836** | 1.13 | 0.5 s | −0.057 |
| 2 s | viterbi (offline) | **0.849** | 0.85 | 0.0 s | −0.059 |
| 2 s | forward@auto | 0.814 | **0.57** | 0.8 s | −0.055 |
| 4 s | none | 0.816 | 0.74 | 0.0 s | −0.075 |
| 4 s | **forward** (causal) | 0.827 | **0.46** | 0.3 s | −0.078 |
| 4 s | viterbi (offline) | 0.843 | 0.56 | 0.0 s | −0.080 |

### RETRACTED: the 0.836/0.849 headline was a single-seed artifact

The table above is seed 0. Replicated at seeds 1 and 2 (win=2 s), the accuracy
gain reverses sign:

| arm | acc per seed (0/1/2) | mean | Δacc | onsets per seed | Δonsets |
|---|---|---|---|---|---|
| none | 0.811 / 0.830 / 0.804 | 0.815 | — | 2.23 / 3.60 / 3.13 | — |
| forward | 0.836 / **0.770** / **0.776** | 0.794 | **−0.021 MIXED** | 1.13 / 1.38 / 1.10 | **−1.79** |
| viterbi | 0.849 / **0.774** / **0.786** | 0.803 | **−0.012 MIXED** | 0.85 / 0.76 / 0.69 | **−2.22** |
| forward@auto | 0.814 / 0.746 / 0.740 | 0.767 | −0.048 MIXED | 0.57 / 0.86 / 0.67 | −2.29 |

Seed 0 drew a weak baseline (0.811 vs 0.830) and a strong filtered arm. Averaged
over three seeds the dwell prior **costs** the CNN 0.021 accuracy.

This was predictable from the controls already in hand: `forward@0.5`, which has
no temporal memory, produced the LARGER accuracy gain on the tangent pipeline
(+0.040 vs +0.024), which already said the accuracy gain was never coming from
temporal structure. The implication should have been carried into the CNN
prediction instead of the seed-0 number being reported as a headline.

**What replicates** is the onset reduction: same sign on every seed, every
cohort, every arm, −1.79/min for causal forward and −2.22 for Viterbi. Roughly
3x fewer spurious activations.

**Corrected claim, and it is a trade rather than a win:**

> A label-space dwell prior cuts spurious walk activations ~3x at a cost of
> ~0.02 balanced accuracy. It does not improve accuracy on a strong decoder.

**Best honest operating points** (conditional leakage negative throughout, and
the bare CNN consumes no motion at inference):

| configuration | acc | onsets/min |
|---|---|---|
| bare CNN, 4 s window | **0.816** | **0.74** |
| bare CNN, 2 s (3-seed mean) | 0.815 | 3.0 |
| bare CNN, 2 s + forward (3-seed mean) | 0.794 | 1.20 |

Window length is the better lever: 4 s alone beats the 2 s prior-filtered arm on
BOTH axes simultaneously. Context: a published lower-limb BMI reports 1.45 false
positives/min.

## Status

- ds007788 window sweep (1/2/3/4 s × 4 arms): running
- ds007788 controlled pass (8 arms incl. all controls + conditional probes): running
- MoBI replication (8 arms, flipped class imbalance — Stop is ~12% there): running

## Caveats

- Single split seed (0) so far; `TEMPORAL_SEED` is settable for replication and
  a multi-seed pass has **not** yet been run. This project has been misled by a
  single-seed leaderboard before.
- Latency numbers are quantised to the 0.5 s window step.
- `forward`/`viterbi` leave the soft posterior untouched, so `r2_post` is
  meaningless for them (literally the same array as the baseline). `r2_dec` and
  the conditional probes are the comparable measures.
