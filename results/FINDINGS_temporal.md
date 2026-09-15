# The selection metric was inverted, and temporal structure is the untried lever

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
