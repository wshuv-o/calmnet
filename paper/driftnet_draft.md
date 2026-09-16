# When in-network drift correction hurts: adaptation rate, class structure, and the limits of alignment in longitudinal EEG decoding

**Manuscript draft.** Every number is traceable to a JSON in `results/`.
Experiments that failed are reported, not omitted.

---

## Summary of claims

| claim | status | evidence |
|---|---|---|
| A label-free alignment layer removes 52 % / 84 % of session-to-session covariance drift | **supported** | 15/15 subjects, two cohorts, p <= 0.0001, no-op control at zero |
| On cohort A it is *necessary*: removing it collapses the model to its bare stem | **supported** | 0.827 = 0.827, ablation |
| The resulting model reaches 0.901 on cohort A, above every published model in the same harness | **supported** | vs ATCNet 0.879, EEGNeX 0.876, ShallowFBCSPNet 0.867, EEGConformer 0.843 |
| The layer's adaptation memory must exceed the class-block duration | **supported** | predicted threshold, step of +0.090 to +0.098 on all 3 alignment arms, 0.000 on both alignment-free arms |
| ...but must also stay short enough to track drift: the rate is **two-sided** | **supported** | the same change is +0.092 on cohort B and **-0.061** on cohort A |
| No single adaptation rate serves both cohorts | **supported** | optima are 0.01 and 0.20 respectively; each is 0.06-0.09 worse at the other's setting |
| **The architecture generalises** | **NOT supported** | on cohort B the *bare stem* beats the full model, 0.737 vs 0.626 |
| **Alignment helps decoding** | **NOT supported** | +0.051 on cohort A, -0.060 to -0.166 on cohort B |
| **Cross-epoch context helps** | **NOT supported** | reverses sign between cohorts |

The only component that transfers across both cohorts is the selective head,
which is SelectiveNet (Geifman & El-Yaniv, 2019) used as published and is not a
contribution of this work.

**The principal finding is not the architecture.** It is:

> Optimising a component against a distribution-level metric can actively harm
> the task that component is meant to serve.

Four independent demonstrations, three from this architecture and one from the
evaluation protocol that preceded it:

1. The layer's adaptation rate was selected because it maximised drift reduction
   (60 % against 25 %). That choice cost ~0.10 accuracy on cohort B.
2. The layer removes *more* drift on cohort B than on cohort A (84 % vs 52 %)
   while costing 0.166 accuracy there.
3. A verified correction to a real covariance-estimation bias moved decoding by
   +0.005 on cohort B and -0.016 on cohort A. The mechanism improved; performance
   did not follow.
   A fourth case is the sharpest, because it is a *prediction we registered and
   lost*: we predicted slowing adaptation would be neutral on cohort A, since its
   class blocks already sit far inside the fastest memory. It cost 0.061. The
   estimate had been doing useful work we had not accounted for -- tracking
   genuine drift -- and the metric we were optimising could not see it.
4. The movement-leakage probe used throughout the preceding work is confounded
   with accuracy: a decoder using no movement information whatsoever measures
   R^2 = +0.863.

The architecture is the worked example that establishes the finding.

---

## 1. Introduction

Decoding walk/stop intent from scalp EEG for lower-limb exoskeleton control is
not well described by per-epoch classification accuracy.

**Longitudinal drift.** The decoder is fitted once and must keep working weeks
later. On cohort A this is literal: fit on sessions 1-3, test on sessions 4-9,
recorded across weeks. Electrode montage, impedance and skin contact all change,
shifting the signal's second-order statistics.

**Asymmetric cost.** A false Walk command moves a person's legs. Balanced
accuracy treats that identically to a missed Walk, and cannot distinguish one
long false run from fifty scattered false windows.

**No option to decline.** A decoder that is uncertain should be able to do
nothing. A classifier trained only for accuracy has no such output.

We built DriftNet to address all three architecturally, and report both what
worked and the more informative ways in which it did not.

## 2. Related work and lineage

DriftNet is not independent of the EEG decoding literature.

- **Convolutional decoders.** ShallowFBCSPNet, Deep4Net (Schirrmeister et al.,
  2017); EEGNet (Lawhern et al., 2018). The multi-scale power stem descends from
  ShallowFBCSPNet's `square -> pool -> log`.
- **Attention decoders.** ATCNet (Altaheri et al., 2022); EEGConformer (Song
  et al., 2022). EEGConformer already pairs a ShallowFBCSP-style stem with a
  transformer, so we do **not** claim that combination as novel. We verified this
  empirically by building it and finding it equalled EEGConformer (0.848 vs
  0.843).
- **Epoch-sequence modelling** is standard in sleep staging (SeqSleepNet,
  XSleepNet, TinySleepNet). The cross-epoch pathway transfers that idea; the
  contribution here is the timescale argument, not the mechanism.
- **Euclidean Alignment** (He & Wu, 2020) applied offline in preprocessing, and
  Riemannian transfer methods generally. The novel element is making it an
  in-network layer that adapts at inference without labels -- and the finding
  that doing so introduces a failure mode the offline version cannot have.
- **Selective prediction** (Geifman & El-Yaniv, 2019).

## 3. Data

**Cohort A -- ds007788 (NeuroRex exoskeleton).** 7 subjects x 9 sessions across
weeks; 60-channel active EEG (actiCAP + MOVE) at 100 Hz; walk/stop from the
exoskeleton's `rexstate`. 4 s windows at 0.5 s stride, ~5031 fit windows per
subject including closed-loop trials. Fit sessions 1-3, test 4-9.

**Cohort B -- MoBI treadmill (Luu et al.).** 8 subjects x 3 trials; 64-channel
EEG at 100 Hz; walk/stand from treadmill phase; six goniometers. Fit trial 1,
test trials 2-3; 2 s windows. Independent laboratory, different sensors,
different task framing, and **reversed class balance** (Stop ~12 % here, majority
in cohort A).

**Temporal structure differs sharply**, and section 7.4 shows this is decisive:

| cohort | contiguous class blocks | median length | longest |
|---|---|---|---|
| A (ds007788) | 109 | 18 windows | 126 |
| B (MoBI) | 7 | 309 windows | 2211 |

## 4. Architecture

Input is a sequence of K = 8 consecutive epochs at stride 3 (14.5 s at 4 s /
0.5 s). The predicted label is the last epoch's and all masks are causal, so only
present and past are used -- the only variant a worn device can run.

### 4.1 Adaptive alignment

Maintains a running mean spatial covariance **M**, applies **M**^(-1/2), and
interpolates between aligned and raw signal through a learned gate. Two
properties distinguish it from a normalisation layer:

- The update is **unsupervised**, using only the covariance of incoming windows,
  so it runs at inference.
- The estimate updates in **eval mode**, unlike BatchNorm. Freezing is correct
  when train and test share a distribution; here they are weeks apart.

**Verified to reach the classifier.** A whitening layer placed before a BatchNorm
invites the objection that the normalisation undoes it. With identical
initialisation on held-out windows, stem outputs with and without alignment
differ by 36 % in relative magnitude and their correlation structure by 0.108.

### 4.2 Multi-scale power stem

Three parallel temporal resolutions (64 / 128 / 256 ms), each with its own
depthwise spatial filters, then `square -> pool -> log`. Normalisation precedes
the squaring so log-power *ratios* survive: scaling by *s* shifts log-power by a
constant a later affine layer absorbs, whereas normalising after the square
destroys the ratio.

Motivated by a separate result: correcting amplitude normalisation was worth
**+0.134** (cohort A) and **+0.174** (cohort B) on band-power features, while a
correlation-only representation moved **+0.000** -- a dose-response tracking how
much marginal power each representation carries.

### 4.3 Causal cross-epoch context

A causal transformer over the K-epoch sequence. Measured state persistence:
self-transition 0.984 (Stop) / 0.955 (Walk) at 0.5 s step, i.e. 12-32 s dwells.

### 4.4 Selective head

Classifier plus abstention gate under a coverage constraint, with a
full-coverage auxiliary term so the backbone keeps learning from every sample.

## 5. Evaluation protocol

| outcome | rationale |
|---|---|
| balanced accuracy | comparability |
| expected calibration error | a device acting on confidence needs it to mean something |
| accuracy @ 90 % coverage | the trained selective operating point |
| spurious activations / min standing | what the wearer experiences |
| detection latency | the cost of persistence, reported with its benefit |

**Guard: conditional movement R².** Unconditioned nuisance-decodability is
confounded with accuracy -- a decoder using no movement information measures
R² = +0.863 -- so leakage is measured conditional on the class label throughout.

## 6. The drift mechanism

Riemannian (affine-invariant) distance between the fit-session mean covariance
and each held-out session's. Riemannian rather than Frobenius because covariances
are not a vector space and Frobenius is dominated by scale, which would flatter
any whitening operation.

| cohort | n | raw | aligned | no-op control | reduction | paired t |
|---|---|---|---|---|---|---|
| A | 7 | 8.372 | 4.082 | 9.173 | **52.0 +- 4.5 %** | t = -9.14, p = 0.0001 |
| B | 8 | 6.479 | 1.074 | 6.309 | **83.9 +- 3.9 %** | t = -20.59, p < 0.00001 |

Reduced in **15/15 subjects**. The layer is primed on fit data only, then adapts
on each held-out session using nothing but incoming covariance.

The **no-op control** (momentum 0) shares every code path and never updates. It
shows no reduction in either cohort, so the effect is adaptation, not the
whitening operation and not the measurement.

## 7. Results

### 7.1 Component ablation, cohort A

Full data, 4 s windows, seed 0 (`results/driftnet_ds.json`).

| arm | components | acc | ECE | acc@90 | onsets/min |
|---|---|---|---|---|---|
| **dn_noctx** | align + gate | **0.901** | **0.039** | **0.876** | 1.64 |
| dn_full | align + ctx + gate | 0.878 | 0.074 | 0.837 | 1.26 |
| dn_nogate | align + ctx | 0.862 | 0.072 | (n/a) | 1.26 |
| dn_noalign | ctx + gate | 0.827 | 0.091 | 0.778 | 0.73 |
| dn_stem | none | 0.827 | 0.082 | 0.823 | 0.73 |

`dn_nogate`'s acc@90 is not reported: with the gate disabled the model emits
constant confidence, so ranking by it selects the first 90 % of the test set in
recording order. That number is an ordering artefact, not a selective accuracy.

**Alignment is necessary, not additive.** `dn_noalign` (0.827) equals `dn_stem`
(0.827): without alignment, context and gate together contribute nothing.

**Cross-epoch context is an operating point, not a default.** Adding it costs
0.023 accuracy and doubles calibration error while cutting spurious activations
23 %. Reported as a selectable safety mode. This is the third temporal-smoothing
mechanism in this project to show that exact trade, which suggests a property of
the task rather than of any implementation.

### 7.2 External validation, cohort B -- the architecture does not transfer

| arm | components | cohort A | cohort B |
|---|---|---|---|
| dn_noctx | align + gate | **0.901** | 0.581 |
| dn_full | align + ctx + gate | 0.878 | 0.626 |
| dn_nogate | align + ctx | 0.862 | 0.606 |
| dn_noalign | ctx + gate | 0.827 | **0.792** |
| dn_stem | none (bare stem) | 0.827 | 0.737 |

**On cohort B the bare stem beats the full model**, 0.737 against 0.626.

| component | cohort A | cohort B | transfers |
|---|---|---|---|
| adaptive alignment | +0.051 | **-0.166** | **no -- reverses** |
| cross-epoch context | -0.023 | **+0.045** | **no -- reverses** |
| selective gate | +0.016 | +0.020 | yes |

### 7.3 A rejected diagnosis

The first explanation was that the covariance estimate was dominated by the
majority class. This was real and measurable: `_cov` summed raw per-window
covariances and normalised once, making the estimate **power-weighted**. On
cohort B, walk is 87.7 % of windows and carries 1.52x the power, so it supplied
91.6 % of the estimate, which sat at Riemannian distance **0.306** from the walk
covariance -- it effectively *was* the majority class.

Per-window normalisation fixed that: the estimate moved to 3.899 / 3.194,
between the classes. **Decoding changed by +0.005 on cohort B and -0.016 on
cohort A.** The mechanism was corrected and performance did not follow, so this
diagnosis is rejected. It is reported because it is the second demonstration of
the paper's central finding.

### 7.4 The correct diagnosis: adaptation rate against class-block duration

The layer tracks whatever is currently streaming. Harmless when class blocks are
short; harmful when they are long. Cohort A has 109 blocks of median 18 windows;
cohort B has 7 blocks of median 309.

An EMA with momentum *m* has effective memory ~1/*m* batches; at batch size 32
that is 160 / 640 / 3200 windows for *m* = 0.2 / 0.05 / 0.01.

| momentum | memory | vs 309-window block | dn_full | dn_noctx |
|---|---|---|---|---|
| 0.20 | ~160 | **shorter** -- tracks the class | 0.626 | 0.581 |
| 0.05 | ~640 | longer -- tracks the session | **0.724** | 0.671 |
| 0.01 | ~3200 | longer | 0.723 | 0.696 |

**The bulk of the gain is realised as the memory crosses the block length**
(momentum 0.20 -> 0.05: +0.098 and +0.090), which is where the mechanism predicts
it. Beyond the crossing the behaviour is arm-dependent: `dn_full` is flat
(+0.001, within the +-0.005 measurement noise) while `dn_noctx` continues to rise
(+0.025).

An earlier version of this section claimed the effect "saturates immediately
after" the threshold. That was written from `dn_full` alone and the second arm
does not support it. The threshold *location* is predicted correctly and the step
is large; the sharpness of the plateau is not established, and a purely
monotonic "slower is better" account cannot be excluded for every arm on three
points.

**Full three-point sweep across all five arms**, which makes this a controlled
experiment rather than a single comparison:

| arm | m = 0.20 | m = 0.05 | m = 0.01 | at crossing | beyond | uses alignment |
|---|---|---|---|---|---|---|
| dn_full | 0.626 | 0.724 | 0.723 | **+0.098** | +0.001 | yes |
| dn_noctx | 0.581 | 0.671 | 0.696 | **+0.090** | +0.025 | yes |
| dn_nogate | 0.606 | 0.700 | 0.726 | **+0.094** | +0.026 | yes |
| dn_noalign | 0.792 | 0.782 | 0.783 | -0.010 | +0.001 | **no** |
| dn_stem | 0.737 | 0.737 | 0.737 | **0.000** | **0.000** | **no** |

Every arm that uses alignment gains +0.090 to +0.098 as the adaptation memory
crosses the class-block length. Both arms that do not use it are unaffected --
`dn_noalign` to within measurement noise, `dn_stem` *exactly*, to three decimals
across all three settings.

**Version caveat on the m = 0.20 column.** Those runs were completed before the
per-window estimator correction of section 7.3; the other two columns came after.
For `dn_noalign` and `dn_stem` this is immaterial -- alignment is disabled, so the
estimator is never invoked, which is why `dn_stem` reproduces 0.737 exactly. For
`dn_full` a matched-version baseline exists (0.631) and moves the step from
+0.098 to **+0.092**. For `dn_noctx` and `dn_nogate` no matched baseline has been
measured yet, so their +0.090 and +0.094 carry an estimator-version confound of
order 0.005 on this cohort. The conclusion does not depend on it -- the confound
is an order of magnitude below the step -- but those two cells are not yet clean,
and the matched runs are queued.

The adaptation rate therefore acts only through the component it controls, which
is what the mechanism requires and what an unrelated confound (optimisation
dynamics, regularisation, run-to-run variance) would not produce.

The alignment-free arms were measured seven times in total across momenta and
estimators (`dn_noalign` 0.792 / 0.788 / 0.783 / 0.782; `dn_stem` 0.737 three
times), fixing measurement noise at **+-0.005** and making the -0.060 residual in
section 7.5 twelve times the noise.

**A one-sided rule would be wrong.** The obvious reading of this table is "set
the adaptation memory longer than the class-block duration, and when in doubt go
slower." Cohort B alone supports it. Section 7.5 tests it on cohort A and it
fails: the same change that recovers +0.092 there costs -0.061 here. The rule is
two-sided, and we state it in its corrected form in 7.5 rather than here.

### 7.5 The rate is two-sided, and that explains the residual

Applying the same momentum change to cohort A produces the **opposite** result:

| cohort | m = 0.20 | m = 0.01 | delta |
|---|---|---|---|
| A (ds007788), class blocks 18 windows | **0.862** | 0.801 | **-0.061** |
| B (MoBI), class blocks 309 windows | 0.631 | **0.723** | **+0.092** |

Both cells in each row come from the same code version (the corrected per-window
estimator of section 7.3). An earlier draft of this table quoted -0.077 and
+0.097 by taking the m = 0.20 baselines from runs that predate that correction;
the direction and magnitude survive, the exact deltas do not. The mixed-version
numbers should not be cited.

We predicted cohort A would be roughly neutral, on the grounds that its class
blocks sit far below even the fastest setting's memory so the estimate was
already class-mixed. It is not neutral: slowing adaptation costs 0.061 there.
**There is no universal setting, and error in either direction costs 0.06-0.09.**

The mechanism is therefore two-sided. The running estimate tracks whatever varies
on its own timescale, and two different things vary:

| adaptation rate | tracks session drift | tracks the streaming class |
|---|---|---|
| fast | **yes** -- the intended behaviour | yes, *if class blocks are long* -- the failure |
| slow | no -- the benefit is lost | no |

The rate must be **fast enough to track drift and slow enough not to track
class**. Cohort A satisfies both: 18-window class blocks leave a wide admissible
band, and fast adaptation captures genuine weeks-long drift, so momentum 0.2 is
correct there and slowing costs the drift tracking.

**This explains the -0.060 residual on cohort B** without a third mechanism. Its
309-window class blocks force the rate slow enough to avoid class tracking -- at
which point the estimate is also too slow to track drift. The layer has nothing
left to contribute and adds only estimation noise. The residual is not an
unexplained remainder; it is what the layer costs when no admissible rate exists.

**Stateable condition.** A valid adaptation rate exists only where the
class-block timescale is shorter than the drift timescale. Where class blocks are
long relative to drift, the admissible band is empty and the layer should be
omitted rather than tuned. Both quantities are measurable from the recording
protocol before any model is fitted.

This also sharpens why offline Euclidean Alignment does not encounter the
failure: a whole-recording estimate is maximally slow, so it never tracks class
-- and it never tracks within-recording drift either, which is precisely the
capability an in-network version is introduced to add.

### 7.6 Reference points

Same harness, full data, 2 seeds (`results/fullbench.json`):

| model | params | balanced accuracy |
|---|---|---|
| ATCNet | 45 280 | 0.879 +- 0.002 |
| EEGNeX | 58 082 | 0.876 +- 0.021 |
| ShallowFBCSPNet | 97 120 | 0.867 +- 0.010 |
| CALMNet-bare (ours, prior) | 3 946 | 0.858 +- 0.020 |
| EEGConformer | 440 706 | 0.843 +- 0.009 |

## 8. Discussion: mechanism validation is not performance validation

Domain adaptation, invariance learning and transfer methods are routinely
justified by a distribution-level quantity -- a divergence, an alignment
distance, a nuisance-decodability score -- with task performance assumed to
follow. This work provides four counterexamples within a single system, three of
which we produced by acting on that assumption ourselves:

1. A hyperparameter tuned to maximise drift reduction cost ~0.10 accuracy.
2. The cohort where *more* drift was removed (84 % vs 52 %) is the cohort where
   accuracy fell (-0.166).
3. Correcting a real, measured estimator bias changed decoding by +0.005 / -0.016.
4. The leakage probe used to judge invariance rises with accuracy whether or not
   the decoder touches the nuisance (R² = +0.863 for a decoder that cannot).

5. We predicted, on record and before running it, that slowing adaptation would
   be neutral on cohort A because its class blocks already sit far inside the
   fastest memory. It cost 0.061 -- the prediction failed because it accounted
   only for the harm the estimate can do and not for the benefit it was
   delivering.

The common structure: each metric is a property of the *representation's
distribution*, and each was improved successfully. None predicted the task
outcome, and optimising against them was actively harmful in three of five cases.

The fifth is the one that completes the picture. A distribution-level metric is
not merely an unreliable proxy for task performance -- it is typically
*one-sided*, scoring a single failure it was designed to detect. Drift reduction
measures how much non-stationarity the layer removes; it has no term for what the
layer destroys. Our block-length diagnosis inherited that one-sidedness: it
correctly identified when adaptation is too fast, and was silent on when it is
too slow. Only testing the correction on a cohort where it should have been free
revealed the other side. **A proxy metric can be right about its own failure mode
and still mislead, because the quantity that matters is the balance of two
effects and the metric sees one.**

## 9. Limitations

- The power stem, epoch-sequence axis and selective head are each adapted from
  cited prior work. The architectural novelty is the in-network adaptive
  alignment and the longitudinal formulation.
- Seven and eight subjects. Split seed alone moves accuracy +-0.01-0.02; all
  comparisons are paired within seed and single-seed rankings are not reported.
  Ablations here are seed 0; multi-seed replication is incomplete.
- Cohort B uses 2 s windows against cohort A's 4 s, so the context pathway spans
  a shorter duration there.
- The two-sided account is fitted to two cohorts with opposite block structure;
  it predicts an interior optimum, but we have not measured one. A cohort whose
  admissible band is narrow but non-empty would test it properly.
- The per-window covariance correction was regression-tested on cohort A with
  one arm only (dn_full, 0.878 -> 0.862).
- Cross-epoch context and alignment both reverse sign between cohorts; neither
  should be deployed on a new protocol without re-validation.

## 10. Submission readiness -- FOR THE AUTHORS, NOT FOR REVIEW

An honest assessment of what this manuscript can currently sustain, and what it
would take to reach a first-quartile venue (JNE, IEEE TNSRE, NeuroImage).

**Blocking gaps**

| gap | why it blocks | work required |
|---|---|---|
| Ablations are seed 0 only | split seed moves accuracy +-0.01-0.02, the same order as several reported effects. The headline ablation must be multi-seed or a reviewer will discount it. | 3 seeds x 5 arms x 2 cohorts, ~2 days compute |
| The central finding is demonstrated only on our own code | sections 7.3, 7.5 and 8 show *we* made these errors. To claim the field does, published work using unconditioned nuisance probes or fast test-time adaptation must be identified and cited. Searches so far have NOT established this. | 1-2 days literature work |
| The design rule is post-hoc | section 7.4 explains cohort B after seeing it fail. **Partly addressed:** we registered a prediction for cohort A before running it and it was falsified, which produced the two-sided rule in 7.5. That is a real out-of-sample test, but a failed one -- the *corrected* rule has still never predicted anything in advance. | ~1 week: obtain a public gait/MI dataset with different block structure, state the prediction from block structure alone, run once |
| ~~The -0.060 residual is unexplained~~ | **CLOSED.** Section 7.5: cohort B's block structure leaves no rate that both avoids class-tracking and tracks drift, so a correctly configured layer has nothing left to contribute. | done |

**What the paper cannot be.** An architecture paper. DriftNet fails external
validation and its novel component reverses sign; that is documented here and in
the repository and cannot be presented otherwise.

**What it can be.** A methods-and-cautionary paper, with the architecture as the
worked example. The four demonstrations in section 8 share one structure and were
produced by acting on the assumption they refute, which is stronger than
observing it in someone else's system.

**Realistic positioning today:** workshop paper, or second-quartile journal.
**After the seeds and the literature grounding:** plausible at JNE or TNSRE.
**With the third-cohort prediction confirmed:** the design rule becomes a tested
hypothesis rather than an explanation, which is the version worth submitting.

## 11. Reproducibility

All experiments including failures are in the repository. Five architectural
interventions were tested and rejected by their own pre-registered controls: an
amplitude side-channel (shuffle control matched it exactly), a
ShallowFBCSPNet x ATCNet merge (equivalent to EEGConformer), capacity scaling
(19k-441k parameters, no effect), an MRCP dual-generator pathway (0.604
single-trial against 0.775 for ERD; combining them scored worse than ERD alone),
and within-epoch attention (+0.024 to remove). Configurations and results are in
`results/` and `FINDINGS_temporal.md`.
