# ATCNet-X: restoring and extending a windowed-attention decoder for longitudinal lower-limb exoskeleton BCI

**Status: DRAFT — numbers marked `[PENDING]` are filled from results/ as runs complete.**
**Every claim in this draft must be traceable to a JSON in results/. No number is written here that a run has not produced.**

---

## Abstract

Decoding walk/stop intent from scalp EEG for lower-limb exoskeleton control is
a longitudinal, safety-critical problem: the decoder must work across sessions
separated by weeks, and a spurious activation moves a person's legs. We make
three contributions. First, we show that ATCNet (Altaheri et al., 2022), the
strongest published decoder on this task, silently loses its defining mechanism
when ported to a lower sampling rate: its temporal hyperparameters are specified
in samples and tuned at 250 Hz, so at 100 Hz a compatibility fallback reduces its
windowed attention ensemble from five windows to three (and to one at 2 s
epochs), while still producing a model that trains and scores well. Second, we
introduce ATCNet-X, which restores the intended parameterisation and adds two
modules motivated by measured properties of the task: causal cross-epoch
attention spanning ~15 s, matched to the 12–32 s dwell times of the walk and
stop states, and a coverage-constrained selective head. Third, we evaluate on
five outcomes rather than accuracy alone — balanced accuracy, calibration error,
selective accuracy at 90 % coverage, spurious activations per minute of standing,
and detection latency — with a per-module ablation and external validation on an
independent cohort.

`[PENDING: headline numbers]`

---

## 1. Introduction

`[TODO: frame the exoskeleton BCI problem, longitudinal drift, and why
per-window accuracy is the wrong currency for a device that starts and stops a
person's legs.]`

Contributions:

1. A reproducibility finding: ATCNet's sample-specified temporal
   hyperparameters do not transfer across sampling rates, and the framework
   fallback that adapts them removes the windowed attention ensemble the model
   is named for. On 4 s epochs at 100 Hz, `n_windows` falls 5 → 3; on 2 s
   epochs, 5 → 1.
2. ATCNet-X, an ATCNet-derived decoder adding rate-matched parameterisation,
   causal cross-epoch context at the dwell timescale, and a selective head.
3. A five-outcome evaluation protocol for exoskeleton BCI, with per-module
   ablation and cross-cohort validation.

## 2. Related work

`[TODO. Must cite and position honestly:]`
- ATCNet (Altaheri et al. 2022) — the base architecture. ATCNet-X is derived
  from it, not independent of it.
- EEGConformer (Song et al. 2022) — ShallowFBCSPNet front end + transformer;
  the closest relative to any "power front end + attention" design, and the
  reason we do NOT claim that combination as novel.
- ShallowFBCSPNet / Deep4Net (Schirrmeister et al. 2017), EEGNet (Lawhern
  et al. 2018), EEGNeX, EEGITNet, BDTCN — the benchmark set.
- Sequence-over-epoch modelling in sleep staging (SeqSleepNet, XSleepNet,
  TinySleepNet) — the cross-epoch context module is a transfer of an
  established idea from that literature, not a new mechanism. Say so.
- Selective prediction (Geifman & El-Yaniv 2019) — the gate is their
  SelectiveNet objective, used as published.
- Gait-EEG artefact debate: Castermans et al. 2014 (substantial contamination)
  vs Nathan & Contreras-Vidal 2016 (negligible with active electrodes);
  Jacobsen et al. 2021.

## 3. Data

**Cohort A — ds007788 (NeuroRex).** 7 subjects, 9 sessions each spanning weeks,
60-channel active EEG (actiCAP + MOVE), 100 Hz, walk/stop labelled from
exoskeleton `rexstate`. Head- and exoskeleton-mounted IMU as the movement
reference. Train on sessions 1–3, test on 4–9 — train-early / test-late, so the
evaluation measures longitudinal transfer, not within-session fit. Fit set
~5 031 windows/subject including closed-loop trials; 4 s windows at 0.5 s stride.

**Cohort B — MoBI treadmill (Luu et al.).** 8 subjects, 3 trials, 64-channel
EEG at 100 Hz, walk/stand from treadmill phase, six goniometers as the movement
reference. Fit on trial 1, test on trials 2–3. 2 s windows (this cohort's
protocol). **Class imbalance is reversed** relative to Cohort A: Stop is ~12 %
here and the majority there — so a model that merely exploits the dominant class
cannot transfer.

## 4. Method

### 4.1 Base architecture

ATCNet as published: a convolutional stem (temporal → depthwise spatial →
BatchNorm → pooling), a sliding-window ensemble over the resulting feature map,
multi-head attention and a TCN per window, and per-window classifiers combined
by averaging.

### 4.2 The sampling-rate defect

ATCNet's temporal hyperparameters are sample counts tuned on BCI-IV-2a
(22 channels, 1125 samples, 250 Hz). Read as durations: a 256 ms temporal
filter, a 64 ms second filter, and 8 × 7 pooling giving ~224 ms feature frames,
so 1125/56 ≈ 20 frames over which 5 sliding windows attend.

At 100 Hz the same counts stretch every receptive field by 2.5×, and the input
falls below the minimum the configuration requires (616 samples), triggering a
compatibility fallback that rescales kernels, pooling, `n_windows` and
`tcn_kernel_size`. The resulting model trains and scores well, which is why the
degradation is easy to miss.

| | 250 Hz (design) | 100 Hz, 4 s | 100 Hz, 2 s |
|---|---|---|---|
| `n_windows` | 5 | **3** | **1** |
| `tcn_kernel_size` | 4 | 2 | 2 |
| temporal kernel | 64 (256 ms) | 41 (410 ms) | — |

**`rate` module.** Match durations rather than sample counts: 26-sample
(256 ms) and 6-sample (64 ms) kernels, pooling chosen to preserve ~20 feature
frames. This also drops the minimum input below the available length, so no
fallback fires and `n_windows = 5`, `tcn_kernel_size = 4` survive.

### 4.3 Cross-epoch context

ATCNet attends over windows *inside* one epoch. The task's temporal structure is
an order of magnitude longer: measured self-transition probability is 0.984
(Stop) and 0.955 (Walk) at a 0.5 s step, i.e. 12–32 s dwells. A fixed
label-space dwell prior exploiting this reduced spurious activations roughly
threefold across both cohorts and four seeds.

**`ctx` module.** A causal transformer over a sequence of K = 8 consecutive
epochs at stride 3, spanning 14.5 s at 4 s/0.5 s. The mask is causal and the
predicted label is the last epoch's, so only present and past are used — the
only variant deployable on a worn device. Contexts never cross recording
boundaries; positions near a stream start are edge-padded.

This is a transfer of epoch-sequence modelling from sleep staging, not a new
mechanism. Its contribution here is the timescale argument and the demonstration
that within-epoch attention does not substitute for it.

### 4.4 Selective head

**`gate` module.** A SelectiveNet head (Geifman & El-Yaniv 2019) trained under a
coverage constraint, jointly with the classifier plus a full-coverage auxiliary
term that keeps the backbone learning from every sample. Abstention is learned
rather than thresholded onto a finished classifier, making accuracy@coverage an
operating point rather than a post-hoc sweep.

### 4.5 Evaluation

Five outcomes, three of which the model is trained for:

| outcome | why it is reported |
|---|---|
| balanced accuracy | comparability with prior work |
| expected calibration error | a device acting on confidence needs confidence to mean something |
| accuracy @ 90 % coverage | the selective operating point |
| spurious activations / min of standing | what a wearer experiences; per-window error hides it, since one long false run and fifty scattered false windows score alike |
| detection latency | the cost of any persistence mechanism, reported alongside the benefit |

Plus a guard: **conditional movement R²** — movement decodability from the
emitted decision *beyond what the class label already implies*. Unconditioned
nuisance-decodability is confounded with accuracy (a decoder using no movement
information whatsoever measures R² = +0.863 on this data), so the conditional
form is used throughout.

## 5. Results

### 5.1 Module ablation, Cohort A

`[PENDING: results/atcplus.json — base / rate / rate+ctx / rate+gate / full
× seeds, five outcomes each, paired within seed]`

### 5.2 External validation, Cohort B

`[PENDING: results/atcplus_mobi.json]`

### 5.3 Benchmark context

Full-data comparison, 4 s windows, 2 seeds, identical harness
(`results/fullbench.json`):

| model | params | balanced accuracy |
|---|---|---|
| ATCNet | 45 280 | 0.879 ± 0.002 |
| EEGNeX | 58 082 | 0.876 ± 0.021 |
| ShallowFBCSPNet | 97 120 | 0.867 ± 0.010 |
| EEGConformer | 440 706 | 0.843 ± 0.009 |

## 6. Limitations

State plainly, do not bury:

- The `ctx` module is an application of epoch-sequence modelling established in
  sleep staging; the novelty is the timescale argument and its evaluation here,
  not the mechanism.
- The `gate` is SelectiveNet, used as published.
- The `rate` fix is a porting correction. Its interest is the *silent* nature of
  the degradation, not the difficulty of the fix.
- Cohort B uses 2 s windows against Cohort A's 4 s, so the context module spans
  a shorter duration there; the two cohorts are not window-matched.
- Seven and eight subjects respectively. Split seed alone moves accuracy by
  ±0.01–0.02, so all comparisons are paired within seed and single-seed
  rankings are not reported.
- `[PENDING: any module that fails its ablation must be reported here as
  failing, not omitted]`

## 7. Reproducibility

All experiments, including failed ones, are in this repository. Negative results
retained deliberately: five architectural interventions were tested and rejected
by their own pre-registered controls (amplitude side-channel, a
ShallowFBCSPNet × ATCNet merge, capacity scaling, an MRCP dual-generator
pathway, and within-epoch attention). Their configurations and results are in
`results/` and `FINDINGS_temporal.md`.
