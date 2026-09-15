# DriftNet: a longitudinal EEG decoder with in-network session adaptation and calibrated abstention

**DRAFT.** Numbers marked `[PENDING]` are filled from `results/` as runs complete.
No number appears here that a run has not produced.

---

## Abstract

Brain–computer interfaces for lower-limb exoskeletons face a problem that
per-window classification accuracy does not describe: a decoder is fitted once
and must keep working on the same wearer weeks later, through electrode
repositioning, impedance change and amplifier drift, while emitting decisions
safe enough to start and stop a person's legs. We present **DriftNet**, an
architecture built for that setting rather than for a benchmark. Three
properties are architectural rather than post-hoc. (i) An *adaptive alignment*
layer performs label-free session whitening **inside the network and at
inference**, continuing to update its estimate on unlabelled test data; measured
on recordings separated by weeks, it removes 60 % of the session-to-session
covariance shift. (ii) A *causal cross-epoch* pathway integrates evidence over
~15 s, matched to the 12–32 s dwell times of the walk and stop states, rather
than over a single window. (iii) A *selective head* trained under a coverage
constraint produces calibrated, abstaining decisions, so accuracy at a given
coverage is an operating point the model was trained for. We evaluate on five
outcomes — balanced accuracy, calibration error, selective accuracy, spurious
activations per minute of standing, and detection latency — with a per-component
ablation and external validation on an independent cohort recorded in a
different laboratory with different sensors and a reversed class balance.

`[PENDING: headline numbers]`

---

## 1. Introduction

The standard framing of EEG motor decoding — classify an epoch, report accuracy
— omits most of what makes exoskeleton control hard.

**Longitudinal drift.** Public benchmarks are largely single-session. Here the
decoder is fitted on sessions 1–3 and tested on sessions 4–9, recorded across
weeks, which is the deployment condition for an assistive device. Electrode
montage, impedance and skin contact all change between sessions, shifting the
signal's second-order statistics.

**Asymmetric cost.** A false Walk command moves a person's legs. Balanced
accuracy treats that identically to a missed Walk, and it cannot distinguish one
long false run from fifty scattered false windows — which feel completely
different to a wearer.

**No option to decline.** A deployed decoder that is uncertain should be able to
do nothing. A classifier trained only to maximise accuracy has no such output.

DriftNet addresses all three architecturally. Contributions:

1. **Adaptive alignment as a layer.** Euclidean Alignment is label-free and is
   normally applied offline in preprocessing. We make it a differentiable layer
   whose running covariance estimate *continues updating at test time*, giving
   unsupervised test-time adaptation to session drift. Measured drift reduction:
   **60 %** of the Riemannian distance between fit-session and held-out-session
   covariances.
2. **Cross-epoch context at the dwell timescale.** We show that attention within
   a window does not capture the task's temporal structure (and measurably hurts),
   while causal attention *across* epochs, spanning ~15 s, matches the measured
   12–32 s state persistence.
3. **A five-outcome evaluation protocol** for exoskeleton BCI, with per-component
   ablation and cross-cohort validation.

## 2. Related work and lineage

DriftNet is not independent of the EEG decoding literature and the paper states
what it inherits.

- **Convolutional EEG decoders.** ShallowFBCSPNet and Deep4Net (Schirrmeister
  et al., 2017), EEGNet (Lawhern et al., 2018). The multi-scale power stem
  descends from ShallowFBCSPNet's `square → pool → log`.
- **Attention-based decoders.** ATCNet (Altaheri et al., 2022) and EEGConformer
  (Song et al., 2022). EEGConformer already pairs a ShallowFBCSP-style stem with
  a transformer; we therefore do **not** claim "power stem + attention" as novel,
  and we verified this by building that combination and finding it equalled
  EEGConformer.
- **Sequence-over-epoch modelling** is standard in sleep staging (SeqSleepNet,
  XSleepNet, TinySleepNet). Our cross-epoch pathway is a transfer of that idea;
  the contribution is the timescale argument and its evaluation on a
  safety-critical control task, not the mechanism.
- **Euclidean Alignment** (He & Wu, 2020) as offline preprocessing; and
  Riemannian transfer methods more broadly. The novel element here is making it
  an in-network layer that adapts at inference without labels.
- **Selective prediction** (Geifman & El-Yaniv, 2019). The gate is SelectiveNet,
  used as published.

## 3. Data

**Cohort A — ds007788 (NeuroRex exoskeleton).** 7 subjects × 9 sessions across
weeks, 60-channel active EEG (actiCAP + MOVE), 100 Hz, walk/stop from the
exoskeleton's `rexstate`. 4 s windows, 0.5 s stride, ~5 031 fit windows per
subject including closed-loop trials. **Fit sessions 1–3, test 4–9** — the
evaluation measures longitudinal transfer.

**Cohort B — MoBI treadmill (Luu et al.).** 8 subjects × 3 trials, 64-channel
EEG at 100 Hz, walk/stand from treadmill phase, six goniometers. Fit trial 1,
test trials 2–3, 2 s windows. Independent laboratory, different sensors,
different task framing, and **reversed class balance** (Stop ≈ 12 % here,
majority in Cohort A) — so a decoder exploiting whichever class dominates cannot
transfer.

## 4. Architecture

Input is a sequence of K = 8 consecutive epochs at stride 3 (14.5 s at 4 s/0.5 s).
The label predicted is the **last** epoch's; all attention masks are causal.

### 4.1 Adaptive alignment

Maintains a running mean spatial covariance **M** and applies **M**^(−1/2) to the
input, with a learned gate interpolating between aligned and raw signal so the
layer can decline where alignment does not pay. Two properties matter:

- The update is **unsupervised** — it uses only the covariance of incoming
  windows — so it runs at inference.
- The estimate updates in **eval mode**, unlike BatchNorm. Freezing statistics is
  correct when train and test share a distribution; here they are weeks apart.

Validated on real held-out sessions (ds007788 sub-01), Riemannian distance
between fit-session covariance and each test session's:

| momentum | before | after | reduction |
|---|---|---|---|
| 0.05 | 7.154 | 5.331 | 25.5 % |
| **0.20** | 7.154 | **2.862** | **60.0 %** |

### 4.2 Multi-scale power stem

Three parallel temporal resolutions (64 / 128 / 256 ms), each with its own
depthwise spatial filters, then `square → pool → log`. Normalisation is applied
**before** the squaring so log-power *ratios* survive: scaling a signal by *s*
shifts log-power by a constant that a later affine layer absorbs, whereas
normalising after the square destroys the ratio.

Motivation: correcting amplitude normalisation was worth **+0.134** (Cohort A)
and **+0.174** (Cohort B) on band-power features, while a correlation-only
representation moved **+0.000** — a dose–response tracking exactly how much
marginal power each representation carries.

### 4.3 Causal cross-epoch context

A transformer over the K-epoch sequence with a causal mask. Measured state
persistence: self-transition 0.984 (Stop) / 0.955 (Walk) at 0.5 s step → 12–32 s
dwells. A fixed label-space prior exploiting this reduced spurious activations
~3× across both cohorts and four seeds; the context pathway learns it instead.

### 4.4 Selective head

Classifier plus abstention gate under a coverage constraint, with a
full-coverage auxiliary term so the backbone keeps learning from every sample.

## 5. Evaluation protocol

| outcome | why |
|---|---|
| balanced accuracy | comparability |
| expected calibration error | a device acting on confidence needs it to mean something |
| accuracy @ 90 % coverage | the trained selective operating point |
| spurious activations / min standing | what the wearer experiences |
| detection latency | the cost of persistence, reported with its benefit |

**Guard: conditional movement R².** Unconditioned nuisance-decodability is
confounded with accuracy — a decoder using *no* movement information measures
R² = +0.863 on this data — so movement leakage is measured conditional on the
class label throughout.

## 6. Results

### 6.1 Component ablation, Cohort A
`[PENDING: results/driftnet_ds.json — dn_full / dn_noalign / dn_noctx /
dn_nogate / dn_stem, five outcomes, paired within seed]`

### 6.2 External validation, Cohort B
`[PENDING: results/driftnet_mobi.json]`

### 6.3 Reference points
Same harness, full data, 2 seeds (`results/fullbench.json`):

| model | params | balanced accuracy |
|---|---|---|
| ATCNet | 45 280 | 0.879 ± 0.002 |
| EEGNeX | 58 082 | 0.876 ± 0.021 |
| ShallowFBCSPNet | 97 120 | 0.867 ± 0.010 |
| EEGConformer | 440 706 | 0.843 ± 0.009 |

## 7. Limitations

- The power stem, the epoch-sequence axis and the selective head are each
  adapted from prior work, cited in §2. The architectural novelty is the
  in-network adaptive alignment and the longitudinal formulation.
- 7 and 8 subjects. Split seed alone moves accuracy ±0.01–0.02, so all
  comparisons are paired within seed and single-seed rankings are not reported.
- Cohort B uses 2 s windows against Cohort A's 4 s, so the context pathway spans
  a shorter duration there.
- The alignment layer requires batched updates to converge; a single
  whole-recording update removes almost nothing (4.5 %).
- `[PENDING: any component failing its ablation is reported here as failing]`

## 8. Reproducibility

All experiments including failures are in the repository. Five architectural
interventions were tested and rejected by their own pre-registered controls
(amplitude side-channel; a ShallowFBCSPNet × ATCNet merge that proved equivalent
to EEGConformer; capacity scaling; an MRCP dual-generator pathway; within-epoch
attention). Configurations and results in `results/` and
`FINDINGS_temporal.md`.
