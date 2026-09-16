# Next revision — outstanding work

Author notes, not part of the manuscript. Submitted 2026-09-17 to Applied Soft
Computing as "Label-free covariance alignment for longitudinal EEG decoding",
with Suva as corresponding author. This list is for the version that follows.

---

## 0. Errors in the submitted version — fix first

These are factual mismatches between the manuscript and the code that produced
its results. None changes a result, but a reviewer attempting reproduction would
fail on them.

- **0.1 Band-pass is wrong in Table 2 and in Preprocessing.** The paper states
  0.5–40 Hz. Every drift arm uses **8–30 Hz** (`dataio.build_epochs` defaults
  `l_freq=8.0, h_freq=30.0`; the caches read are `..._8.0-30.0_...`). No script
  in the codebase uses 0.5–40 Hz. The true band is *more* artefact-protective
  than the paper claims, since it excludes the gait cycle and most EMG.
- **0.2 Optimiser is wrong in Table 2.** The paper states Adam, lr 10⁻³.
  `exp_calmnetx.py:229` uses **AdamW, lr 3×10⁻⁴, weight decay 10⁻², OneCycleLR**
  (max_lr 3×10⁻⁴).
- **0.3 Leakage-guard scope is overstated.** The injection bound ("contamination
  under 0.5 % of signal power") was measured by `exp_sensitivity.py` on the
  tangent+EA representation (clean accuracy 0.763), and the paragraph presents it
  as covering "the decoder". It does not cover the headline decoder directly.
  The evidence that *does* extend it already exists: the headline align+gate arm
  scores cond R² **−0.068** at 0.901 accuracy, while reaching ~0.90 through
  injected motion drives cond R² to **+0.067** (2 % injection). State the
  representation, and add the headline decoder's value.
- **0.4 Figure 1 contradicts the text.** The Selective Head box reads
  `0.876 @ 90 %`. The paper reports abstention as lowering accuracy
  (0.901 → 0.876, and in 14 of 18 arms). Edit in draw.io.
- **0.5 Confirm the abstract format.** The structured headings
  (Background/Problem/Solution/Significance) were removed on editorial advice.
  If Applied Soft Computing requires a structured abstract, restore them.

## 1. Closes when the running arm finishes

- **1.1 Pick one estimator (was C5).** `dn_nogate` under the trace-normalised
  estimator is training (`driftfix_ds_nogate.json`). With it, the
  trace-normalised component ablation on cohort A has every arm: `dn_stem` is
  inert to the estimator, so its 0.827 carries over. The headline can then be
  reported on one estimator throughout, the original only in its own section.

## 2. Experiments, ordered by value for the cost

Measured cost reference: one arm on cohort A takes ~31 min without the
cross-epoch transformer and ~3.4 h with it, on the RTX 2060 here (throttled at
~86 °C).

- **2.1 Artefact control (was C1). Highest priority.** Re-run align+gate on
  cohort A after (a) ICA + ICLabel or ASR, and (b) a narrower EMG-excluded band.
  Report the accuracy change and address Castermans et al. 2014 and Kline et al.
  2015 directly. This is the only experiment that settles whether walk/stop
  accuracy is neural rather than motion. The conditional probe cannot close it
  alone: as commit 381b170 records, it is blind to artefact information that is
  locked to the label with no within-class variation. ~1 h, no transformer.
- **2.2 Measure τ_drift without a decoder (was B5).** `src/exp_tau_drift.py` is
  written and its fit validated to 0.3 % on synthetic data. CPU only. Until it
  runs, the band's upper bound is inferred from where alignment stops helping,
  which is circular.
- **2.3 The align-only arm.** Not in the original plan. The selective head adds
  +0.016 on cohort A and the best configuration contains it, but that is 1.6× the
  noise floor. An align-only arm separates the head's apparent regularising role
  from its failed selection role. ~31 min.
- **2.4 Baselines on cohort B (was C3).** No published decoder has ever run on
  cohort B, so 0.792 can be called transferable but not competitive. The
  baselines are already wired; they have never been pointed at MoBI.
- **2.5 Interior rate points (was C4).** `m = 0.10` (memory 320, beside cohort
  B's τ_blk = 309) and `m = 0.02`.
- **2.6 Multi-seed the headline (was B2).** Both cohorts, five arms, three
  momenta, three seeds: ~90 arm-runs, roughly 47 h at the measured rate. An
  overnight job, and the prerequisite for 3.1, 4.2 and 4.3.
- **2.7 Single pipeline (was C2).** ATCNet, EEGNeX, ShallowFBCSPNet, EEGNet and
  CALMNet-bare under identical preprocessing, seeds and selection.
- **2.8 A third cohort (was A1). The one that changes the paper's standing.**
  Choose a public gait or MI dataset whose block structure differs from both
  cohorts. Compute τ_blk from the protocol description alone, state in advance
  whether alignment should be enabled, run once, do not tune. This converts
  Condition 1 from a hypothesis-generating explanation into a tested prediction.
  ~1 week.

## 3. Analysis on data that already exists or will

- **3.1 Paired tests on the rate tables (was B1).** Blocked until 2.6: the
  existing rate JSONs hold cohort means only. `exp_calmnetx.py` now logs
  `per_subject`, so any re-run supplies it. Already done for the temporal
  controls (`paired_tests.py`) and the normalisation family
  (`paired_zscore.py`).
- **3.2 Log the learned gate (was A2).** Record `alignment_strength()` per
  subject on both cohorts. The paper argues the gate cannot close against a
  failure absent from the fitting data. If `g` stays near its ~0.88
  initialisation on cohort B, that is confirmed; if it falls and accuracy still
  drops, the argument must be withdrawn.
- **3.3 One conservative rate for both (was A3).** Test whether a single μ above
  cohort B's τ_blk costs cohort A little. If so, the protocol switch is
  unnecessary and the paper is simpler.

## 4. Writing, after the experiments land

- **4.1 Restructure results around the finding (was D1).** State the
  class-tracking failure and the rate band first; present the decoder as the
  vehicle.
- **4.2 Merge Tables 4, 10 and 11 (was D5).** Blocked on 2.6; the merged cells
  are meant to carry seed SD.
- **4.3 Shrink the limitations (was D7).** Remove "single seed" and "no paired
  tests" once 2.6 and 3.1 land. Keep cohort size, the hypothesis-generating
  status of Condition 1, and any artefact residual 2.1 leaves.
- **4.4 Align the title with the paper's centre of gravity.** The title now
  leads with the alignment layer; the body's sharpest contribution is the rate
  condition. One introduction sentence framing the condition as what makes the
  alignment usable resolves it.
- **4.5 The IEEE variant is stale.** `ieee_calmnet.tex` still carries the
  pre-revision title, including "Parameter-Efficient" and "Abstention". Update
  or retire it.

## 5. Done in this revision

Kept for the record against the original review plan.

- Slow-rate cost on the proposed align+gate arm: **−0.085** under the
  trace-normalised estimator (0.884 → 0.799 ± 0.087). Larger than the −0.060
  previously shown on align+ctx+gate.
- Selective head resolved as a negative result: abstention lowers balanced
  accuracy in 14 of 18 arms, mean −0.045.
- No-op control explained: priming ignores momentum, and trace renormalisation
  after the gated map breaks congruence invariance. Scatter −28.8 % to +45.0 %.
- Test-stream ordering verified sequential; §6.7 stands.
- Paired Wilcoxon with Holm on the temporal-control and normalisation families.
- Proposition 1 renamed Condition 1 with assumptions (i)–(iv) stated.
- Academic-voice pass, verified style-only by fingerprint (475 numbers, 60 cite
  keys, 4 equations, 318 inline math expressions unchanged).
- Tables fitted to their columns; numbered magenta citations; 69 of 73
  references carry a verified DOI or arXiv URL.
- Retitled; authors, CRediT roles and corresponding author set; cover letter and
  highlights file produced.

## 6. Housekeeping

- **Add a per-subject progress print to `exp_calmnetx.py`.** Two lines. Tonight's
  arms trained for up to 3 h with no output between start and finish, so
  progress could only be read by attaching py-spy to the live process. Do this
  before 2.6.
- One corrupt cache predates this revision:
  `sub-04_..._w2.0_st0.5_13.0-30.0_z1_v2imu.npz`, 15.5 MB against ~146 MB
  siblings. It fails loudly with `BadZipFile`. Delete it to regenerate.
- Disk is the binding constraint on this machine. The cache build for the
  cohort-A trial splits consumed ~4 GB and filled the drive twice. Keep several
  GB free before launching 2.6.
