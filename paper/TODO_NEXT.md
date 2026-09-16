# Next revision — outstanding work

Author notes, not part of the manuscript. Ordered by what a reviewer is most
likely to demand first. Items marked DONE were closed in the current revision
and are kept so the list stays readable against the original review plan.

---

## A. Ideas raised in discussion, not yet in the paper

These came out of reading the cohort-A/cohort-B split and are the most
promising directions, not merely gap-filling.

### A1. Make the protocol switch a tested prediction, not an explanation

The band `tau_blk < mu < tau_drift` currently picks the right answer on both
cohorts, but it was written knowing both answers. That is explanation, not
prediction.

**What to do:** obtain a third public gait or MI dataset with block structure
different from both cohorts. Compute `tau_blk` from the protocol description
alone. State in advance whether alignment should be enabled. Run once. Do not
tune afterwards.

This is the single change that moves the contribution from post-hoc rule to
tested hypothesis, and `SUBMISSION_READINESS.md` reached the same conclusion
independently.

### A2. The learnable gate cannot learn the switch — quantify it

Section on external validation now argues that `g = sigmoid(gate)` cannot close
against a failure that is absent from the fitting distribution. The argument is
currently structural. It should be measured:

- Log the learned `g` per subject on both cohorts (`alignment_strength()`
  already exists on the layer; it is not recorded in any results JSON).
- Prediction: `g` stays near its initialisation (~0.88) on cohort B despite
  alignment costing 0.211 there. If `g` does fall substantially and accuracy
  still drops, the structural argument is wrong and must be withdrawn.

Cheap — no new training if logging is added to an existing run.

### A3. Can one configuration be stable across both without the switch?

Open question, worth one experiment before claiming the switch is necessary.
Candidates:

- Set `mu` above `tau_blk` for the *worst* expected protocol rather than per
  cohort, and measure the cost on cohort A. If a single conservative `mu` gives
  up little on A while fixing B, the switch is unnecessary and the paper is
  simpler.
- Freeze the running estimate at test time (adapt only across sessions, never
  within), which removes class tracking by construction while keeping drift
  tracking. Not yet tested.

Note the current evidence is against a single setting existing: on cohort B the
band is empty, so no `mu` both avoids class tracking and tracks drift. But that
conclusion rests on `tau_drift` being inferred rather than measured — see B5.

---

## B. Tier 1 from the review plan — experiments

- **B1. Per-participant rows for the rate tables.** The rate-sweep JSONs store
  cohort means only. `exp_calmnetx.py` now emits `per_subject` (added this
  revision), so any re-run yields them. Until then no paired test is possible on
  Tables 4/10/11.
  *Partially DONE:* paired Wilcoxon computed for the temporal-control family
  (`src/paired_tests.py`) and the normalisation family (`src/paired_zscore.py`),
  both from data already on disk.
- **B2. Multi-seed the headline.** Table 4, both cohorts, five arms, three
  momenta, three seeds minimum. Report mean +- seed SD. ~90 arm-runs; one arm
  measured at >23 min on an RTX 2060, so budget accordingly — this is an
  overnight job, not an afternoon one.
- **B3. Slow-rate cost on align+gate, cohort A.** The -0.060 is currently shown
  only on align+ctx+gate; the proposed configuration must be the one tested.
  *Launched this session (`driftmom_ds_0.01_noctx.json`), did not finish.*
- **B4. Selective head.** *DONE as a negative result:* abstention lowers
  balanced accuracy in 14 of 18 arms, mean -0.045. Removed from title, abstract
  and highlights. **Still open:** whether a temperature-scaled softmax threshold
  works where the learned gate does not. If it does, abstention can return as a
  contribution.
- **B5. Measure `tau_drift` without a decoder.** Script written this session
  (`src/exp_tau_drift.py`): per-session covariance per 32-window block,
  Riemannian distance against lag, fit `d(L) = d_inf (1 - exp(-L/tau))`. Fit
  validated to 0.3% on synthetic data. **Not yet run on real data** — deferred
  for memory reasons while training occupied the GPU. Until this runs, the upper
  bound of the band is inferred from where alignment stops helping, which is
  circular.
- **B6. Explain the no-op control.** *DONE:* it primes from the first batch
  regardless of momentum, so at momentum 0 it whitens by the fit covariance
  rather than the identity; the gated blend makes it a fixed linear map whose
  congruence-invariance is broken by the per-matrix trace normalisation applied
  after it. Scatter is -28.8% to +45.0% and is measurement noise, not adaptation.

## C. Tier 2 from the review plan

- **C1. Artefact control.** Re-run align+gate on cohort A after (a) ASR or
  ICA+ICLabel, and (b) an 8--30 Hz EMG-excluded band. Cite and address
  Castermans et al. 2014 and Kline et al. 2015 directly.
- **C2. Single pipeline.** ATCNet, EEGNeX, ShallowFBCSPNet, EEGNet and
  CALMNet-bare under identical preprocessing, seeds and model selection.
- **C3. Baselines on cohort B.** Not in the original plan but now a stated
  limitation: no published decoder has ever been run on cohort B, so 0.792
  cannot be called competitive. The baselines are already wired up; they have
  simply never been pointed at MoBI. Cheap and high value.
- **C4. Interior rate points.** `m = 0.10` (memory 320, near cohort B's
  `tau_blk = 309`) and `m = 0.02` give a fourth and fifth point on the curve.
- **C5. Pick one estimator.** Report the trace-normalised version as the main
  result everywhere. **Blocked:** `driftfix_ds.json` has 3 of 5 arms;
  `dn_nogate` has never been run with the fixed estimator. One arm.
  *Launched this session (`driftfix_ds_nogate.json`), did not finish.*
- **C6. Test-stream ordering.** *DONE:* `infer()` walks the index sequentially
  in batches of 32 with no shuffle; the `shuffle=True` is on the training loader
  only. Section 6.7 stands, no re-run needed.

## D. Tier 3 — writing

- **D1. Restructure results around the finding.** State the class-tracking
  failure and the rate band first, then present the decoder as the vehicle.
  Not attempted — too large to do safely at speed.
- **D2. Retitle.** *DONE.*
- **D3. Proposition -> Condition with explicit assumptions.** *DONE.*
- **D4. Cut self-narration, aim 20% shorter.** *Partially DONE:* three passages
  cut. The 20% reduction target is not met.
- **D5. Merge Tables 4/10/11 into one.** Blocked on B2 — the merged cells are
  meant to carry seed SD, which does not exist yet.
- **D6. Fix specific errors.** *DONE:* author fields; Table 5/6 cross-reference
  (Table 6 was referenced nowhere); Viterbi flagged offline-only; Figure 3b
  caption, which claimed accuracy was "recovered" by abstention; the
  +0.028/+0.051 pair traced to its runs. No error was found in the last of
  these — both endpoints verify.
- **D7. Limitations should shrink after Tier 1.** Cannot shrink yet; grew by one
  entry (no cohort-B baseline). Removing "single seed" depends on B2.

---

## E. Housekeeping

- One corrupt cache predating this session:
  `sub-04_..._w2.0_st0.5_13.0-30.0_z1_v2imu.npz`, 15.5 MB where siblings are
  ~146 MB, truncated by an earlier disk-full event. Fails loudly with
  `BadZipFile` rather than returning wrong numbers. Delete to regenerate.
- Disk is the binding constraint on this machine, not compute. Cohort A caches
  are now built (~7.7 GB total), so re-runs go straight to training.
- CRediT role split in `cas_calmnet.tex` is marked `% TODO(authors)` and needs
  the authors to confirm who did what.
