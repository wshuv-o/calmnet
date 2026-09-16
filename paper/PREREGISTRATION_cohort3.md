# Pre-registered prediction — third cohort

Written and committed **before any model was trained or evaluated on this
cohort**. The git commit that adds this file is the timestamp. Nothing below
may be revised after results exist; any deviation from it is reported as a
deviation.

## Cohort

PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB; Schalk et al. 2004;
Goldberger et al. 2000). Independent of both existing cohorts: different
laboratory, amplifier, montage, task and population.

- Subjects: S001–S020, every subject whose files load. None excluded on the
  basis of results.
- Runs: motor **execution** only, 3, 5, 7, 9, 11, 13. Imagery runs excluded.
- Task: binary, **rest (T0) against movement (T1 ∪ T2)**, the closest analogue
  of walk against stop.
- Split: fit on runs 3, 5, 7, 9; test on runs 11, 13. A grouped 30 % of the fit
  data is held out for model selection, as for the other cohorts.
- Preprocessing: identical to cohort B. 8–30 Hz zero-phase fourth-order
  Butterworth, resample to 100 Hz, 64 channels, 2 s windows at 0.5 s step. A
  window is labelled only if it lies wholly inside one annotation; boundary
  windows are dropped.
- Model and training: identical to cohorts A and B. No hyperparameter is tuned
  on this cohort. Seed 0.

## The quantity Condition 1 needs, from the protocol alone

The published protocol alternates rest and task in periods of about 4.1–4.2 s.
With 2 s windows at 0.5 s step and boundary windows dropped, a class block
holds about

    tau_blk ≈ (4.2 − 2.0) / 0.5 + 1 ≈ 5 windows.

The estimator memory at the default momentum is mu ≈ 32 / m = 160 windows at
m = 0.2, and 3200 at m = 0.01.

So tau_blk ≈ 5 ≪ mu = 160. The running estimate spans roughly thirty
alternating blocks and cannot converge to either class. Condition 1's lower
bound is satisfied by a factor of about thirty.

## Predictions

Arms, all without the cross-epoch transformer:

- `gate`: no alignment (reference)
- `align+gate`, m = 0.2
- `align+gate`, m = 0.01

**P1 — no class-tracking loss at the default rate.**
mean over subjects of [ acc(align+gate, m = 0.2) − acc(gate) ] **> −0.05**.

Reason: class tracking on cohort B cost −0.211 against no alignment. A
class-tracking failure is large and negative; −0.05 is five times the cohort-A
noise floor. **Falsified if the mean difference is ≤ −0.05.**

**P2 — slowing adaptation does not help.**
mean over subjects of [ acc(align+gate, m = 0.01) − acc(align+gate, m = 0.2) ]
**< +0.05**.

Reason: on cohort B, slowing to m = 0.01 recovered +0.090 by ending class
tracking. Here there is no class tracking to end, so no comparable recovery is
expected. This is the prediction that discriminates between the regimes: the
same intervention is predicted to help on B and not here. **Falsified if the
mean difference is ≥ +0.05.**

Accuracy is balanced accuracy on the test runs, averaged over subjects.
Per-subject values, the count of subjects in each direction, and a paired
Wilcoxon signed-rank test are reported for both, but the confirm/falsify
decision is the mean threshold stated above and nothing else.

## Scope, stated in advance

This tests the short-block side of Condition 1, the same regime as cohort A,
together with the contrast against cohort B through P2. It does not test an
empty band, which needs a protocol whose blocks outlast drift. A confirmed P1
and P2 would show that the condition predicted a new cohort correctly in
advance; they would not by themselves establish the upper bound.
