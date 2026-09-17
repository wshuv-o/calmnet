# Next experiments — all runs on the RTX 5080

The 2060 machine runs nothing from now on. Every job below runs on the 5080,
which holds cohort B's data. Pull before starting and push after every job.

```
git pull origin representation-beats-architecture
```

---

## 1. The goal

The manuscript is finished and pushed (`paper/cas_calmnet.pdf`, 21 pages). Its
weakness is one thing: **no component of the architecture is shown to raise
accuracy on any cohort.**

- Cohort A: the bare stem leads on the mean (0.882 over seeds 1–2), and no arm
  differs from it (Holm-corrected p ≥ 0.94, Table 4 of the paper).
- Cohort B: alignment costs 0.149 at the default rate.
- Cohort C: alignment is worth +0.002, not significant.

The job now is to find an addition that raises accuracy by a margin larger than
the seed noise, on all three cohorts. Parameter count is **no longer a
constraint**. Accuracy is the target.

---

## 2. What counts as a win

Each cohort's bar is its own seed spread, measured in `tab:compare`:

| Cohort | Participants | Seed SD (our arm) | Gain needed | Compared against |
|---|---|---|---|---|
| A (exoskeleton) | 7 | 0.026 | **+0.03** | align + gate, 0.868 |
| C (motor execution) | 20 | 0.010 | **+0.02** | align + gate, 0.796 |
| B (treadmill) | 8 | 0.006 | **+0.02** | gate only (alignment off), 0.729 |

A result counts when:

1. three data-split seeds (0, 1, 2), same sign on all three;
2. paired Wilcoxon across participants, seeds averaged within participant;
3. higher in at least 6 of 7 participants on cohort A, and p < 0.05 on cohort C;
4. the direction holds on all three cohorts and clears the bar on at least two.

**Develop on cohort A only.** Run C and B once, at the end, with no further
tuning. Tuning on C or B destroys their value as confirmation.

---

## 3. Rules that keep the existing baselines usable

The published decoders were trained in `src/exp_calmnetx.py` (pipeline C) over
seeds 0–2 and are stored in:

- cohort A: `results/bd_pipeline_c.json` + `results/bd_eegnex_s12.json`
- cohort C: `results/bd_cohort_c.json`
- cohort B: `results/bd_cohort_b.json`

Those numbers stay valid **only** while the new model uses the same cohorts,
participants, splits, seeds, preprocessing, window length and stride, optimiser,
schedule, early stopping, model selection, temperature scaling and metric code.
Add new arms; change nothing else.

Two of the experiments below (E2 pretraining, E4 augmentation) change the
**training recipe**. If one of them wins, the baselines must be rerun under the
same recipe before any comparison is published. That is the same script with
more runs, so budget for it.

---

## 4. Experiment queue, in priority order

Every job writes its own JSON under `results/` and keeps per-participant rows
(the pipeline already does this), so paired tests are possible afterwards.

### E1 — Causal temporal state model on the outputs  *(best evidence, cheapest)*

Walk and Stop persist: self-transition 0.984 (Stop) and 0.955 (Walk) at a 0.5 s
step, with dwell times of 12–32 s. Every model in pipeline C still treats
windows as independent.

Already measured in the older pipeline (`results/temporal_ds007788ctrl*.json`):
an HMM forward filter over the decoder output gave **+0.024 on cohort A over 4
seeds**, higher in 6 of 7 participants (Holm p = 0.063); Viterbi gave +0.038 but
needs the whole sequence, so it is not deployable. Shuffling window order
collapsed the gain to 0.113 **below** baseline, so it rests on real temporal
structure.

**What to do:** apply the same filter inside pipeline C, so the number is
comparable with the baselines.

- Reuse `forward_filter`, `transition_matrix` and `calibrate_tau` from
  `src/exp_temporal.py`.
- In `run()` of `src/exp_calmnetx.py`, after `p = softmax_np(lg, T)`, estimate
  the transition matrix from the **fitting** labels only, run the causal forward
  filter per stream (`strm_t` already gives stream boundaries), then take
  `pred = argmax` of the filtered posterior.
- Gate it behind an env var, e.g. `CX_SMOOTH=1`, so the unsmoothed arm still runs.
- Keep it causal. No Viterbi, no backward pass.

```
CX_MODEL=driftnet CX_COHORT=ds007788 CX_ARMS=dn_noctx,dn_stem CX_SEEDS=0,1,2 \
CX_SMOOTH=1 CX_OUT=a_smooth_3seed.json python src/exp_calmnetx.py
```

Report: smoothed vs unsmoothed, paired over participants, for both arms.

### E2 — Cross-subject pretraining, then per-subject fine-tuning  *(highest ceiling)*

Today each participant's decoder sees only that participant's own fitting
sessions, a few thousand windows. The accuracy that is missing sits in the weak
participants (sub-06 and sub-07 at 0.55–0.74, against sub-03 at 0.99).

**What to do:** pretrain one model on the other participants of the same cohort
(leave-one-participant-out), then fine-tune on the target participant's fitting
sessions. Test sessions stay untouched.

- Same cohort only at first; montages differ between cohorts (60 vs 64 channels).
- Keep every other setting identical.
- New arm name, e.g. `dn_pre`, new env var `CX_PRETRAIN=1`.

```
CX_MODEL=driftnet CX_COHORT=ds007788 CX_ARMS=dn_noctx CX_SEEDS=0,1,2 \
CX_PRETRAIN=1 CX_OUT=a_pretrain_3seed.json python src/exp_calmnetx.py
```

This changes the training recipe: if it wins, the baselines need the same
treatment (see section 3).

### E3 — Seed and architecture ensembling

Average the softmax outputs of the three seeds, and separately of
`dn_noctx` + `dn_stem` + one published decoder. Typically worth 0.01–0.03 and
costs only compute.

Requires saving per-window posteriors. Add `CX_SAVE_PROBS=1` writing
`results/probs/<cohort>_<arm>_s<seed>.npz`, then combine offline.

### E4 — Training-time augmentation

With a few thousand windows per participant this often beats any architectural
change: channel dropout, time shift, frequency masking, mixup. Apply to the
training batches only.

`CX_AUG=1`, arm `dn_noctx`, seeds 0–2, cohort A. Recipe change, same caveat as E2.

### E5 — Test-time self-training on the unlabelled session

Alignment corrects the input statistics but never moves the decision boundary.
Take high-confidence predictions on the test session (after E1's smoothing),
and update only the final linear layer. No labels are used, so it fits the
paper's label-free story.

Guard it with the confidence gate and the block-length rule, and stop if
accuracy on cohort B falls: this is the experiment most likely to reinforce its
own mistakes.

### E6 — Two cheap estimator improvements

- **Shrinkage:** Ledoit-Wolf shrinkage toward a scaled identity in
  `AdaptiveAlignment._cov` before the inverse square root. The code currently
  clamps eigenvalues at 1e-6, which is crude with 60 channels and 32-window
  batches. Alignment actively hurt sub-05.
- **Tangent-space branch:** a second branch projecting the covariance to the
  tangent space at the running mean (already maintained by the layer), then a
  linear layer concatenated before the classifier. Complements log band power.

### Not worth trying

More capacity on its own. The 595k-parameter transformer costs accuracy
(0.868 → 0.859), and a 24-configuration size sweep showed no trend.

---

## 5. Order of work

1. E1 on cohort A, seeds 0–2. Go or no-go within a day.
2. E2 on cohort A, seeds 0–2.
3. Whatever clears +0.03 on A, stack with E1 and rerun on A.
4. Then, once only: cohort C and cohort B.
5. If E2 or E4 is in the winning recipe, rerun the eight baselines under that
   recipe on all three cohorts.

---

## 6. Reporting back

After each job:

1. push the JSON under `results/`;
2. append a short entry to `results/ALL_NUMBERS.md` (regenerate with
   `python tools/collect_numbers.py`);
3. state, in the commit message: the arm, the cohort, the seeds, the mean, the
   paired difference against its control, how many participants improved, and
   the Wilcoxon p.

Do not edit anything under `paper/`. The manuscript is final; it will be updated
only when a result clears the bar in section 2.
