# Overnight log — 2026-09-17

Autonomous session while the authors sleep. Presentation of the updated
manuscript at 10:00. Every decision is recorded here with its reason, in the
order it was taken, so the morning read is a full account rather than a
summary of outcomes.

## Starting state (01:53)

- Submitted version: `b022e49`-era manuscript, desk rejection expected.
- Running: `dn_nogate`, trace-normalised estimator, cohort A. 4 of 7 subjects
  done at 01:31, ETA ~02:36 by measured rate.
- Disk: C: 1.5 GB free (too tight for caches). E: 21 GB free.
- RAM: 4.3 GB free while the running arm holds the rest.
- GPU: RTX 2060, throttling at ~86 °C. One job at a time.
- Tooling: `mne` 1.12 present; `mne_icalabel`, `meegkit`, `asrpy` absent.

## Plan and priorities

Ordered by value for the presentation, subject to one GPU:

1. Fix the four mismatches between the submitted manuscript and the code
   (TODO_NEXT.md section 0). Text only, no GPU.
2. Fold in the running arm when it lands: completes "pick one estimator".
3. Artefact control: align+gate on cohort A after ICA + ICLabel. Highest
   scientific value; the only test that settles whether walk/stop accuracy is
   neural.
4. Align-only arm: separates the selective head's two roles.
5. Third cohort with a prediction registered before running. The largest
   change to the paper's standing.
6. tau_drift, decoder-free, on CPU in parallel.
7. Integrate, rebuild the manuscript and zip, prepare presentation material.

## Decisions

### 01:56 — Methods now describe the code that produced the results

Checking the pipeline before running anything found that Table 2 and the
Preprocessing paragraph did not match the code. Verified line by line against
`exp_calmnetx.py`, `dataio.py`, `dataio_mobi.py` and `atcnet_plus.py`:

| setting | submitted paper | actual code | fixed |
|---|---|---|---|
| band-pass | 0.5-40 Hz | 8-30 Hz, both cohorts | yes |
| filter | "MNE-Python" | A: zero-phase FIR (MNE); B: zero-phase 4th-order Butterworth | yes |
| artefact step | "none" | cohort A regresses EOG out of the EEG | yes |
| optimiser | Adam, lr 1e-3 | AdamW, lr 3e-4, wd 1e-2, OneCycle | yes |
| early stopping | not stated | patience 12 | added |
| gradient clipping | not stated | norm 1.0 | added |
| coverage target | 0.90 | 0.90 (atcnet_plus default) | correct, unchanged |

No script in the codebase uses 0.5-40 Hz. The true band is more
artefact-protective than the paper claimed: it excludes the ~1-2 Hz gait cycle
and most EMG above 30 Hz.

Leakage-guard paragraph rescoped. The injection calibration was measured on the
tangent-space representation (clean 0.763), not the headline decoder, and was
presented as bounding "the decoder". It now states the representation, reports
the probe values alongside accuracy (+0.067 at the 2 % injection where accuracy
is 0.903), and gives the headline decoder's own probe value, -0.068 at 0.901,
with all eight cohort-A arms between -0.066 and -0.082. Both limits are named:
the cross-representation comparison, and the probe's blindness to artefact
information constant within a class. It is written as evidence, not a bound.

Rebuilt: 0 errors, 0 undefined. Table 2 measured inside its column
(54.4-285.5, edge 289). The remaining full-width blocks are frontmatter and
the two starred floats, all legitimate.

Also added a per-subject progress print to `exp_calmnetx.py` so every run from
here on reports as it goes. Affects new runs only; the running arm had already
loaded the old module.

### 01:59 — Third cohort chosen and prediction registered first

**EEGMMIDB** (PhysioNet motor movement/imagery). Public, independent lab and
hardware, downloadable through MNE. Motor-execution runs as rest vs movement,
the nearest analogue of walk vs stop. Its protocol alternates the classes every
~4 s, sharply unlike cohort B's 309-window blocks.

The prediction was committed at **01:59:52 (`b7ca694`)**, before any model
touched the cohort. P1: enabling alignment costs less than 0.05. P2: slowing
to m = 0.01 gains less than 0.05, where on cohort B it gained 0.090. P2 is the
discriminating test. Thresholds and scope are fixed in
`PREREGISTRATION_cohort3.md`.

Afterwards, as a diagnostic that feeds no decision, the loader measured the
median block length on S001: **5.0 windows**, exactly the value derived from
the protocol description beforehand.

### 02:00 — E: drive is not writable

You authorised E: for the third cohort. Every folder on it returned access
denied, from both Git Bash and PowerShell, with and without the sandbox. This
is a Windows permission on the drive, not something fixable from here.
Everything went to C: instead.

### 02:03 — Made room on C:

Removed 3.85 GB of regenerable caches: 31 files for settings no experiment
tonight uses (2 s windows, z-scored, mu-only, beta-only, tangent features),
including the corrupt sub-04 file. Those experiments are finished and their
results are in `results/*.json`; each cache rebuilds itself on its next use.
Exact list: `paper/cache_cleanup_0203.txt`. All 14 caches tonight's runs need
were kept. Free space went from 1.4 GB to 5.0 GB.

### 02:05 — ICA artefact control implemented

`dataio._ica_clean`: average reference, 1-45 Hz fit copy, Picard extended
(approximating the extended Infomax ICLabel was trained on), ICLabel, and
removal of eye, muscle, heart, line-noise and channel-noise components. Brain
and "other" are kept. Every recording's decision is logged to
`results/ica_components.csv`.

Tested on one real recording (sub-01 ses-01 trial01, 59 s). ICLabel removed
**7 of 30 components: 5 muscle, 1 eye blink, 1 channel noise**, and **45 %**
of the 8-30 Hz variance remained. Part of the drop is the average reference.
Most of it is muscle. This recording is heavily muscle-contaminated even
within the decoding band, so the control will be informative: accuracy that
depended on muscle should fall once it is removed.

**Scope decision.** Cleaning both cache sets would need 4.41 GB and leave
~0.4 GB free on a drive that filled twice tonight. The control therefore runs on
the training task only (`CX_FULL=0`) for **both** its arms, uncleaned and
cleaned. They differ only in the cleaning step, and the pair needs 1.77 GB.

### 02:05 — First ICA precompute failed on memory, safely

With 3 workers, every subject failed with MemoryError, even on 12 MB
allocations, while 5.2 GB of physical RAM was free. The cause is Windows commit
charge: 34.6 GB committed against a 37.0 GB limit, because the pagefile cannot
grow on a full C:. Run 2 alone holds 11.6 GB. The rest is spread across
Chrome, VS Code and other applications, which I did not close, since they may
hold your unsaved work.

Nothing was damaged: run 2 is unaffected and no partial cache was written.
The fix is sequencing. ICA now starts only after run 2 exits.

### 02:08 — Overnight queue launched (`tools/overnight_queue.py`, pid 14268)

Written in Python, because a shell `sleep` loop launched from this session was
killed after a few seconds earlier tonight. Every wait is on a real condition
with a timeout, and a failed job is logged and skipped.

1. Wait for run 2 to exit.
2. ICA precompute, 2 workers, on CPU.
3. GPU: artefact control, uncleaned arm.
4. Wait for the full EEGMMIDB download.
5. GPU: third cohort, `gate` + `align+gate` at m = 0.2, then `align+gate` at
   m = 0.01.
6. Wait for ICA; GPU: artefact control, cleaned arm.
7. GPU: align-only arm on full data.

Also added two arms: `dn_gate` (no alignment, no context), required by the
pre-registration so that alignment is the only differing factor; and
`dn_align` (alignment alone), for the selective-head question.

### 02:10 — Evaluation committed before any result (`7200b65`, 02:10:10)

`src/eval_overnight.py` applies the pre-registered P1/P2 thresholds literally.
Wilcoxon tests and direction counts are reported but cannot change a verdict.
With the decision rule committed as code before results exist, it cannot be
adjusted afterwards.

### 02:12 — Presentation deck built (`presentation/build_update_deck.py`)

15 slides in the midterm deck's style: serif type, black on white, ruled
tables. Slide titles have no colons. The deck reads tonight's results from
`results/overnight_eval.json` when it is built, so rebuilding fills in the
pending slides.

The midterm deck said "the reported accuracy is largely the confound" and put
invariant decoding at ~0.65. The updated paper reports 0.901, so the audience
will ask whether that accuracy is neural. Slide 8 answers it, and the ICA
control is its centrepiece. Slide 2 explains the change directly: the
leakage measure used at the midterm scored R² = 0.863 on a decoder carrying no
movement information at all.

Every slide was rendered through PowerPoint and checked visually. Slide 12's
text crowded its table; slide 13's cards were oversized. Both fixed.

### 02:17 — Removed the contradictory abstention label from Figure 1

The architecture figure's selector box read `abstain / act  0.876 @ 90 %`,
presenting as a feature the figure the paper reports as a negative result
(0.876 is below the arm's own 0.901). It would have contradicted slide 13 in
front of the audience.

The committed `paper/fig_arch.drawio` is stale: `FIGURE1_README.md` records
that `results/fig_arch.pdf` came from an edited source that was never checked
in. Re-rendering would have replaced the new figure with the old layout, so
the figure was not regenerated. It is a vector PDF, and the label is a single
real text span (LiberationSans 8.6 pt), so only that span was redacted, with
no fill and line art preserved. `abstain / act` stays, since it describes the
component. Before/after renders confirmed nothing else moved and no white
patch appeared. Original kept as `results/fig_arch_original_with_0876.pdf`.

### 02:16 — Second queue so the GPU is not idle after ~04:40

Queue 1 should finish around 04:40, which would leave the GPU idle for four
hours. `tools/overnight_queue2.py` (pid 26912) starts when queue 1 logs
QUEUE DONE and runs, in priority order:

1. Cohort A, align+gate, seeds 0-2, per-subject rows. Removes "single seed"
   for the headline. The arm has no transformer, and the paper reports such
   arms bit-identical across repeats, so the seed-0 rerun should reproduce
   0.884 exactly. That makes it a free reproducibility check.
2. Cohort B, gate and align+gate, seeds 0-2.
3. Cohort A, gate, seeds 0-2.

The result is one design across all three cohorts: gate against align+gate,
alignment the only differing factor, with paired tests. This is the same
design the cohort C pre-registration uses, so A, B and C become directly
comparable. No new job starts after 07:00, so integration always has time.

Not pursued: published baselines on cohort B. `exp_published.py` has no cohort
switch, and adapting a second training pipeline overnight is where bugs get in.
It stays in TODO_NEXT.md.

The evaluation for queue 2 was added to `eval_overnight.py` before any of its
results existed, including the reproducibility comparison.

### 02:40 — Run 2 finished: the trace-normalised ablation is complete

`dn_nogate` (align + context, no gate) under the trace-normalised estimator:
**0.862** (ECE 0.065, cond R² −0.066), with per-subject rows. It took
12,497 s, 3.5 h. The transformer arms run about 6.6× slower than the
31-minute arms without it, which is worth knowing when planning seeds.

That fills the last missing cell, and every cohort-A component arm now exists
under the trace-normalised estimator:

| arm | original | trace-normalised |
|---|---|---|
| align + gate | 0.901 | 0.884 |
| align + ctx + gate | 0.878 | 0.862 |
| align + ctx | 0.862 | 0.862 |
| ctx + gate | 0.827 | 0.834 |
| stem only | 0.827 | 0.827 (inert) |

"Pick one estimator" (TODO 1.1) is now unblocked. The ordering is the same
under both estimators, and alignment still adds to every arm.

Per subject: 0.934, 0.904, 0.984, 0.942, 0.887, 0.813, **0.568**. Sub-07 is
the weak participant in this arm too.

One detail for the abstention count: in this no-gate arm acc@90 (0.860) sits
0.002 below acc. All four arms where abstention helped had no trained gate,
but not every no-gate arm is helped. The paper's statement concerns which arms
helped, so it still holds.

The queue started at once: ICA precompute (2 workers) and the uncleaned
artefact-control arm.

### 02:47 — Artefact control, uncleaned arm done

Align + gate, cohort A, training task only, no cleaning: **mean 0.836** over
seven participants (0.969, 0.898, 0.955, 0.835, 0.839, 0.673, 0.680). This is
the paired reference for the cleaned arm. It is below the headline 0.884
because it fits on ~803 windows per participant rather than ~5031. Both
control arms share that restriction, so the comparison between them stays
fair. Runtime was 6 minutes, and the new per-subject progress print worked.

ICLabel decisions across the first 46 recordings: mean **6.5 of 30** components
removed (median 6, range 1-16; 21.7 %). By class: **muscle 45 %**, eye blink
40 %, channel noise 10 %, heart 4 %. Muscle is the largest class removed.

### 02:49 — tau_drift scheduled behind ICA

`tools/run_tau_drift_after_ica.py` (pid 5660) waits for the ICA precompute to
report DONE and then runs `exp_tau_drift.py` on both cohorts. It is CPU-only
and loads one subject at a time, so it runs alongside the GPU queue without
competing with the ICA workers for cores or commit charge.

### 02:52 — tau_drift, as designed, measures the wrong thing on cohort A

The ICA precompute finished cleanly: 7/7 subjects in 518 s, ~251 MB each,
1.77 GB total as budgeted. The waiter started tau_drift only after DONE.

tau_drift's first values, sub-01 18.2 and sub-02 17.9 windows, match cohort
A's class-block length of 18. That is not a coincidence. The script computes a
covariance per 32-window block regardless of class, and cohort A switches class
about every 18 windows, so almost every block spans a class boundary. The
block-to-block distance therefore saturates at the class-switching timescale.
**It re-measures tau_blk, not drift.** The fitted tau also falls below the
32-window lag resolution, so it is extrapolated.

**Decision: tau_drift stays out of the paper tonight.** The correct design
uses class-conditional covariances, but cohort A's walk class has ~75 windows
per session (about two blocks), too few for a lag curve. A second option is
the session-to-session drift across the nine sessions, in days. Either needs
careful thought and should not be rushed at 3 am. A confounded number in the
paper would be worse than leaving the upper bound inferred, as it stands now,
and the limitation already says so. The run continues on CPU for the record;
cohort B's 309-window blocks make most 32-window blocks single-class, so its
value may be less confounded. The corrected design goes to TODO_NEXT.md.

### 02:51 — Cohort B's data is not on this machine; queue 2 corrected

tau_drift loaded **0** cohort-B subjects. The cause: neither
`data/mobi_treadmill` (raw) nor `data/cache_mobi` exists on this machine.
`data/` holds only cohort A, its cache, and the new EEGMMIDB download. The
cohort-B results in the repository were produced on the other machine, in the
EsmeAbha commits. The cleanup did not remove it: it only touched named
patterns inside `data/cache`, and these directories were never present.

**This matters for reproducibility.** The paper's cohort-B numbers cannot be
regenerated here. Copy `mobi_treadmill` across from the other machine before
any cohort-B rerun.

Queue 2's planned cohort-B job would have failed. While it was still waiting,
with no GPU job in flight, queue 2 was stopped (pid 26912) and relaunched
(pid 26484) with the job replaced by **a cohort-C seed replication, seeds
1-2**, of both pre-registered comparisons. It is fast and uses local data. The
registered verdict remains seed 0, as registered; the replication is reported
separately and cannot change it. Its evaluation was added to
`eval_overnight.py` before any of its results existed.

New queue-2 order: C replication (m 0.2, then 0.01), then A align+gate seeds
0-2, then A gate seeds 0-2. The 07:00 cutoff still applies.

The multi-seed paired design now covers cohorts A and C. B keeps its existing
single-seed results.

tau_drift completed at 02:49:38. Cohort A median tau = 17.9 windows (range
15.5-29.1), confirming the class-switching confound logged above. Cohort B:
no data. The result is kept in `results/tau_drift.json` for the record and does
not enter the paper.

### 03:13 — Pre-registered P1: CONFIRMED

All 20 EEGMMIDB subjects entered (none skipped, as registered), ~420 fit
windows each.

| arm (cohort C, seed 0) | balanced acc. |
|---|---|
| gate, no alignment | 0.783 |
| align + gate, m = 0.2 | 0.785 |

**Mean difference +0.002, above the registered threshold of −0.05, so P1 is
CONFIRMED.** Eleven of 20 participants were higher and 9 lower (Wilcoxon
p = 0.68). The largest single loss was S010 at −0.087, and no participant
approached cohort B's −0.211.

Enabling alignment costs nothing on cohort C, where the same comparison cost
0.211 on cohort B. The condition predicted this from the protocol alone:
blocks of ~5 windows, far inside the 160-window memory, leave nothing for the
estimator to track.

Read with care: alignment does not *help* here either. That fits the band,
since a single-session recording has little drift to correct, but the result
shows the absence of harm and should not be reported as a benefit. The decoder
reaches ~0.78 balanced accuracy on a third independent dataset, against 0.50
chance.

P2 (slowing to m = 0.01 gains less than 0.05) is running.

