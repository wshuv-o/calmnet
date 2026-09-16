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

