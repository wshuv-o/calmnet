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

