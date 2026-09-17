# Handoff — machine `TH-ODIN-WS36`

Third machine. Has **both** cohorts' data (`data/ds007788`, `data/mobi_treadmill`).
Not the RTX 5080, so its jobs do not contend with the 5080 queue.

Owns: runs and analysis only. **Does not touch `paper/`** — the manuscript is
owned by the other session.

All rows below ran on code commit **`f91b557`**.

---

## Result files on this machine

| file | cohort | arms | rate | seeds | participants/seed | commit | status | known problems |
|---|---|---|---|---|---|---|---|---|
| `b_floor_m0.10.json` | B | `dn_noctx`, `dn_noalign` | m=0.10 (μ=320) | 0,1,2 | 8 | `f91b557` | **final** | none |
| `b_floor_m0.02.json` | B | `dn_noctx`, `dn_noalign` | m=0.02 (μ=1600) | 0,1 of 0,1,2 | 8 | `f91b557` | **partial** | snapshot mid-run; `dn_noctx` not started |
| `tau_drift_session.json` | A, B | n/a (no model) | n/a | n/a | 7 (A), 8 (B) | `1ebbee7` | **INVALID** | see below |

### Not on this machine

`b_gate_aligngate_3seed.json`, `b_aligngate_m001_3seed.json`, `bd_cohort_b.json`,
`a_ablation_s12.json` — these are the 5080 queue's outputs and have never
existed here.

`drift_fixed.json` — the no-op control rerun has **not been run yet**; it is
third in this machine's queue. The trace-normalisation fix it depends on is
also not yet committed from here (it was lost in a pull yesterday and needs
re-applying to `src/exp_drift.py`).

---

## `b_floor_m0.10.json` — the point of the run

Cohort B has τ_blk = 309 windows. The existing sweep measures μ = 160, 640 and
3200, so it brackets the threshold without landing on it. m=0.10 gives μ = 320,
essentially **on** τ_blk.

| μ (windows) | m | `dn_noctx` (align+gate) | `dn_noalign` | alignment contributes |
|---|---|---|---|---|
| 160 | 0.20 | 0.581 | 0.792 | −0.211 |
| **320** | **0.10** | **0.638 ± 0.001** | **0.767 ± 0.020** | **−0.129** |
| 640 | 0.05 | 0.671 | 0.782 | −0.111 |
| 3200 | 0.01 | 0.696 | 0.783 | −0.087 |

Only the μ=320 row is 3-seed; the others are single seed from the earlier sweep
and are **not** directly comparable to it.

**What this supports.** Most of the recovery happens crossing τ_blk: +0.082 of
the total +0.124 occurs between μ=160 and μ=320. That is where the floor
predicts it.

**What this does not support.** Alignment is still −0.129 at μ=320, which is
*above* the class-block length, and never becomes positive at any rate tested.
Crossing the floor recovers most of the loss but does not make the layer
beneficial on this cohort. Anyone writing "the layer works once the memory
exceeds the class block" should not cite this file for it.

**Seed spread is strongly asymmetric** and worth knowing before quoting any
single-seed number on cohort B:

- `dn_noctx`: 0.6374, 0.6401, 0.6373 → spread **0.003**
- `dn_noalign`: 0.7896, 0.7526, 0.7588 → spread **0.037**

The alignment-free arm is twelve times noisier than the aligned one. The
existing single-seed sweep values for `dn_noalign` (0.792, 0.782, 0.783) sit
inside that spread and look far more stable than they are.

---

## `tau_drift_session.json` — INVALID, do not use

Measures Riemannian distance between session covariances against session gap
and fits a saturating exponential. **The fitted timescales are unusable.**

- All 7 cohort-A subjects fit τ < 1 session, but 1 session is the **smallest
  gap sampled**, so the value is extrapolated from the assumed curve shape.
- d(gap) is flat from gap 1 to gap 8 for five subjects and **decreases** for
  sub-02 (13.41 → 8.75) and sub-07 (11.79 → 8.50), which is the opposite of
  drift accumulating.
- Cohort B cannot be fitted at all: 3 trials give 2 distinct gaps.

This is the same error as the earlier `tau_drift.json` (17.9 windows,
within-session): fitting a timescale where the data carry no gradient. **Both
numbers, 17.9 and 129 windows, are unusable.**

The data support one statement only: session-to-session drift is complete
within one session. That does not constrain the estimator memory μ, which is
far shorter than a session. **The admissible-band ceiling is unmeasurable on
both cohorts. Claim the floor only.**

The file carries an `_INVALID` key stating this, so it cannot be picked up
downstream by accident.

---

## Scripts added or changed here

| path | purpose |
|---|---|
| `tools/floor_locate.sh` | the cohort B interior-rate sweep above |
| `src/exp_tau_drift_session.py` | across-session τ_drift (produces the invalid file; kept so the failure is reproducible) |

---

## Still queued on this machine

1. **Cohort A slow rate**, `dn_noctx`, m=0.01, 3 seeds. Currently single-seed
   only (0.799). This is the reviewer's point that the slow-rate cost was
   measured on `dn_full`, not on the proposed configuration.
2. **No-op control rerun** without trace normalisation. Cheap, no GPU. Expected
   to make the control exactly null and to lower the reported drift reduction
   by roughly 5 points (58.2% → 53.0% on cohort A sub-01 in a spot check).
