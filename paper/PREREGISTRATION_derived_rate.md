# Pre-registration: deriving the adaptation rate from the protocol

**Written before any derived-rate run. Committed before the first result.**

Author: 5080 machine. Protocol agreed with the 2060 machine on 2026-09-18.

---

## 1. Why this exists

The manuscript states Condition 1 as a band on the estimator memory and reports
`tau_blk` per cohort, but the rate actually used was chosen by trying `m=0.2` and
`m=0.01` and keeping whichever worked. That is per-cohort tuning, and a reviewer
will say so. The contribution becomes a rule only if the rate is **derived from
the label sequence before training** and then holds without accuracy being
consulted to pick it.

This document fixes the derivation and the single free constant in advance, so
that the result cannot be retrofitted.

---

## 2. The derivation, fixed now

Let `N = 32` be the number of windows per estimator update (the batch size, an
existing constant of the pipeline, not chosen here). The running covariance has
an exponential memory of

```
mu = N / m        windows
```

Condition 1 requires `mu > tau_blk`: the memory must outlast a single-class
block, or the estimate converges on the streaming class. Taking the **smallest
admissible memory** maximises the rate at which the estimate can follow genuine
drift while still satisfying the condition, so with a safety margin `k`:

```
mu* = k * tau_blk
m*  = N / (k * tau_blk)
```

**The margin is fixed at `k = 2`** — the memory must span at least two class
blocks. Chosen now, before any run, and applied unchanged to every cohort. No
per-cohort adjustment is permitted under this protocol.

`tau_blk` is the **median contiguous single-class block length in windows,
computed from the fitting-split labels only**. Fitting-split only, because at
deployment that is all that exists; the held-out sessions are not available when
the rate must be chosen.

---

## 3. The derived rates

Computed by `tools/tau_blk_rule.py` from labels alone. No model was trained or
read and no accuracy was consulted.

| Cohort | `tau_blk` (windows) | `mu* = 2*tau_blk` | **derived `m*`** | rate already run |
|---|---|---|---|---|
| A (exoskeleton, n=7) | 18 | 36 | **0.889** | 0.2 |
| B (treadmill, n=8) | 214 | 428 | **0.0748** | 0.2 and 0.01 |
| C (motor execution, n=20) | to be computed on the 2060 | | | 0.2 |

`tau_blk` is identical for every participant within a cohort (18 for all 7 of
cohort A, 214 for all 8 of cohort B), which is expected: it is a property of the
recording protocol, not of the person.

**Note on cohort B.** The manuscript's Table 2 reports `tau_blk = 309` for this
cohort over 7 blocks. Recomputing over all 8 participants gives **214** over 66
blocks, while the longest block matches the published 2211 exactly. The
published row appears to be one participant while its "longest" column is
pooled. This is flagged for the 2060 to resolve. The rule below is evaluated at
214; at 309 the derived `m*` would be 0.0518 and every prediction in Section 4
is unchanged in sign.

---

## 4. Predictions, registered before the runs

**P1.** On every cohort, the derived `m*` is at or near the best of the rates
tried, where "near" means within the seed spread of that cohort's arm.

**P2.** Cohort B at `m* = 0.0748` recovers the collapse seen at `m = 0.2`:
its accuracy will be **at least that of `m = 0.01` (0.7177)**, and not near the
0.500 chance line.

**P3.** Cohort A at `m* = 0.889` does **not** collapse, because `mu* = 36`
still exceeds `tau_blk = 18`. It may lose accuracy relative to `m = 0.2` through
a noisier estimate, but it will stay above the stem-only control minus one seed
spread.

**P4.** The ordering of admissibility already holds on the rates run so far and
must continue to hold:

| cohort | m | mu | tau_blk | admissible | measured |
|---|---|---|---|---|---|
| A | 0.2 | 160 | 18 | yes | +0.045 |
| B | 0.2 | 160 | 214 | **no** | −0.175 |
| B | 0.01 | 3200 | 214 | yes | −0.026 |

**What would falsify the rule.** Any of: the derived `m*` is worse than a rate
the rule declares inadmissible; cohort B at `m*` stays near chance; cohort A at
`m*` falls below its stem-only control by more than one seed spread. Any of
these and the rule is reported as failed, not adjusted.

---

## 5. Division of work

- **5080:** cohort B at `m* = 0.0748`, 3 seeds, arm `dn_stem_tan`, reference
  `running`, shrink 0.1. Control `dn_stem` at 0.7436 already exists.
- **2060:** cohorts A and C at their derived `m*`, same arm and settings.

Everything else is held at the values already published. Nothing in this
protocol may be revised after a result is seen.
