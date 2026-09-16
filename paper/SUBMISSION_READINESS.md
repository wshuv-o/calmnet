# Submission readiness -- AUTHOR NOTES, NOT PART OF THE MANUSCRIPT

Kept out of `driftnet_draft.md` so it cannot be submitted by accident.

An honest assessment of what this manuscript can currently sustain, and what it
would take to reach a first-quartile venue (JNE, IEEE TNSRE, NeuroImage).

**Blocking gaps**

| gap | why it blocks | work required |
|---|---|---|
| Ablations are seed 0 only | split seed moves accuracy +-0.01-0.02, the same order as several reported effects. The headline ablation must be multi-seed or a reviewer will discount it. | 3 seeds x 5 arms x 2 cohorts, ~2 days compute |
| The central finding is demonstrated only on our own code | sections 7.3, 7.5 and 8 show *we* made these errors. To claim the field does, published work using unconditioned nuisance probes or fast test-time adaptation must be identified and cited. Searches so far have NOT established this. | 1-2 days literature work |
| The design rule is post-hoc | section 7.4 explains cohort B after seeing it fail. **Partly addressed:** we registered a prediction for cohort A before running it and it was falsified, which produced the two-sided rule in 7.5. That is a real out-of-sample test, but a failed one -- the *corrected* rule has still never predicted anything in advance. | ~1 week: obtain a public gait/MI dataset with different block structure, state the prediction from block structure alone, run once |
| ~~The -0.060 residual is unexplained~~ | **CLOSED.** Section 7.5: cohort B's block structure leaves no rate that both avoids class-tracking and tracks drift, so a correctly configured layer has nothing left to contribute. | done |

**What the paper cannot be.** An architecture paper. DriftNet fails external
validation and its novel component reverses sign; that is documented here and in
the repository and cannot be presented otherwise.

**What it can be.** A methods-and-cautionary paper, with the architecture as the
worked example. The four demonstrations in section 8 share one structure and were
produced by acting on the assumption they refute, which is stronger than
observing it in someone else's system.

**Realistic positioning today:** workshop paper, or second-quartile journal.
**After the seeds and the literature grounding:** plausible at JNE or TNSRE.
**With the third-cohort prediction confirmed:** the design rule becomes a tested
hypothesis rather than an explanation, which is the version worth submitting.

