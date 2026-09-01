# Session findings — representation beats architecture

> **Read section 0 first.** The result below is a solid *internal* finding, but
> the winning method is standard practice in the BCI literature, not a
> contribution. This document originally framed it as a discovery; that framing
> was wrong and is corrected in section 0.

Headline: a **1,831-parameter classical pipeline outperforms 131 deep
architectures and 18 published deep models** on this task, on two cohorts, and
is the only method that is genuinely movement-invariant.

```
covariance -> Euclidean Alignment -> log-Euclidean tangent -> logistic regression
```

| representation | ds007788 acc | R² | MoBI acc | R² |
|---|---|---|---|---|
| **raw \| tangent_EA** | **0.776 ± 0.010** | **−0.265** | **0.675** | **−0.961** |
| raw \| tangent+PLV | 0.780 ± 0.035 | −0.256 | 0.689 | −0.950 |
| car \| tangent_EA | 0.774 ± 0.012 | −0.267 | 0.669 | −0.941 |
| raw \| FBCSP | 0.764 ± 0.029 | **+0.187** | 0.689 | −0.134 |
| raw \| PLV | 0.710 ± 0.035 | +0.020 | 0.646 | −0.247 |
| raw \| bandpower | 0.634 ± 0.010 | +0.040 | 0.574 | −0.066 |
| deep composed architecture | 0.695 ± 0.024 | — | 0.588 | — |

R² is intent→motion recoverability. **Negative means movement is not
recoverable**, i.e. genuinely invariant. R² > 0 means it is not.

Why it works: movement artefact lives in the **second-order structure** of the
signal and shifts between sessions. Euclidean Alignment whitens each session by
its own mean covariance — label-free, computed from that session's own data — so
the artefact is removed before any classifier sees it. Every deep method here
instead tried to *learn* invariance against it (adversary, HSIC, decorrelation,
in-network cancellation). All were fighting in the loss for something a
whitening transform does in closed form.

---

## 0. Novelty assessment — the winning method is not new

The pipeline that won is **established practice**, not a discovery:

- Riemannian tangent-space classification of EEG covariance: Barachant et al.,
  2012.
- Euclidean / Riemannian Alignment for cross-session and cross-subject transfer:
  He & Wu, and tangent-space alignment validated across **18 BCI databases,
  349 subjects** (PMC9755175).
- "Riemannian methods match or beat deep learning on small-sample BCI" is
  reproduced routinely in MOABB benchmarks.

So the honest description of this session's headline is: **the standard pipeline
for this problem was not tried until late, and when it was, it beat everything
built before it.** That is a process failure, not a result. Any write-up that
presents 0.776 as a contribution will be desk-rejected by anyone who knows the
Riemannian BCI literature.

### What might still be unclaimed

1. **Closed-form second-order alignment beats learned adversarial invariance.**
   Adversarial gradient reversal, HSIC, cross-covariance decorrelation and
   in-network motion cancellation were all run against Euclidean Alignment on
   the same task. EA won on accuracy *and* on invariance. A direct head-to-head
   of this kind, on a movement-confounded paradigm, is the most promising
   residue here.
2. **The invariance probe as a diagnostic** — measuring how recoverable the
   nuisance variable is from the decoder's own representation, and showing that
   accuracy gains track it at r = +0.60 across 131 architectures.

### Why neither is publishable yet

- **The real baselines were never run.** No pyriemann MDM, no MOABB pipelines,
  no published domain-adaptation comparators. Without them there is no way to
  separate a contribution from a textbook result — which is exactly the mistake
  made above.
- **Every R² predating the probe fix is unscored**, including the +0.603
  correlation the whole argument rests on.
- **The literature has not been checked** for movement-artefact-invariant BCI
  decoding, or for existing EA-versus-adversarial comparisons. The residue above
  may already be claimed.
- N = 7 + 8, and the two cohorts may share a lab.

### Next action is not another experiment

A proper literature search on (a) movement-artefact-invariant BCI decoding and
(b) alignment versus adversarial domain-invariance comparisons. If the residue
survives that, most of the supporting experiments are already built and only
need re-scoring with the corrected probe.

---

## 1. A metric error invalidated earlier conclusions

The original probe fitted a ridge on the **training** sessions and scored it on
the **test** sessions. That conflates invariance with distribution shift: a
representation whose features merely move between sessions scores strongly
negative while still encoding movement perfectly well inside any session.

| representation | cross-split probe (wrong) | within-test CV probe (honest) |
|---|---|---|
| FBCSP | −0.501 "most invariant" | **+0.187** — clearly leaky |
| tangent_EA | −0.089 | **−0.258** — more invariant |
| bandpower | +0.007 | +0.034 |

Corrected in `features.invariance_r2_cv` (session-grouped cross-validation
inside the evaluation set). **Every R² produced before this fix is suspect**,
including the 131-architecture correlation, the module-ablation verdicts and the
backbone selection. Those need re-scoring; the models are already trained, so it
is a re-measurement, not a re-training.

## 2. The confound is real, and larger than previously reported

IMU-only (no EEG at all) reaches **0.870**, above EEGNet (0.819) and
EEG-Conformer (0.826). Previously reported as 0.839 — understated because
`dataio` silently substitutes zeros when a session's motion file is missing, and
those sessions then score exactly 0.500 in the movement baseline. Six sessions
across five subjects are affected (`calmnet_msa.imu_valid_mask`).

## 3. Architecture search does not move the honest number

131 variants (5 backbone families, 7 readouts, 15 augmentations, 4 independence
penalties, 4 optimisers, 5 frequency bands, 7k–391k params):

- **0 of 131** beat baseline with R² ≤ 0
- corr(accuracy, leakage) = **+0.603**
- capacity buys leakage faster than accuracy: corr(log params, R²) = +0.379 vs
  corr(log params, accuracy) = +0.262
- a 310k-param DeepConvNet (0.681) loses to a 7k band-power net (0.688)

Published backbones tell the same story: **EEGConformer scores 0.827 with
R² +0.456** — the highest accuracy and the worst leakage of 18 models.

## 4. Frequency band traces the leakage/accuracy trade-off monotonically

| band | acc | R² |
|---|---|---|
| mu 8–13 | 0.612 | −0.013 |
| 8–30 (default) | 0.688 | +0.032 |
| beta 13–30 | 0.709 | +0.101 |
| broad 4–40 | 0.708 | +0.109 |
| full 1–45 | 0.718 | +0.127 |

Every Hz added outside mu buys accuracy by buying movement. Survived seed
replication. This was the signal that the **representation** axis mattered more
than the architecture axis, and it should have been followed sooner.

## 5. Seed variance kills two published claims

Mean seed sd = **0.028** over 23 replicated variants.

- **Abstention gain (+0.025) is inside the noise.** "Executed accuracy
  0.694 → 0.719 @ 80% coverage" cannot be claimed from a single seed.
- **Single-seed leaderboards are selection on noise.** `bb_tcn` topped the
  131-variant sweep at 0.782; replicated it is **0.712 ± 0.057**. Mean shrinkage
  across replicated variants: −0.016.

## 6. Negative results worth keeping

- **Multi-subject pooling fails.** Subject identity stayed 0.76–0.80 recoverable
  (chance 0.143) across every configuration; the shared encoder never found a
  subject-invariant representation. 7 subjects is far below what adversarial
  subject-invariance needs. `calmnet_msa.py`, `exp_msa.py`.
- **In-network motion cancellation fails.** Removed 96.2% of a *synthetic*
  artefact, but on real data the network used the motion reference as an input
  feature and decoded the label from it: accuracy 0.806 → 0.908, leakage
  +0.294 → +0.582. Worst module in the ablation, dropped in both stages.
- **The composed deep architecture does not transfer.** Best configuration on
  ds007788 (0.622), worst on MoBI (0.588). Module selection fitted to one cohort.
- **ASR was untestable as run.** `dataio` z-scores every window to unit variance,
  and ASR detects artefact by variance exceeding a calibration threshold, so it
  was a guaranteed no-op. All 10 ASR cells are identical to their non-ASR
  counterparts. Testing it properly requires applying it to continuous data
  before epoching.

## 7. Paper/code discrepancies found and fixed

- **Adaptive conformal update sign was inverted** in the paper (both §LSC and
  Algorithm 1) relative to `calibrate.py`. As printed, coverage collapses to
  0.009 against a 0.90 target. Fixed.
- **Invariance claim overstated.** "Movement is no longer linearly or
  nonlinearly recoverable" was based on the MLP probe (−0.31); the linear probe
  on the same models gives −0.02, positive for 2 of 7 subjects. A more expressive
  probe returning a *more* negative R² indicates probe overfitting, not stronger
  invariance. Rewritten to report both.
- **CORAL** is claimed in §LSC but absent from the pipeline producing every
  band-power number. Qualified.
- Still outstanding: the selective head and `L_cal` in the training objective do
  not exist in the code (`calmnet_v2.py` now implements a real one), and the
  three-term SAS gate has never been run with more than one term.

---

## New code

| file | purpose |
|---|---|
| `features.py` | representation library: CAR, Laplacian, ASR, CSP/FBCSP, covariance→EA→tangent, PLV, and the corrected `invariance_r2_cv` probe |
| `exp_features.py` | 5 preprocessings × 5 feature sets, fixed logistic regression |
| `exp_features_validate.py` | split robustness + cross-cohort transfer, honest probe |
| `dataio_mobi.py` | second cohort loader (Luu et al. treadmill BCI, 8 subjects, goniometers) |
| `motion_ts.py` | per-window motion waveforms aligned to EEG windows |
| `calmnet_arch.py` | motion-referenced canceller + spectral leakage gate (canceller since dropped) |
| `exp_ablate.py` | module ablation: add-one, leave-one-out, stack |
| `arch_zoo.py`, `arch_zoo2.py`, `exp_sweep100.py` | 131-variant architecture search |
| `braindecode_zoo.py`, `select_backbone.py` | 18 published backbones through the same harness |
| `calmnet_msa.py`, `exp_msa.py` | multi-subject pooling (negative result) |
| `calmnet_v2.py` | selective head + Mondrian conformal + wrong-walk bound |
| `tools/dashboard.py` | live experiment dashboard |

## Next

1. **Literature positioning first** (see section 0). Establish what, if
   anything, in section 0's residue is unclaimed before running more compute.
2. Run the actual baselines: pyriemann MDM, MOABB standard pipelines, published
   domain-adaptation comparators. Without these there is no contribution claim.
3. Re-score the ablation, the 131-sweep and the backbone selection with
   `invariance_r2_cv`. Prior verdicts used the broken probe.
4. Rebuild the paper around whatever survives 1-3 -- NOT around the 0.776
   number, which is a replication of standard practice.
3. Consolidate the sweep files (`exp_sweep.py` is superseded; `arch_zoo`/
   `arch_zoo2` should merge; `exp_sweep_resume.py` should be a flag).
4. Check whether ds007788 and the MoBI cohort share a lab — if so the external
   validation is weaker than it looks and should be described as a different
   paradigm and sensor rather than an independent replication.
