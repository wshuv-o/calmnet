# Literature positioning — step 1 of the Next list

Answers the question section 0 of `SESSION_NOTES.md` raised: of the residue left
after the 0.776 headline turned out to be standard practice, what is actually
unclaimed?

**Evidence level.** Abstracts, publisher landing pages and search snippets, plus
one full-text fetch that failed (arXiv 2502.09203 PDF would not extract; PMC
blocked with a CAPTCHA). Nothing below rests on a full read. Every verdict is
therefore *provisional* and marked with what it would take to firm it up. This
is a positioning pass, not a systematic review.

---

## 1. The winning pipeline — confirmed standard practice

Section 0 was right, and the case is stronger than it stated.

- **Junqueira, Aristimunha, Chevallier & de Camargo (2024)**, *A systematic
  evaluation of Euclidean alignment with deep learning for EEG decoding*,
  J. Neural Eng. 21(3). EA + DL, shared multi-subject models: **+4.33%** on the
  target subject and **>70%** faster convergence. This is precisely the
  "EA helps, cheaply" result, already systematised.
  <https://iopscience.iop.org/article/10.1088/1741-2552/ad4f18>
- **Wan, Wu et al. (2025)**, *Revisiting Euclidean alignment for transfer
  learning in EEG-based BCIs*, J. Neural Eng. — position is that after EA,
  auxiliary-subject data can be pooled directly with a little calibration data
  and performance is already good "without any other sophisticated TL
  techniques." <https://iopscience.iop.org/article/10.1088/1741-2552/addd49>
  (preprint: <https://arxiv.org/abs/2502.09203>)
- **Junqueira et al. (2024)**, *Combining Euclidean alignment and data
  augmentation for BCI decoding*. <https://arxiv.org/pdf/2405.14994>

Consensus in this literature is that EA **should be a standard preprocessing
step** for cross-subject models. Publishing 0.776 as a finding is not viable.
Confirmed; no further checking needed.

---

## 2. Residue #1 — closed-form alignment beats *learned* adversarial invariance

**Verdict: partially open, but narrower than section 0 hoped, and the framing
has to change.**

What is already claimed:

- EA-versus-deep-DA is a well-populated field. Adversarial MI decoders with
  gradient reversal are routine (e.g. *Conditional Adversarial Domain
  Adaptation Neural Network for MI EEG Decoding*, Entropy 2020,
  <https://www.mdpi.com/1099-4300/22/1/96>), and adversarial invariance against
  a nuisance variable in EEG is itself established (*Learning Invariant
  Representations from EEG via Adversarial Inference*, IEEE Access 2020,
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7971154/> — could not be fetched,
  **needs a read**).
- The general claim "simple alignment is competitive with sophisticated transfer
  learning" is the explicit message of the Revisiting-EA paper. So "closed-form
  beats learned" is *already the prevailing view* for subject/session shift. As
  a headline it is a replication.

What I could not find, after four targeted searches:

- A **direct head-to-head** of EA against GRL / HSIC / decorrelation on the
  **same** task, ranked by both accuracy and a nuisance-recoverability
  criterion.
- Any such comparison where the nuisance is **movement measured by a
  synchronised IMU** rather than subject or session identity.

So the defensible framing is not "we discovered closed-form beats learned." It
is: *the alignment-versus-adversarial comparison has been made for subject and
session shift, but not for a physically measured nuisance that is correlated
with the label itself* — which is the hard case, because here the confound
carries the signal rather than merely displacing it. That is a narrower and
more honest claim, and it is the one this project's data can support.

**To firm up:** read the IEEE Access adversarial-inference paper and the
Revisiting-EA full text; check whether either runs EA as a baseline against its
own adversarial method.

---

## 3. Residue #2 — the invariance probe as a diagnostic

**Verdict: the mechanism is not new; the application might be.**

- Probing a representation for a nuisance variable is standard practice under
  several names — attribute recovery, leakage auditing, shortcut/Clever-Hans
  detection. Recent EEG work audits foundation-model embeddings for exactly this
  (*Pretrained, Frozen, Still Leaking*, <https://arxiv.org/abs/2606.09189>),
  reporting subject-disjoint recovery bounds of 0.157–0.374.
- Shortcut learning and Clever-Hans effects in biosignal classifiers are a
  developed literature with named mitigations (RRR loss, ClArC).
- Test-set leakage through preprocessing statistics is already a documented
  BCI-methods failure.

What I did not find: anyone using nuisance-recoverability as a **model-selection
criterion** — an admissible set defined by R² ≤ 0, with the accuracy ranking
declared invalid inside it. The 131-architecture sweep result
(corr(accuracy, leakage) = **+0.603**; 0 of 131 admissible-and-better) is a
population-level statement about an architecture search, not a single-model
audit, and I found no equivalent.

**Caveat that matters:** that +0.603 was computed with the **broken probe**. It
is currently unscored. Until it is recomputed with `invariance_r2_cv`, this
residue has no evidence behind it. Step 3 of the Next list is therefore a
precondition for this claim, not a follow-up to it.

---

## 4. The confound itself is old news — and contested

The finding that IMU-only reaches 0.870 while every EEG decoder sits lower lands
inside a decade-old, still-unsettled debate:

- **Castermans et al. (2014)**, *About the cortical origin of the low-delta and
  high-gamma rhythms observed in EEG signals during treadmill walking*, Neurosci.
  Lett. Motion artifact contaminates up to ~15 Hz with many harmonics of the
  stepping frequency; puts the cortical origin of those bands in doubt.
  <https://pubmed.ncbi.nlm.nih.gov/24412128/>
- **Nathan & Contreras-Vidal (2015)**, *Negligible motion artifacts in scalp EEG
  during treadmill walking*, Front. Hum. Neurosci. — the direct rebuttal.
  <https://www.frontiersin.org/articles/10.3389/fnhum.2015.00708/full>
- **Kline et al. (2015)**, ICA of gait-related movement artifact.
- IMU-as-artifact-reference is active: *IMU-Enhanced EEG Motion Artifact Removal
  with Fine-Tuned Large Brain Models* (Sept 2025, LaBraM + correlation attention,
  9.2M params, benchmarked against ASR-ICA) <https://arxiv.org/abs/2509.01073>.
  This overlaps directly with the in-network motion canceller — which failed
  here — so that negative result now has a positive-claiming contemporary to be
  positioned against.

Implication: "walking EEG is movement-contaminated" is not publishable. What is
potentially publishable is the **quantitative instrument** — an IMU-referenced
recoverability score applied to a decoder's own representation, used to
adjudicate a debate that has so far been argued with spectra and ICA components
rather than with decoder-level evidence.

---

## 5. What this changes about the plan

1. Residue #1 survives only in its narrow form (physically measured,
   label-correlated nuisance). Say that, not the broad version.
2. Residue #2 is **blocked on re-scoring**, not on literature. Next-list step 3
   is now a precondition for step 1's conclusion, so the ordering in
   `SESSION_NOTES.md` needs one swap.
3. Step 2 (real baselines) is unchanged and still mandatory: pyriemann MDM,
   MOABB pipelines, and — new — at least one published adversarial DA method
   (e.g. the Entropy 2020 CDAN-style decoder) so the head-to-head is against
   something someone published, not only against in-house adversaries.
4. Add Castermans-vs-Nathan as the framing of the confound section. The
   contribution is the measurement, not the observation.

## Not checked

- Full texts of every paper above.
- The gait/exoskeleton BCI literature for prior IMU-only control analyses.
- Whether ds007788 and the MoBI cohort share a lab (still open from the
  previous session).
- MOABB leaderboards for the specific numbers claimed in section 1.
