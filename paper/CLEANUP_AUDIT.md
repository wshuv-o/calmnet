# Cleanup audit: remove the development history, keep the final architecture

**The only architecture in this paper:** EEG window → multi-scale power stem +
tangent-space covariance branch → fusion → linear classifier.

CTX (cross-epoch transformer), adaptive Align, and the Gate / selective head are
components of earlier architectures. They are not components of the proposed
model and, after this cleanup, a reader should not need to know they existed.

## The decisive finding

**The estimator-memory / block-length condition does not need the Align
history.** Every experiment that supports it is `dn_stem_tan` against
`dn_stem` — the branch against the stem, with no alignment arm present:

| evidence | arms | alignment involved? |
|---|---|---|
| `tab:batchsweep` — μ = 8, 40, 160 on A; 160, 1280, 3200 on B | `dn_stem_tan` vs `dn_stem` | no |
| `tab:five` / `fig:cohorts` — five cohorts | `dn_stem_tan` vs `dn_stem` | no |
| `tab:tanref` — running / frozen / none reference | `dn_stem_tan` | no |

So under rule 6 the answer is: **the claim can be established directly from the
tangent-branch experiments, and the Align history is removed completely.**

---

## Audit

| Location | Old component | Action | Reason |
|---|---|---|---|
| **Title, Abstract, Highlights** | — | KEEP | Already state the branch only. One clause names the alignment layer indirectly; reword. |
| **Keywords** | selective prediction | REWRITE | Keyword exists only because of the dropped selective head. Replace with Riemannian / tangent space. |
| **Introduction, framing** | — | KEEP | Already reframed on what a log-power stem discards. |
| **Introduction, contributions** | — | KEEP | Six items already about the branch. |
| **Related work → Alignment and transfer in BCI** | Align | REWRITE | Keep as related work (EA, Procrustes, parallel transport are genuine prior art for a covariance method) but remove the sentence promising "we respond by moving the estimator inside the network as a layer". |
| **Related work → Riemannian and tangent-space** | — | KEEP | Directly the final method's literature. |
| **Related work → Test-time adaptation** | — | KEEP | The branch's reference is a test-time statistic; this is its literature. |
| **Related work → Calibration and selective prediction** | Gate | DELETE | Exists only to motivate the selective head. No part of the final model abstains. |
| **Problem formulation** | — | KEEP | Covariate shift on the spatial covariance is the branch's premise. |
| **Method → opening** | Align, CTX, Gate | REWRITE | Remove the paragraph announcing three components "tested and not retained". |
| **Method → Multi-scale power stem** | — | KEEP | Final component. |
| **Method → Tangent-space branch** | — | KEEP | Final component. Absorb the running-reference update equation here. |
| **Method → Components tested and not retained** | Align, CTX, Gate | **DELETE** | This subsection exists only to document abandoned architectures. |
| **`fig:arch`** | Align, CTX, Gate | REWRITE | Remove the greyed "built and measured, not retained" row. Two paths only. |
| **Experimental setup → Data** | — | KEEP | Five cohorts, current. |
| **Experimental setup → Preprocessing** | — | KEEP | Shared by all arms. |
| **Experimental setup → Outcome measures** | Gate | REWRITE | Remove Acc@90 / coverage, which exist only for the selective head. Keep balanced accuracy, ECE, false onsets per minute. |
| **Statistical treatment / measurement noise** | CTX | REWRITE | Noise floor is currently defined by the transformer arms' seed spread. Redefine from the branch and stem seed spreads, which are measured. |
| **Results → The drift mechanism** | Align | **DELETE** | Measures covariance-distance reduction by the alignment whitening. The branch does not whiten; nothing in the final model makes this claim. |
| **Results → Component ablation, cohort A** | Align, CTX, Gate | **REWRITE** | Rebuild as stem vs stem+branch only. Delete the eight-arm comparison. |
| **Results → Artefact control** | Align+Gate | REWRITE | The ICA control was run on the align+gate configuration. Either rerun on the branch or state the control is on the stem and narrow the claim. **Evidence gap flagged below.** |
| **Results → What the alignment layer did** | Align | **DELETE** | Written by me to preserve the inversion. Exactly the development narrative to remove. |
| **Results → Correcting the covariance estimator** | Align | **DELETE** | An estimator bug in the alignment layer's `_cov`. The branch has its own estimator; this is development history. |
| **Results → The adaptation rate condition** | Align | **REWRITE** | Keep Condition 1 and the argument; re-derive on the branch's reference. All supporting data is already branch-only. |
| **Results → A prediction registered before a third cohort** | Align | **REWRITE** | The registered prediction was about alignment on cohort C. Narrow to the cohort C result for the branch, or drop the pre-registration framing. **Evidence gap flagged below.** |
| **Results → A second-order branch** | — | KEEP, merge | Overlaps the five-cohort section; merge. |
| **Results → Five cohorts** | — | KEEP | Core result. |
| **Results → Published decoders on the two new cohorts** | — | KEEP | Current. |
| **Results → The reported model against the field** | — | KEEP | Core result (Table 3). |
| **Discussion → Validating a mechanism separately from performance** | Align | **REWRITE** | Five counterexamples are all alignment measurements. Either restate using the branch's reference-mode results (`tab:tanref`), which show the same thing, or delete. |
| **Discussion → Temporally correlated TTA** | — | KEEP | About the branch's reference. |
| **Limitations** | Align, CTX | REWRITE | Remove entries about dropped components; keep the branch's limitations. |
| **Conclusion** | — | KEEP | Already branch-only. |

### Floats

| Float | Old component | Action | Reason |
|---|---|---|---|
| `fig:arch` | all three | REWRITE | Drop the "not retained" row. |
| `tab:blocks` | — | KEEP | Protocol property, no arms. |
| `tab:hyper` | — | KEEP | Shared training config. |
| `fig:drift`, `tab:drift` | Align | **DELETE** | Alignment's whitening effect. |
| `fig:ablation` | all three | REWRITE | Stem vs stem+branch only. |
| `tab:ablation`, `tab:ablation3` | all three | **DELETE** | Archives of abandoned architectures. |
| `tab:ablation_rep` | all three | REWRITE | Keep the reported model rows; drop the seven old-arm rows. |
| `tab:artefact` | Align+Gate | REWRITE | See evidence gap. |
| `fig:topo` | Align | **DELETE** | Scalp maps of `M^{-1/2}`, the alignment transform. |
| `fig:inversion`, `tab:external` | all three | **DELETE** | Five old arms on two cohorts. |
| `fig:tracking` | Align | **KEEP, REWRITE** | Shows the running covariance tracking the streaming class. That estimator *is* the branch's reference; reframe as such, no alignment needed. |
| `fig:rate` | — | KEEP | Already regenerated on the branch (μ manipulation). |
| `tab:prereg` | Align | REWRITE | See evidence gap. |
| `tab:tangent`, `tab:tanref`, `tab:five`, `tab:batchsweep`, `tab:newbase`, `tab:compare`, `fig:cohorts` | — | KEEP | All branch-only. |
| `tab:rate`, `tab:ratecurve` | Align | **DELETE** | Superseded by `tab:batchsweep` on the branch. |
| `tab:repr` | — | KEEP | Identifying evidence for the stem, which is in the final model. |
| `tab:norm` | — | KEEP | Normalisation control, applies to the stem. |
| `tab:noise` | CTX, Gate | REWRITE | Rebuild from the branch and stem seed spreads. |
| `fig:metrics` | Align | REWRITE or DELETE | See Discussion row. |

---

## Evidence that will be missing after this cleanup

Stated explicitly rather than papered over, per the instruction not to fabricate
replacements.

1. **Artefact control.** The ICA comparison (0.836 without ICA against 0.785
   with) was run on the align+gate configuration. After cleanup there is no
   artefact control for the reported model. Either it is rerun on
   `dn_stem_tan` (one job, cohort A, three seeds) or the artefact claim is
   removed. **Recommend rerunning; it is cheap and the claim is worth having.**

2. **Pre-registration.** The cohort C predictions registered in advance were
   about alignment. The cohort C branch result (+0.064, 20/20, p<10⁻⁴) is
   genuine but was *not* pre-registered as such. After cleanup the paper can no
   longer claim a pre-registered prediction unless
   `paper/PREREGISTRATION_derived_rate.md` is cited instead, which registered
   the derived-rate prescription and was falsified. **Recommend: drop the
   pre-registration claim, keep the falsified-prescription report.**

3. **Mechanism-versus-performance counterexamples.** The five cases in
   `fig:metrics` are alignment measurements. `tab:tanref` shows the same
   dissociation on the branch (the no-reference variant improves cohort B while
   being null on C), but it is one case, not five. **The claim must narrow from
   "five counterexamples" to what the branch data supports.**

4. **Drift quantification.** The 45.9 % / 62.5 % covariance-distance reductions
   are alignment-only. No equivalent number exists for the branch, which does
   not whiten. **The drift-reduction claim is removed entirely**; the
   longitudinal motivation survives in the Introduction without it.

---

## Order of execution

1. Delete: drift mechanism, alignment-findings, estimator-correction sections
   and their floats.
2. Delete: Components tested and not retained; rewrite `fig:arch`.
3. Rebuild ablation as stem vs stem+branch.
4. Rewrite the rate condition on the branch's reference; delete `tab:rate`,
   `tab:ratecurve`.
5. Outcome measures, noise, keywords.
6. Discussion, Limitations.
7. Global terminology audit; rebuild; verify no unresolved references and no
   truncating percent signs.
