# CALM-Net — Results & Findings (ds007788)

End-to-end pipeline: OpenNeuro download → BIDS/EDF preprocessing → Walk/Stop
epoching → decoders → temperature scaling + split/adaptive conformal → selective
abstention, evaluated **longitudinally** (per subject: train on sessions 1–3, test
on the later 6; `training` task only, 60 ch × 2 s @ 100 Hz, 7 subjects).

The story has three parts: **(A)** baselines expose a movement confound that
motivates the design; **(B)** motion-invariant disentanglement (MID) recovers an
honest neural decode; **(C)** the full CALM-Net pipeline adds calibration,
abstention, and a distribution-free longitudinal coverage guarantee.

---

## A. Baselines + the movement confound (motivation) — `fig_multi_subject.png`

Mean over each subject's 6 test sessions:

| subj | span | EEGNet | Conformer | Conf-AUROC | **IMU-only** | motor-only |
|---|---|---|---|---|---|---|
| sub-01 | 21d | 0.930 | 0.906 | 0.845 | 0.862 | 0.862 |
| sub-02 | 15d | 0.792 | 0.760 | 0.783 | 0.791 | 0.814 |
| sub-03 | 14d | 0.894 | 0.882 | 0.929 | **0.981** | 0.878 |
| sub-04 | 35d | 0.741 | 0.865 | 0.890 | **0.964** | 0.899 |
| sub-05 | 44d | 0.849 | 0.769 | 0.798 | 0.886 | 0.760 |
| sub-06 | 80d | 0.868 | 0.793 | 0.833 | **0.561** | 0.769 |
| sub-07 | 39d | 0.657 | 0.808 | 0.835 | 0.828 | 0.610 |
| **MEAN** | | 0.819 | 0.826 | 0.845 | **0.839** | 0.799 |

- **The movement confound is dataset-wide.** Mean IMU-motion-only accuracy (**0.839**)
  matches/beats both EEG decoders; for sub-03/04 a single motion feature *beats* the
  60-ch net. The `training`-task Walk/Stop label is largely a *movement* label
  (head-IMU acceleration ~8.4× higher in Walk; per-subject channel controls:
  chance 0.50 < frontal/EOG 0.73 < motor 0.86 ≈ IMU 0.86 < full-EEG 0.93 on sub-01).
- **sub-06 proves genuine neural signal exists.** There IMU is near chance (0.561)
  yet EEG decodes 0.79–0.87 — when movement is uninformative the EEG still carries
  Walk/Stop signal.
- **EEGNet / Conformer are baselines, not the contribution.** Conformer is the more
  robust, better-calibrated baseline (usable confidence AUROC 0.78–0.93 vs EEGNet's
  saturated ~0.60), but neither addresses the confound. That is what CALM-Net does.

## B. Motion-invariant disentanglement (MID) — `paper/fig_mid.pdf`

An adversarial gradient-reversal split (intent vs artefact subspace) disentangled
against a **12-D head+exo IMU vector** (accel/gyro magnitude mean/std/max) plus a
cross-covariance decorrelation penalty. Compared with the same backbone, MID off.

| | no MID | + MID |
|---|---|---|
| intent→IMU R² (nonlinear probe, mean) | 0.15 | **−0.31** |
| balanced accuracy (mean) | 0.81 | 0.65 |

- **Invariance achieved**, including on the movement-dominated subjects where a
  scalar-target v1 failed: sub-03 R² 0.56 → −0.14, sub-04 0.36 → −0.21. Movement is
  no longer recoverable from the code that drives the command.
- **The honest neural decode is ~0.65**, above chance for every subject, with the
  largest accuracy drop exactly where the IMU baseline is highest — that drop *is*
  the movement contribution being removed. Conservative lower bound (MID slightly
  over-regularises when motion is already uninformative, e.g. sub-06).
- This is, to our knowledge, the **first movement-invariant estimate of neural
  walk/stop decodability** on this dataset. (v1 with a scalar target did **not**
  work — the 12-D target + decorrelation penalty were necessary.)

## C. Full CALM-Net pipeline — `paper/fig_full.pdf`, `results/calmnet_full.json`

MID decoder → temperature scaling → split + **adaptive conformal** → selective
abstention, run longitudinally. Mean over 7 subjects:

| metric | value | note |
|---|---|---|
| movement-invariant acc | 0.694 | above chance 6/7 (sub-02 at chance) |
| ECE raw → +temperature | 0.154 → **0.114** | improves 5/7 |
| confidence AUROC | 0.698 | usable where neural signal exists |
| executed acc @ 80% coverage | 0.694 → **0.719** | abstention lifts committed accuracy |
| **adaptive conformal coverage** | **0.902** (target 0.90) | holds on every subject (0.898–0.905) |
| static conformal coverage | 0.880 | drifts under session gap (0.78–0.95) |

- **Adaptive conformal is the flagship result:** it holds the 0.90 coverage target
  on all 7 subjects under cross-session drift, where static conformal misses it —
  the distribution-free longitudinal guarantee the framework is built for.
- **Graceful degradation:** sub-02 (decode at chance) is flagged by uninformative
  confidence (AUROC 0.47) rather than emitting confident errors.
- Temperature needs a sane clamp ([0.5, 5]); an unconstrained fit on well-separated
  training-session logits collapses to pathological sharpening (T≪1) under drift.

---

## Framing for the paper
- Lead with the **framework** (multimodal + MID + LSC + SAS), not an accuracy number.
- The movement confound is the *motivation*, resolved by MID — not a caveat to
  apologise for.
- Decisive metrics are executed-command safety and calibrated/covered confidence.
- EEGNet / EEG-Conformer are comparison **baselines**.

## D. Encoder ablation: band-power vs Riemannian+XFCA (`results/calmnet_riemann.json`)

The full designed encoder (multi-band spatial covariance → SPD tangent space → XFCA
cross-frequency attention + source-free CORAL) is implemented (`src/riemann.py`) and
run through the same MID + calibration + adaptive-conformal pipeline. Honest trade-off:

| Encoder | Acc | ECE | AUROC | exec@80 | cov(adpt) | intent→IMU R² (noMID→MID) |
|---|---|---|---|---|---|---|
| Band-power (primary) | 0.694 | 0.114 | 0.698 | 0.719 | 0.902 | 0.15 → **−0.31** |
| Riemannian+XFCA | **0.805** | 0.137 | 0.741 | **0.835** | 0.900 | 0.13 → −0.24 |

- **Riemannian+XFCA is a much stronger decoder** (0.805 vs 0.694) but its accuracy is
  **more movement-coupled**: MID disentangles it less/inconsistently and *fails* on the
  two highest-accuracy subjects (sub-01 R² 0.30→0.43, sub-03 0.63→0.39 — still
  movement-predictive); calibration is worse on sub-03/04 (ECE ~0.3).
- **Band-power + MID disentangles more cleanly** (R² → −0.31, drops sub-03 to −0.14) →
  the trustworthy movement-invariant estimate. **Adaptive-conformal coverage holds for
  both** (0.902 / 0.900) — encoder-agnostic guarantee.
- **HSIC test (`results/riemann_hsic.json`, `src/exp_riemann_hsic.py`):** a nonlinear
  Hilbert-Schmidt independence penalty + disentanglement-aware model selection drives
  the Riemannian invariance much further (mean intent→IMU R² −0.24 → **−0.465**, below
  noMID for every subject), but accuracy **collapses 0.805 → 0.645**. This confirms the
  Riemannian encoder's 0.80 was **movement leakage**: enforcing genuine invariance
  lands it at ~0.65, comparable to band-power. **The ~0.7 movement-invariant decode is
  a ceiling robust to encoder richness AND disentanglement strength.**
- **Implication for improving results:** better disentanglement will NOT raise the
  honest accuracy (the missing accuracy *is* the movement). To raise the ceiling,
  attack the confound at the source: decode the **pre-movement preparation window**,
  or **regress the IMU out of the EEG** at the signal level. These change the data, not
  the model.
- Conclusion: report band-power for the movement-invariant claim, Riemannian+XFCA as
  the higher-accuracy-but-coupled alternative. Written up in the extended paper
  (`paper/calmnet_paper_extended.pdf`, encoder-ablation).

## E. Pre-movement window test (`src/exp_prep_window.py`, `results/prep_window_run.log`)

Tested whether decoding the **Stop→Walk transition onset** (before head motion builds)
reduces the confound. It does **not**: IMU-only baseline stays **0.936** (stable) vs
**0.936** (prep), and EEG decode drops 0.849→0.787. The exoskeleton commits to motion
at the command instant (the exo IMU separates classes from t=0), and the walk/stop
label *is* the exo state — **there is no movement-free window.** Confirms the ~0.7
movement-invariant ceiling is intrinsic to the paradigm, not fixable by windowing.

**Overall conclusion on "improving the result":** the movement-invariant accuracy
(~0.65-0.70) is a hard, data-imposed ceiling — confirmed by (i) HSIC showing better
disentanglement can't raise it, and (ii) the prep-window test showing no movement-free
contrast exists. The improvable axes are the *safety* metrics (per-session
recalibration, class-conditional conformal for the wrong-walk cost, deep ensembles)
and using more data (walk6min/stop6min); raising the *accuracy* ceiling requires a
different paradigm (movement-free MI) or patient data. The contribution is the
calibrated, abstaining, coverage-guaranteed framework, not the accuracy number.

## F. Safety-metric improvements (`src/exp_safety.py`, `results/safety.json`)

Applied to the primary band-power CALM-Net, longitudinal, mean over 7 subjects:

| metric | baseline | improved | verdict |
|---|---|---|---|
| confidence-AUROC | 0.697 | **0.740** (K=3 ensemble) | ✓ consistent |
| executed acc @80% | 0.719 | **0.771** (ensemble) | ✓ |
| **wrong-walk rate** | 0.144 (argmax) | **0.051** (class-cond. threshold, target 0.05) | ✓✓ every subject |
| ECE | 0.114 | 0.170 (ensemble) / 0.192 (per-session) | ✗ no gain |

- **Wrong-walk control is the key win**: a class-conditional threshold calibrated to
  bound P(commit walk | true stop) cuts the dangerous false-walk rate ~3× to the 0.05
  target on all 7 subjects, routing walk-uncertain windows to safe stop (walk-recall
  0.37 mean). Directly realises the wrong-walk ≫ wrong-stop asymmetry.
- **Deep ensemble (K=3)** improves confidence ranking (AUROC, exec@80) consistently.
- **Honest null**: ensembling and short per-session recal do NOT lower ECE on the
  already-well-calibrated band-power model (0.114→0.17); we keep the single global
  temperature. Gains are in ranking + bounding the dangerous error, not ECE.
- Written up in the extended paper (Discussion, "Controlling the dangerous error").

## G. Data augmentation with closed-loop trials (`exp_extra_data.py`, `exp_extra_invariance.py`)

Added trial01-12 (rexstate-labelled, walk/stop interleaved within each recording -> no
recording-level confound; walk6min/stop6min deliberately avoided as single-recording-
per-class). ~+4.5k training epochs/subject. Mean accuracy 0.694 -> 0.753, BUT the
invariance probe (intent->IMU R²) shows the gain is **subject-dependent, not uniform**:

| | genuine invariant gain | movement leakage |
|---|---|---|
| data-starved subjects | sub-02 0.52→0.74 (R²≈−0.08), sub-05 0.66→0.77 (R²=−0.22) | — |
| already-strong subject | — | sub-01 0.79→0.87 (R² 0.08→**0.23**) |
| flat | sub-03/04/07 (acc ~unchanged) | — |

**Honest read:** more data helps subjects that were data-limited *reach* the ~0.7
invariant ceiling (sub-02 rescued from chance, invariantly), but leaks movement on
subjects already near it. It does NOT push the ceiling higher. Safe recommendation:
more data + stronger (HSIC) disentanglement to suppress the leakage. (An earlier
per-subject "LEAKAGE" auto-flag over-called this; the absolute aug-R² shows most gains
are invariant.)

## Open / next
- More data + HSIC disentanglement jointly (prevent the sub-01-style leakage).
- Complete reference details + fill author/affiliation in the paper.

_Artifacts: `results/{longitudinal,multi_subject,mid_validation,calmnet_full}.json`,
`results/fig_*.png`, `paper/fig_{architecture,mid,full}.pdf`._
