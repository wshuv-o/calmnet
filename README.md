# CALM-Net

**A Multimodal, Motion-Disentangled, Longitudinally Self-Calibrating framework for
closed-loop lower-limb exoskeleton control**, on the NeuroRex dataset
(OpenNeuro **ds007788**, 7 subjects × 9 sessions over weeks).

The Walk/Stop label on this paradigm is partly a *movement* label (the exoskeleton
moves the wearer during Walk). CALM-Net treats the inertial signals as a modality to
**disentangle against** rather than a confound to hide: it decodes movement-invariant
neural intent, stays calibrated across sessions, and abstains to a safe *stop* when
unsure. EEGNet / EEG-Conformer are comparison **baselines**, not the backbone.

## Components
- **MID** — motion-invariant disentanglement: adversarial gradient-reversal split
  into an intent subspace (forced invariant to a 12-D head+exo IMU vector) and an
  artefact subspace (forced to predict it) + a cross-covariance decorrelation penalty.
- **LSC** — longitudinal self-calibration: temperature scaling + split and
  **adaptive conformal** (Gibbs–Candès) that holds prediction-set coverage under drift.
- **SAS** — safety-asymmetric selective abstention (confidence ∧ conformal singleton
  ∧ kinematic contamination → else STOP).

## Layout
```
src/
  download.py        # OpenNeuro subset fetch (openneuro-py)
  dataio.py          # BIDS/EDF load, EOG regression, 8-30 Hz, rexstate epoching,
                     #   12-D head+exo IMU features per window
  models.py          # EEGNet, EEG Conformer (baselines) + CALMNet band-power backbone
  mid.py             # CALMNetMID: gradient-reversal disentanglement + invariance probe
  calibrate.py       # temperature scaling, ECE/Brier, split + adaptive conformal
  abstain.py         # risk-coverage, selective risk, confidence AUROC, executed accuracy
  splits.py          # segment-grouped, leakage-free splits
  experiments.py     # baseline longitudinal sweep + confound controls
  experiments_multi.py  # baselines across all 7 subjects
  exp_mid.py         # MID validation (invariance vs accuracy) across subjects
  calmnet_full.py    # full pipeline: MID -> temp -> adaptive conformal -> abstention
  make_figures*.py   # result figures
data/ds007788/       # downloaded EEG (gitignored)
results/             # *.json metrics, figures, FINDINGS.md
paper/               # IEEE journal + conference LaTeX, draw.io architecture, figures
```

## Reproduce
```bash
# CPU env has torch-cpu; GPU runs use the testenv interpreter + KMP flag:
#   D:/EEG-TransNet/testenv/python.exe  with  KMP_DUPLICATE_LIB_OK=TRUE
python src/download.py sub-01 ... sub-07      # ~3.7 GB, all 9 sessions each
python src/experiments_multi.py               # baselines -> results/multi_subject.json
python src/exp_mid.py                         # MID validation -> results/mid_validation.json
python src/calmnet_full.py                    # full pipeline -> results/calmnet_full.json
```

## Labels
`training` task only (scripted open-loop MI calibration block). Labels from the
`rexstate` events: `x81`→Walk, `x0`→Idle/Stop; transitions `x5`/`x8` excluded.
2 s windows @ 100 Hz, 0.5 s step, per-epoch per-channel z-score.

## Headline results (mean over 7 subjects, longitudinal)
- Movement baseline (IMU-only): **0.84** — matches/beats the raw EEG decoders, so
  raw accuracy is movement-inflated.
- **MID** drives movement-recoverability from the neural code to zero (intent→IMU
  R² 0.15 → −0.31); the honest movement-invariant decode is **~0.69**, above chance
  for every subject.
- **Full pipeline:** ECE 0.154 → 0.114, executed accuracy 0.694 → 0.719 @ 80%
  coverage, and **adaptive conformal holds 0.90 coverage on all 7 subjects** (static
  drifts 0.78–0.95).

Full analysis: `results/FINDINGS.md`. Paper: `paper/calmnet_paper_journal.pdf`.
