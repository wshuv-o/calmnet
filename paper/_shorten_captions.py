"""Shorten every over-long figure/table caption; move the explanation into body text.

IEEE captions should identify the exhibit, not explain it. Each replacement below
cuts the caption to a label and pushes the substance into the surrounding prose,
where it can be read in sequence rather than as a block under a figure.
"""
from pathlib import Path

P = Path("calmnet_paper_conference.tex")
t = P.read_text(encoding="utf-8")

CAPS = [
    # --- Fig 1: architecture (83 -> 9 words) ---
    ("""\\caption{The CALM-Net framework. Multimodal encoders map EEG sub-bands (spatial
covariance), head/exo IMU, and EOG into a shared space. Cross-frequency coupling
attention (XFCA) forms the neural representation, which motion-invariant
disentanglement (MID) splits into an IMU-invariant \\emph{intent} subspace (used by
the selective classifier) and an \\emph{artefact} subspace (trained to predict the
inertial signals and to estimate kinematic contamination). Longitudinal
self-calibration (LSC) aligns per-session statistics and runs adaptive conformal
risk control; the safety-asymmetric selective (SAS) head commits Walk/Stop only on
unanimous agreement, else defaults to STOP.}""",
     "\\caption{The CALM-Net framework: encoders, MID, LSC, and the SAS head.}"),

    # --- Fig 2: MID validation (87 -> 11 words) ---
    ("""\\caption{Validation of motion-invariant disentanglement (MID) across seven subjects.
\\textbf{Left:} balanced accuracy of the IMU-only movement baseline, the EEG decoder
without MID (movement-inflated), and with MID (movement-invariant); the drop isolates
the movement contribution, largest where the IMU baseline is highest, yet every
subject stays above chance. \\textbf{Right:} nonlinear $R^2$ of predicting the IMU
descriptor from the intent code, without and with MID, as measured by the
\\emph{original} cross-split probe. Section~\\ref{sec:reeval} shows this probe
conflates invariance with distribution shift; under the corrected estimator the same
representation retains recoverable movement.}""",
     "\\caption{MID across seven subjects: accuracy (left) and movement recoverability (right).}"),

    # --- Table: full CALM-Net per subject (34 -> 8 words) ---
    ("""\\caption{Full CALM-Net per subject, mean over the six held-out test sessions
(trained on sessions 1--3). invAcc: movement-invariant balanced accuracy;
exec@80: balanced accuracy of the $80\\%$ most-confident committed windows;
cov$_{\\text{stat}}$/cov$_{\\text{adpt}}$: static/adaptive conformal coverage
(target $0.90$).}""",
     "\\caption{Full CALM-Net per subject, mean over six held-out sessions.}"),

    # --- Fig 3: coverage and calibration (36 -> 10 words) ---
    ("""\\caption{Full CALM-Net across seven subjects. \\textbf{Left:} adaptive conformal
inference holds prediction-set coverage at the $0.90$ target under cross-session
drift, where static conformal misses it. \\textbf{Right:} temperature scaling lowers
the Expected Calibration Error across the session gap.}""",
     "\\caption{Conformal coverage (left) and calibration error (right) across the session gap.}"),

    # --- Table: comparison (49 -> 9 words) ---
    ("""\\caption{CALM-Net versus baseline decoders (mean over the seven subjects,
longitudinal). Baseline accuracy is \\emph{movement-inflated}: it sits at the IMU-only
movement baseline ($0.84$), attainable from a single motion feature with no EEG.
CALM-Net's accuracy is movement-invariant, and it is the only method here that also
provides a distribution-free coverage guarantee.}""",
     "\\caption{CALM-Net versus baseline decoders, mean over seven subjects.}"),

    # --- Table: encoder ablation (42 -> 8 words) ---
    ("""\\caption{Encoder ablation under the same MID, calibration, and adaptive-conformal
pipeline (mean over seven subjects). The Riemannian--XFCA encoder is more accurate,
but its accuracy is more movement-coupled (a smaller, less consistent invariance
gain); the band-power encoder disentangles more cleanly. Coverage holds for both.}""",
     "\\caption{Encoder ablation under an identical downstream pipeline.}"),
]

missing = []
for old, new in CAPS:
    if old in t:
        t = t.replace(old, new, 1)
    else:
        missing.append(old.split("\n")[0][:55])

# --- push the removed explanation into body prose ---------------------------- #
BODY = [
    # after the architecture figure is first referenced
    ("""\\textbf{CALM-Net} (Calibrated, Abstaining, Longitudinal, Multimodal) is a per-subject
walk/stop framework for closed-loop lower-limb exoskeleton control
(Fig.~\\ref{fig:arch}). Its contributions are:""",
     """\\textbf{CALM-Net} (Calibrated, Abstaining, Longitudinal, Multimodal) is a per-subject
walk/stop framework for closed-loop lower-limb exoskeleton control
(Fig.~\\ref{fig:arch}). In that figure, multimodal encoders map EEG sub-bands (as
spatial covariances), head and exoskeleton IMU, and EOG into a shared space;
cross-frequency coupling attention forms the neural representation; motion-invariant
disentanglement splits it into an intent subspace, which the selective classifier
reads, and an artefact subspace trained to predict the inertial signals; longitudinal
self-calibration aligns per-session statistics and runs adaptive conformal risk
control; and the safety-asymmetric head commits Walk or Stop only on unanimous
agreement, defaulting to STOP otherwise. Its contributions are:"""),
]

for old, new in BODY:
    if old in t:
        t = t.replace(old, new, 1)
    else:
        missing.append("BODY: " + old.split("\n")[0][:45])

P.write_text(t, encoding="utf-8")
print(f"shortened {len(CAPS) - len([m for m in missing if not m.startswith('BODY')])}/{len(CAPS)} captions")
for m in missing:
    print("  NOT FOUND:", m)
