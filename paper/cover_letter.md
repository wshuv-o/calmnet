# Cover letter

To the Editor-in-Chief
Applied Soft Computing

**Re: Submission of "Label-free covariance alignment for longitudinal EEG decoding"**

Dear Editor,

We submit the above manuscript for consideration as an original research
article in Applied Soft Computing.

A brain–computer interface that drives a powered lower-limb exoskeleton is
fitted once and must keep working for weeks, while electrode impedance,
montage and skin contact alter the second-order statistics that every spatial
filter depends on. Correcting that shift inside the network, by re-estimating
the input covariance at inference and whitening by it without labels, removes
the need for a per-session calibration block. We show that this design works,
characterise a failure mode it carries that its offline counterpart cannot,
and give a check that predicts which of the two regimes a given recording
protocol falls into before any model is trained.

The manuscript reports three results. First, a 24,181-parameter decoder
reaches 0.901 balanced accuracy with an expected calibration error of 0.039 on
a seven-participant, nine-session exoskeleton cohort recorded across weeks, at
53% of the parameters of the strongest baseline evaluated in the same pipeline.
Second, the alignment layer reduces session-to-session covariance distance by
52.0 ± 4.5% and 83.9 ± 3.9% on two independent cohorts, in 15 of 15
participants (p ≤ 10⁻⁴), against a control that shares every code path but
never updates its estimate. Third, a running covariance estimate tracks
whichever class is currently streaming; where a protocol presents classes in
blocks longer than the estimator's memory, the layer whitens away the signal it
was inserted to protect. The admissible adaptation rate is therefore bounded
below by the class-block timescale and above by the drift timescale, and both
bounds are properties of the experimental design rather than of the model.

We draw the editor's attention to two aspects of the work. Evaluation is
external: the second cohort was recorded in an independent laboratory, with
different sensors, a different task framing and reversed class balance. The
decoder transfers, reaching 0.792 there, while the alignment layer does not,
and the manuscript reports that inversion in full rather than confining itself
to the cohort on which the layer succeeds. We also report a component that did
not work: the trained selective head lowers balanced accuracy on retained
windows in 14 of the 18 arms measured, and we present this as a negative result
rather than as a feature of the architecture.

The work suits Applied Soft Computing in combining a compact neural
architecture for a physiological control problem with an analysis of when a
widely used adaptation technique is applicable. Its practical contribution is a
design-time check that requires no training run.

The manuscript is original, has not been published previously, and is not under
consideration elsewhere. All authors have approved the submission and declare
no competing financial or personal interests. The data supporting the primary
cohort are publicly available (OpenNeuro ds007788).

Thank you for considering our submission.

Yours sincerely,

**Md Wahiduzzaman Suva**
Department of Computer Science
American International University-Bangladesh, Dhaka, Bangladesh
26-94088-2@student.aiub.edu
*on behalf of Esm E Moula Chowdhury Abha and Dr. Muhammad Hasibur Rashid Chayon*
