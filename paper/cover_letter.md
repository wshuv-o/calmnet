# Cover letter

To the Editor-in-Chief
Applied Soft Computing

**Re: Submission of "Label-free covariance alignment for longitudinal EEG decoding"**

Dear Editor,

We submit the above manuscript for consideration as an original research
article in Applied Soft Computing.

A brain–computer interface that drives a powered lower-limb exoskeleton is
fitted once and must keep working for weeks, while electrode impedance and
montage change the covariance that its spatial filters depend on. We place a
label-free covariance alignment layer inside a compact decoder, so that each new
session is whitened by a running estimate of its own covariance at inference,
without a calibration block. We also identify when such a layer fails, and give
a check, read from the recording protocol, that predicts this before any model
is trained.

The manuscript reports three results. First, the alignment layer reduces
session-to-session covariance distance by 45.9 ± 4.9% and 62.5 ± 12.2% on two
independent cohorts, in 15 of 15 participants, against a control that shares
every code path but never updates its estimate. Second, the 24,181-parameter
decoder reaches 0.868 ± 0.026 balanced accuracy over three data-split seeds on a
seven-participant exoskeleton cohort recorded across weeks, level with eight
published decoders trained in the same pipeline, and 0.796 with an expected
calibration error of 0.061 on a twenty-participant motor-execution cohort,
against 0.783 for the strongest published decoder. Third, a running estimate
tracks whichever class is currently streaming. Where class blocks outlast the
estimator's memory, the layer removes the class signal, which costs 0.149
balanced accuracy on a treadmill cohort, lower in 8 of 8 participants. The lower
bound on the adaptation rate that follows from this was tested on the
motor-execution cohort with both predictions committed before any model was
trained on it, and both held.

We draw the editor's attention to two aspects of the work. The evaluation spans
three cohorts from different laboratories and paradigms, with every published
decoder trained in one pipeline over the same seeds and compared by paired tests
across participants. The manuscript also states the limits of its evidence: on
the exoskeleton cohort no component, alignment included, is shown to change
accuracy relative to the convolutional stem alone, a trained selective head adds
no accuracy, and the upper bound on the adaptation rate could not be measured
on these cohorts.

The work suits Applied Soft Computing in combining a compact neural architecture
for a physiological control problem with an analysis of when a widely used
adaptation technique applies. Its practical contribution is a design-time check
that requires no training run.

The manuscript is original, has not been published previously, and is not under
consideration elsewhere. All authors have approved the submission and declare
no competing financial or personal interests. All three datasets are publicly
available, and all code, configurations and result files are in the
accompanying repository.

Thank you for considering our submission.

Yours sincerely,

**Md Wahiduzzaman Suva**
Department of Computer Science
American International University-Bangladesh, Dhaka, Bangladesh
26-94088-2@student.aiub.edu
*on behalf of Esm E Moula Chowdhury Abha and Dr. Muhammad Hasibur Rashid Chayon*
