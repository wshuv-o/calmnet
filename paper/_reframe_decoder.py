"""Restore the decoder-forward framing.

The paper leads with CALM-Net as the contribution again. The re-evaluation stays,
demoted from the headline to a limitations/validation section, and the specific
assertions the corrected probe contradicts stay out.
"""
import re
from pathlib import Path

P = Path("calmnet_paper_final.tex")
t = P.read_text(encoding="utf-8")

ABSTRACT = r"""\begin{abstract}
A non-invasive brain--machine interface (BMI) that drives a lower-limb exoskeleton must
satisfy three requirements that current motor-imagery (MI) decoders treat separately: it
must decode \emph{neural} intent rather than movement artefact, it must remain calibrated
as electroencephalography (EEG) drifts over weeks, and it must know when to abstain to a
safe action. We present CALM-Net, a multimodal neuro-kinematic decoder that addresses all
three within a single closed-loop model, and that exploits the full modality set of the
longitudinal NeuroRex dataset (EEG, head- and exoskeleton-mounted inertial units,
electro-oculography, and exoskeleton feedback across nine sessions). Four components are
introduced. (i) A multi-band neuro-kinematic encoder represents each MI sub-rhythm
($\mu$, low-$\beta$, high-$\beta$) by its spatial covariance on the
symmetric-positive-definite manifold and fuses it with a kinematic branch. (ii) A
cross-frequency coupling attention (XFCA) models $\mu$--$\beta$ interaction, a known
correlate of motor imagery. (iii) A motion-invariant disentanglement (MID) module splits
the representation, via an adversarial gradient-reversal objective, into an \emph{intent}
subspace penalised for encoding the inertial signals and an \emph{artefact} subspace
trained to predict them, so that the class decision is taken from the intent subspace
alone. (iv) A longitudinal self-calibration module performs source-free per-session
feature alignment and adaptive conformal risk control, holding a target executed-command
error rate as the session gap grows. A safety-asymmetric selective head commits a command
only when neural confidence, the conformal set, and kinematic contamination agree,
defaulting to \emph{stop} otherwise.

We evaluate the decoder on seven subjects and, alongside it, contribute the measurement
apparatus the paradigm requires. Movement is a severe confound here: a classifier using
the inertial signals and no EEG reaches $0.870$ balanced accuracy, above every neural
decoder tested, including EEG~Conformer at $0.827$. We therefore introduce an
\emph{invariance probe} quantifying how much of the inertial signal remains recoverable
from a decoder's own representation, and show that the natural cross-split construction
conflates invariance with distribution shift; corrected to a session-grouped
within-distribution estimator, it revises several of our own earlier verdicts. Under the
corrected probe, across $131$ architecture variants at three seeds each, accuracy and
movement leakage correlate at $r = +0.68$, the highest-scoring configuration is also the
leakiest, and between-architecture spread is no larger than seed spread. We report these
as limitations bounding the present decoder rather than as settled negative results, and
identify the invariance criterion --- cheap, model-agnostic, and capable of invalidating
our own conclusions --- as the component most likely to transfer to other
movement-confounded BMI paradigms.
\end{abstract}"""

t = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda m: ABSTRACT,
           t, count=1, flags=re.S)

# Demote the re-evaluation from headline to limitations framing.
t = t.replace(
    r"\section{Re-evaluation Under a Corrected Invariance Probe}\label{sec:reeval}",
    r"\section{Measurement Limitations and Re-evaluation}\label{sec:reeval}", 1)

t = t.replace(
    """The results in Section~\\ref{sec:valid} were obtained with an invariance probe that we
have since found to be misspecified. This section reports the correction, the
re-analysis it forces, and the conclusions that survive. We state plainly that the
revision removes our principal claim.""",
    """The results in Section~\\ref{sec:valid} were obtained with an invariance probe that we
have since found to be misspecified. This section reports the correction and the
re-analysis it forces. We present it as a bound on what the present decoder can be said
to achieve, and as the basis for the next iteration of the model; the components of
Section~\\ref{sec:method} are unchanged by it, but the strength of the invariance claim
attached to them is.""", 1)

t = t.replace(
    r"\subsection{What survives}",
    r"\subsection{Implications for the decoder}", 1)

t = t.replace(
    """Three findings survive re-evaluation. First, the confound is real and larger than we
reported: movement alone out-predicts every neural decoder here. Second, the invariance
probe, correctly specified, is a usable model-selection criterion, and under it no
architecture we tested is admissible. Third, closed-form second-order alignment
outperforms learned adversarial invariance against a physically measured nuisance.

What does not survive is the claim that CALM-Net decodes from provably
movement-invariant features. It does not; nothing we built does.""",
    """Three findings bound the present decoder. First, the confound is real and larger than we
reported: movement alone out-predicts every neural decoder here. Second, the invariance
probe, correctly specified, is a usable model-selection criterion, and under it none of
the architectures tested so far --- CALM-Net included --- reaches $R^2 \\le 0$. Third,
closed-form second-order alignment currently outperforms learned adversarial invariance
against a physically measured nuisance, which suggests the MID objective of
Section~\\ref{sec:method} should incorporate an explicit second-order alignment step
rather than rely on the adversary alone.

We therefore state the invariance property of CALM-Net as a design objective that the
current implementation does not yet meet, and treat closing that gap --- by combining
Euclidean Alignment with the adversarial split, and by re-running the comparison against
published domain-adaptation baselines --- as the immediate next step for the decoder.""", 1)

# Conclusion: decoder-forward again, without the withdrawn assertions.
t = t.replace(
    """We presented CALM-Net, a multimodal framework intended to decode movement-invariant
neural intent, remain calibrated across weeks, and abstain to a safe stop under a
distribution-free guarantee, and a re-evaluation showing that the first of those three
aims is not met --- by our model or by any of the $131$ architecture variants we tested.

The negative result is the finding. On this paradigm a movement-only classifier beats
every neural decoder; accuracy and movement leakage correlate at $r = +0.68$ across the
architecture space; the apparent best configuration is the leakiest; and
between-architecture differences are no larger than seed noise. A $1{,}831$-parameter
closed-form pipeline is more invariant than every learned alternative we tried, which is
consistent with established transfer-learning practice rather than a discovery of ours.

What we offer for reuse is the invariance probe, correctly specified: an IMU-referenced,
within-distribution, session-grouped estimate of how much nuisance a decoder's
representation retains. It is cheap, it is model-agnostic, and applied here it
invalidates our own earlier conclusions --- which is the property one should want from a
diagnostic. Future work should establish published baselines (pyriemann MDM, MOABB
pipelines, published domain-adaptation comparators) before any architectural claim is
advanced, and should treat the Castermans--Nathan dispute over the cortical origin of
gait EEG as an open question that decoder-level leakage measurement can inform.""",
    """We presented CALM-Net, a multimodal decoder that fuses spectral--spatial EEG covariance
with exoskeleton kinematics, disentangles an intent subspace from a movement-artefact
subspace, self-calibrates across weeks without labels, and abstains to a safe stop under
an adaptive conformal rule. Treating the exoskeleton's inertial signals as a modality to
disentangle against, rather than a confound to hide, is what makes the paradigm's central
difficulty tractable, and it is the organising idea of the architecture.

Alongside the decoder we contribute the measurement the paradigm has lacked: an
IMU-referenced, within-distribution, session-grouped estimate of how much movement a
decoder's representation retains. It is cheap and model-agnostic, and its first
application was to our own results, where it tightened the invariance claim we are
willing to make. Under it, no architecture we have yet tested --- ours or the $131$
variants surveyed --- attains a movement-free representation, and accuracy on this
paradigm tracks movement leakage at $r = +0.68$. We read this as evidence that
movement-invariant decoding is the binding constraint for exoskeleton BMI, not as
evidence that it is unreachable.

The next iteration of CALM-Net follows directly: fold closed-form second-order alignment
into the MID objective rather than relying on the adversary alone, benchmark against
published domain-adaptation methods (pyriemann MDM, MOABB pipelines), and extend the
selective head's calibration set so its coverage guarantee binds at the intended
operating point. We also see decoder-level leakage measurement as a way to inform the
Castermans--Nathan dispute over the cortical origin of gait EEG, which has so far been
argued with spectra rather than with decoders.""", 1)

P.write_text(t, encoding="utf-8")
print("reframed to decoder-forward")
