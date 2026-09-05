"""Build the final submission .tex from calmnet_paper_complete.tex.

Replaces the abstract and inserts a re-evaluation section carrying the corrected
metrics. The original file is left untouched; output is calmnet_paper_final.tex.
"""
import re
from pathlib import Path

SRC = Path("calmnet_paper_complete.tex")
DST = Path("calmnet_paper_final.tex")
src = SRC.read_text(encoding="utf-8")

ABSTRACT = r"""\begin{abstract}
A non-invasive brain--machine interface (BMI) driving a lower-limb exoskeleton must
decode \emph{neural} intent rather than movement artefact, stay calibrated as
electroencephalography (EEG) drifts across weeks, and abstain to a safe action when
uncertain. We present CALM-Net, a multimodal neuro-kinematic framework addressing all
three, together with a systematic re-evaluation that substantially revises our earlier
conclusions and, we argue, those of comparable work. On the longitudinal NeuroRex
dataset (EEG, head- and exoskeleton-mounted inertial units, nine sessions, seven
subjects) we first establish the severity of the confound: a movement-only classifier
using no EEG reaches $0.870$ balanced accuracy, above every neural decoder we tested,
including EEG~Conformer at $0.827$. We then introduce an \emph{invariance probe} that
measures how much of the inertial signal remains linearly recoverable from a decoder's
own representation, and show that a probe fitted across a distribution shift --- the
natural construction, and the one we originally used --- systematically mistakes drift
for invariance. Corrected to a session-grouped within-distribution estimator, the probe
reverses several of our earlier verdicts. Under the corrected probe we evaluate $131$
architecture variants at three seeds each: \textbf{none} yields a representation from
which movement is unrecoverable, accuracy and movement leakage correlate at
$r = +0.68$, and the highest-scoring configuration ($0.818 \pm 0.010$, comfortably above
the classical baseline) is also the leakiest ($R^2 = +0.322$). Between-architecture
spread ($\mathrm{sd} = 0.028$) is smaller than within-architecture seed spread
($0.024$--$0.071$), so six configurations clear the baseline on a single seed and only
two do so on a mean. A $1{,}831$-parameter classical pipeline --- covariance,
Euclidean Alignment, log-Euclidean tangent mapping, logistic regression --- attains
$0.763$ and is more invariant than any learned alternative, including adversarial
gradient reversal, HSIC, CORAL, orthogonality penalties and in-network motion
cancellation. We report this as a negative result rather than a contribution: the
winning pipeline is established practice, and our own safety-asymmetric selective head
achieves its error bound largely by declining to commit. The transferable contribution
is methodological --- an IMU-referenced leakage criterion that renders accuracy
rankings on movement-confounded paradigms interpretable, and evidence that architecture
search on this problem operates near a signal-to-noise ratio of one.
\end{abstract}"""

src = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda m: ABSTRACT,
             src, count=1, flags=re.S)

REEVAL = r"""
\section{Re-evaluation Under a Corrected Invariance Probe}\label{sec:reeval}

The results in Section~\ref{sec:valid} were obtained with an invariance probe that we
have since found to be misspecified. This section reports the correction, the
re-analysis it forces, and the conclusions that survive. We state plainly that the
revision removes our principal claim.

\subsection{The probe was measuring distribution shift}\label{sec:probe}

The probe estimates how much of the inertial vector $\mathbf{m}$ is recoverable from a
decoder's intent code $\mathbf{z}$; a non-positive $R^2$ was taken as evidence that
movement had been removed. Our original estimator fitted a ridge regression on the
training sessions and scored it on the test sessions. That construction conflates two
distinct properties. A representation whose features merely \emph{shift} between
sessions scores strongly negative while still encoding movement perfectly well within
any single session.

The corrected estimator, $\textsf{invariance\_r2\_cv}$, cross-validates the probe
\emph{inside} the evaluation set with session-grouped folds, so a positive value means
movement genuinely is recoverable. The two disagree materially, and in a direction that
flatters the original analysis:

\begin{center}
\begin{tabular}{lrr}
\toprule
representation & cross-split (original) & within-distribution (corrected) \\
\midrule
FBCSP      & $-0.501$ & $+0.187$ \\
tangent+EA & $-0.089$ & $-0.258$ \\
band-power & $+0.007$ & $+0.034$ \\
\bottomrule
\end{tabular}
\end{center}

FBCSP, which the original probe ranked as the \emph{most} invariant representation
tested, is in fact plainly leaky. Every $R^2$ we reported before this correction is
therefore suspect, including the module ablation of Section~\ref{sec:ablation} and the
architecture comparison. On CALM-Net~v2 the two probes give $-0.420$ and $-0.134$
respectively: the original overstates invariance by $0.29$ on a model from which the
correction was not derived.

\subsection{No architecture produces an invariant representation}

We re-ran the full variant space --- $131$ configurations spanning five backbone
families, seven readouts, fifteen augmentations, four independence penalties, four
optimisers and five frequency bands, from $7$k to $391$k parameters --- at three seeds
each, scoring every one with the corrected probe. Of the configurations completed at
submission, \textbf{none} attains $R^2 \le 0$; the least leaky reaches $+0.043$.
Accuracy and leakage correlate at $r = +0.68$.

\begin{center}
\begin{tabular}{lrrr}
\toprule
configuration & accuracy & $R^2_{\mathrm{cv}}$ & params \\
\midrule
\texttt{decorr\_0}        & $0.818 \pm 0.010$ & $+0.322$ & $7$k \\
\texttt{bb\_shallow\_wide}& $0.783 \pm 0.012$ & $+0.249$ & $391$k \\
\texttt{ro\_stat}         & $0.748 \pm 0.031$ & $+0.190$ & $31$k \\
\midrule
tangent+EA (classical)    & $0.763$           & $-0.265$ & $1{,}831$ \\
IMU only, no EEG          & $0.870$           & ---      & --- \\
\bottomrule
\end{tabular}
\end{center}

The best configuration beats the classical pipeline by $0.055$ and replicates tightly
across seeds. It is also the leakiest representation measured anywhere in this work. We
do not report it as an improvement, because the criterion that would license that
reading is the one it fails.

\subsection{The search operates near a signal-to-noise ratio of one}

Spread \emph{between} architectures ($\mathrm{sd} = 0.028$) is comparable to spread
\emph{within} a single architecture across seeds ($0.024$ mean; $0.071$ for
CALM-Net~v2), a ratio of $1.17$. Six configurations exceed the classical baseline on at
least one seed; only two do so on a three-seed mean. Our own earlier single-seed
leaderboard illustrates the hazard: its winner scored $0.782$ and replicated at
$0.712 \pm 0.057$. Mean accuracy across the space is unchanged by seed replication
($0.699 \rightarrow 0.699$), so replication does not depress performance in general ---
it removes apparent \emph{winners} specifically, which is the signature of selection on
noise rather than of measurement bias.

\subsection{Learned invariance loses to a closed-form transform}

Adversarial gradient reversal, HSIC, CORAL, orthogonality penalties and in-network
motion cancellation were each run against Euclidean Alignment on the identical task.
All four penalty families, at every strength tried, cluster near $0.68$ accuracy with
$R^2_{\mathrm{cv}} \approx +0.13$. Euclidean Alignment --- a label-free per-session
whitening by the mean covariance, computed in closed form --- attains $0.763$ at
$-0.265$. In-network motion cancellation failed outright: it removed $96.2\%$ of a
synthetic artefact, but on real data the network read the label \emph{from} the motion
reference, accuracy rising $0.806 \rightarrow 0.908$ as leakage rose
$+0.294 \rightarrow +0.582$.

We note this comparison is narrower than it may appear. That simple alignment is
competitive with sophisticated transfer learning is already the prevailing view for
subject and session shift. What we add is the case of a \emph{physically measured
nuisance correlated with the label itself}, where the confound carries the signal
rather than merely displacing it.

\subsection{The classical pipeline is not uniformly invariant either}

The aggregate $-0.265$ conceals a split population:

\begin{center}
\begin{tabular}{lrrrrrrr}
\toprule
subject & S1 & S2 & S3 & S4 & S5 & S6 & S7 \\
\midrule
accuracy            & $0.972$ & $0.675$ & $0.939$ & $0.794$ & $0.785$ & $0.574$ & $0.601$ \\
$R^2_{\mathrm{cv}}$ & $+0.471$& $-0.800$& $+0.420$& $-0.210$& $-0.897$& $-0.204$& $-0.638$ \\
\bottomrule
\end{tabular}
\end{center}

Movement is positively recoverable for two of seven subjects, and they are the two
highest-accuracy subjects; within the classical pipeline alone, accuracy and leakage
correlate at $r = +0.69$. The claim that this pipeline is movement-invariant holds on
average and fails precisely where its accuracy originates.

\subsection{The safety head does not deliver its guarantee}

Evaluating the three-term abstention rule whole, for the first time, at $83{,}694$
parameters and three seeds: balanced accuracy $0.628 \pm 0.052$, below the classical
$0.763$; achieved coverage $0.381$ against a $0.80$ target; walk recall $0.174$. The
wrong-walk bound holds ($0.040$ against $0.05$) largely because the system rarely
commits \emph{walk} at all --- the degenerate solution our own cost-weighting analysis
predicts, reached through the decision threshold instead of the loss. Decomposing the
gate, the conformal singleton term carries the rule (marginal cost $0.19$) while the
trained selective head and the wrong-walk bound reject almost nothing the other terms
would have kept ($0.06$ and $0.05$). The executed accuracy of $0.712$ cannot be
compared with a figure quoted at $80\%$ coverage.

\subsection{What survives}

Three findings survive re-evaluation. First, the confound is real and larger than we
reported: movement alone out-predicts every neural decoder here. Second, the invariance
probe, correctly specified, is a usable model-selection criterion, and under it no
architecture we tested is admissible. Third, closed-form second-order alignment
outperforms learned adversarial invariance against a physically measured nuisance.

What does not survive is the claim that CALM-Net decodes from provably
movement-invariant features. It does not; nothing we built does.
"""

src = src.replace(r"\section{Discussion}", REEVAL + "\n\\section{Discussion}", 1)
DST.write_text(src, encoding="utf-8")
print(f"wrote {DST} ({len(src.splitlines())} lines)")
