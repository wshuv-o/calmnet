"""Derive the extended paper (adds implementation notes, baseline comparison,
Discussion, Limitations, Conclusion) from the journal version, kept as a separate file."""

s = open("calmnet_paper_journal.tex", encoding="utf-8").read()

# 1) mark it as the extended version via an author footnote
s = s.replace(
    "a CC0 licence.}}",
    "a CC0 licence.}\n\\thanks{Extended version: adds a baseline comparison, "
    "implementation notes, discussion, and limitations.}}", 1)

# 2) implementation-notes subsection at the end of the Method section
impl = r"""\subsection{Implementation notes}
The experiments below validate the framework's decisive components: motion-invariant
disentanglement, the calibration layer, the adaptive-conformal guarantee, and the
selective decision. The encoder used here is a compact \emph{band-power}
instantiation of the multi-band design (learnable temporal filters, grouped spatial
filters, log-variance pooling, and temporal attention). We additionally implement the
full design of Section~\ref{sec:method}, the multi-band Riemannian tangent-space
encoder with cross-frequency coupling attention (XFCA) and source-free CORAL
alignment, and compare the two in Section~\ref{sec:ablation}
(Table~\ref{tab:enc}). The disentanglement target $\phi(M)$ is the $12$-dimensional head- and
exoskeleton-IMU descriptor (accelerometer and gyroscope magnitude, mean/std/max), and
the reported abstention combines calibrated confidence with the adaptive-conformal
set; the kinematic-contamination gate is implemented but not separately ablated here.

\section{Preliminary Results}\label{sec:valid}"""
s = s.replace(r"\section{Preliminary Results}\label{sec:valid}", impl, 1)

# 3) comparison table + Discussion + Conclusion, inserted before the bibliography
add = r"""\begin{table}[t]
\caption{CALM-Net versus baseline decoders (mean over the seven subjects,
longitudinal). Baseline accuracy is \emph{movement-inflated}: it sits at the IMU-only
movement baseline ($0.84$), attainable from a single motion feature with no EEG.
CALM-Net's accuracy is movement-invariant, and it is the only method here that also
provides a distribution-free coverage guarantee.}
\label{tab:cmp}
\centering
\begin{tabular}{lccccc}
\toprule
Method & Mov.-inv. & Acc & ECE & AUROC & Cov. guar. \\
\midrule
IMU-only (movement) & -- & 0.839 & -- & -- & -- \\
EEGNet~\cite{lawhern2018} & no & 0.819 & 0.086 & 0.813 & no \\
EEG Conformer~\cite{song2023} & no & 0.826 & 0.067 & 0.845 & no \\
\textbf{CALM-Net (ours)} & \textbf{yes} & \textbf{0.694} & 0.114 & 0.698 & \textbf{yes} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Encoder ablation}\label{sec:ablation}
Table~\ref{tab:enc} compares the two encoders under the identical MID, calibration,
and adaptive-conformal pipeline. The Riemannian--XFCA encoder is the stronger
\emph{decoder}: mean balanced accuracy $0.805$ versus $0.694$ and executed accuracy
$0.835$ versus $0.719$ at $80\%$ coverage. However, its accuracy is more
\emph{movement-coupled}: disentanglement reduces the intent-to-IMU $R^2$ less and
less consistently, still failing to disentangle the two highest-accuracy subjects
whose covariance features remain movement-predictive, and its calibration is worse on
two subjects. The band-power model, though less accurate, disentangles more cleanly
and yields the more trustworthy movement-invariant estimate. The adaptive-conformal
coverage guarantee is encoder-agnostic, holding the $0.90$ target for both.
Strengthening the disentanglement with a nonlinear Hilbert--Schmidt independence
(HSIC) penalty and a disentanglement-aware model selection drives the
Riemannian--XFCA invariance further (mean intent-to-IMU $R^2=-0.47$) but lowers its
accuracy to $0.65$, revealing that its apparent accuracy advantage was movement
leakage and that the ${\sim}0.7$ movement-invariant decode is a \emph{ceiling robust
to encoder richness and disentanglement strength}. Raising it therefore requires
attacking the confound at the source, for example decoding the pre-movement
preparation window or regressing the inertial signals out of the EEG, rather than a
stronger disentanglement penalty. We report the band-power model for the
movement-invariant claim and the Riemannian--XFCA encoder as the higher-accuracy but
movement-coupled alternative.

\begin{table}[t]
\caption{Encoder ablation under the same MID, calibration, and adaptive-conformal
pipeline (mean over seven subjects). The Riemannian--XFCA encoder is more accurate,
but its accuracy is more movement-coupled (a smaller, less consistent invariance
gain); the band-power encoder disentangles more cleanly. Coverage holds for both.}
\label{tab:enc}
\centering
\begin{tabular}{lcccccc}
\toprule
Encoder & Acc & ECE & AUROC & exec@80 & cov$_{\text{adpt}}$ & $R^2$: noMID$\rightarrow$MID \\
\midrule
Band-power (primary) & 0.694 & 0.114 & 0.698 & 0.719 & 0.902 & $0.15\rightarrow-0.31$ \\
Riemannian--XFCA & 0.805 & 0.137 & 0.741 & 0.835 & 0.900 & $0.13\rightarrow-0.24$ \\
\bottomrule
\end{tabular}
\end{table}

\section{Discussion}
\textbf{What CALM-Net buys.} On this paradigm, raw accuracy is not a measure of neural
decoding. A single head-motion feature reaches $0.84$ (Table~\ref{tab:cmp}), and the
EEGNet and EEG~Conformer baselines sit at that movement ceiling; for two subjects a
motion sensor alone out-predicts the $60$-channel network. CALM-Net trades that
inflated number for one that means what it says: a movement-invariant decode of
$0.69$, calibrated confidence, and prediction sets whose coverage provably tracks its
target as the recording drifts over weeks. For a device that moves a person's legs, an
honest $0.69$ that abstains when unsure is safer than a movement-inflated $0.83$ that
cannot decline. This reframes the field's usual objective: the contribution is not a
higher number but a trustworthy one.

\textbf{Controlling the dangerous error.} The safety asymmetry is operationalised
with a class-conditional threshold calibrated to bound the committed \emph{wrong-walk}
rate. Across the seven subjects this cuts the false-walk rate from $0.144$ (arg-max)
to $0.051$ at a $0.05$ target, on every subject, routing the remaining walk-uncertain
windows to the safe stop (mean walk recall $0.37$); where the decode is at chance the
rule correctly almost never commits walk. A $K{=}3$ deep ensemble further sharpens the
confidence used for this decision (mean AUROC $0.70\rightarrow0.74$, executed accuracy
$0.72\rightarrow0.77$ at $80\%$ coverage). We note honestly that the
already-well-calibrated band-power model's ECE is \emph{not} improved by ensembling or
short per-session recalibration on this data, so we retain the single global
temperature; the gains are in confidence ranking and, above all, in bounding the
dangerous error.

\textbf{Why disentanglement, not exclusion.} Because the exoskeleton moves the wearer
during walking, no clean movement-free imagery contrast exists to train on. We
verified this empirically: decoding the Stop-to-Walk \emph{transition onset}, before
head motion builds up, does not lower the movement baseline (a motion sensor still
separates the classes at $0.94$), because the exoskeleton commits to motion at the
command instant and the walk/stop label is its state. There is no movement-free
window to escape to. Rather than discard the movement-coupled data, CALM-Net uses the
synchronised inertial signals as the supervisory variable to render the neural code
invariant to them. That
the movement-invariant decode stays above chance for every participant, and that
subject~6 (whose movement is uninformative) is decoded well, shows a recoverable
neural signal underlies the movement in each subject.

\textbf{Limitations.} (i) The cohort is seven healthy adults on a single dataset;
generalisation to spinal-cord-injury or stroke patients, whose cortical signals and
movement coupling differ, is untested. (ii) Evaluation is offline replay, not a live
closed loop; the adaptive-conformal update assumes outcome feedback that a deployed
system must approximate. (iii) The movement-invariant accuracy is a conservative lower
bound: MID can over-regularise when movement is already uninformative (subject~6), so
true neural decodability may be higher. (iv) The full Riemannian--XFCA encoder is
implemented and compared (Table~\ref{tab:enc}): it decodes more accurately but its
accuracy is more movement-coupled, so the band-power model is used for the invariance
claim; a disentanglement objective matched to the richer encoder is future work.
(v) Subject~2's decode
is at chance, correctly flagged by an uninformative confidence, but confirms the
method cannot manufacture signal where little exists.

\section{Conclusion}
We presented CALM-Net, a multimodal framework that decodes movement-invariant neural
intent, remains calibrated across weeks, and abstains to a safe stop under a
distribution-free guarantee. Treating the exoskeleton's inertial signals as a modality
to disentangle against, rather than a confound to hide, converts the paradigm's central
difficulty into its contribution: to our knowledge the first movement-invariant estimate
of neural walk/stop decodability on the NeuroRex dataset, together with an
adaptive-conformal coverage guarantee that holds on all seven subjects. Future work
implements and ablates the full Riemannian/attention encoder, validates on patient
populations, and closes the loop online.

\begin{thebibliography}{00}"""
s = s.replace(r"\begin{thebibliography}{00}", add, 1)

open("calmnet_paper_extended.tex", "w", encoding="utf-8").write(s)
ok = ("Implementation notes" in s and "tab:cmp" in s and "\\section{Discussion}" in s
      and "\\section{Conclusion}" in s and "---" not in s)
print("extended tex written | all sections + no em dash:", ok)
