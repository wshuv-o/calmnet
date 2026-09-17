"""Fix the three Section VI tables that overflow the column and collide with text.

They were bare center+tabular blocks dropped inline, so nothing constrained them
to the column width. Each becomes a proper float: table* (both columns) where the
content is genuinely wide, table (one column) with shortened headers where it fits.
"""
from pathlib import Path

P = Path("calmnet_paper_conference.tex")
t = P.read_text(encoding="utf-8")

# --- 1. probe comparison: headers are the problem, not the data ------------- #
OLD1 = r"""\begin{center}
\begin{tabular}{lrr}
\toprule
representation & cross-split (original) & within-distribution (corrected) \\
\midrule
FBCSP      & $-0.501$ & $+0.187$ \\
tangent+EA & $-0.089$ & $-0.258$ \\
band-power & $+0.007$ & $+0.034$ \\
\bottomrule
\end{tabular}
\end{center}"""

NEW1 = r"""\begin{table}[t]
\caption{Movement recoverability $R^2$ under the original cross-split probe and
the corrected within-distribution probe. Negative values indicate movement is not
recoverable.}
\label{tab:probe}
\centering
\begin{tabular}{lrr}
\toprule
representation & cross-split & corrected \\
\midrule
FBCSP      & $-0.501$ & $+0.187$ \\
tangent+EA & $-0.089$ & $-0.258$ \\
band-power & $+0.007$ & $+0.034$ \\
\bottomrule
\end{tabular}
\end{table}"""

# --- 2. top configurations -------------------------------------------------- #
OLD2 = r"""\begin{center}
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
\end{center}"""

NEW2 = r"""\begin{table}[t]
\caption{Highest-accuracy configurations under the corrected probe, with the
classical and movement-only references. Accuracy is the mean over three seeds.}
\label{tab:topcfg}
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccr}
\toprule
configuration & accuracy & $R^2_{\mathrm{cv}}$ & params \\
\midrule
\texttt{decorr\_0}         & $0.818 \pm 0.010$ & $+0.322$ & $7$k \\
\texttt{bb\_shallow\_wide} & $0.783 \pm 0.012$ & $+0.249$ & $391$k \\
\texttt{ro\_stat}          & $0.748 \pm 0.031$ & $+0.190$ & $31$k \\
\midrule
tangent+EA                 & $0.763$           & $-0.265$ & $1{,}831$ \\
IMU only, no EEG           & $0.870$           & ---      & --- \\
\bottomrule
\end{tabular}
\end{table}"""

# --- 3. per-subject: 8 columns, needs the full text width ------------------- #
OLD3 = r"""\begin{center}
\begin{tabular}{lrrrrrrr}
\toprule
subject & S1 & S2 & S3 & S4 & S5 & S6 & S7 \\
\midrule
accuracy            & $0.972$ & $0.675$ & $0.939$ & $0.794$ & $0.785$ & $0.574$ & $0.601$ \\
$R^2_{\mathrm{cv}}$ & $+0.471$& $-0.800$& $+0.420$& $-0.210$& $-0.897$& $-0.204$& $-0.638$ \\
\bottomrule
\end{tabular}
\end{center}"""

NEW3 = r"""\begin{table*}[t]
\caption{Per-subject accuracy and movement recoverability for the classical
tangent+EA pipeline. Movement is positively recoverable for S1 and S3, which are
also the two highest-accuracy subjects.}
\label{tab:persubj}
\centering
\begin{tabular}{lrrrrrrr}
\toprule
 & S1 & S2 & S3 & S4 & S5 & S6 & S7 \\
\midrule
accuracy            & $0.972$ & $0.675$ & $0.939$ & $0.794$ & $0.785$ & $0.574$ & $0.601$ \\
$R^2_{\mathrm{cv}}$ & $+0.471$& $-0.800$& $+0.420$& $-0.210$& $-0.897$& $-0.204$& $-0.638$ \\
\bottomrule
\end{tabular}
\end{table*}"""

missing = []
for old, new in ((OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)):
    if old in t:
        t = t.replace(old, new, 1)
    else:
        missing.append(old.split("\n")[1][:50])

# the prose referred to these inline; give them float references
t = t.replace(
    "The two disagree materially, and in a direction that\nflatters the original analysis:",
    "The two disagree materially, and in a direction that\nflatters the original analysis (Table~\\ref{tab:probe}).", 1)
t = t.replace(
    "Accuracy and leakage correlate at $r = +0.68$.",
    "Accuracy and leakage correlate at $r = +0.68$ (Table~\\ref{tab:topcfg}).", 1)
t = t.replace(
    "The aggregate $-0.265$ conceals a split population:",
    "The aggregate $-0.265$ conceals a split population (Table~\\ref{tab:persubj}).", 1)

P.write_text(t, encoding="utf-8")
print(f"fixed {3 - len(missing)}/3 tables")
for m in missing:
    print("  NOT FOUND:", m)
