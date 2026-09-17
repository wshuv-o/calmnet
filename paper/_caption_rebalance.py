"""Rebalance captions to normal IEEE length (~15-30 words).

The first pass cut them to 7-11 words, which is terser than convention and makes
the exhibits unreadable out of sequence. A caption should be self-contained: what
the exhibit shows and what to look at, without duplicating the analysis.
"""
from pathlib import Path

P = Path("calmnet_paper_conference.tex")
t = P.read_text(encoding="utf-8")

CAPS = [
    (r"\caption{The CALM-Net framework: encoders, MID, LSC, and the SAS head.}",
     r"""\caption{The CALM-Net framework. Multimodal encoders map EEG sub-bands, IMU, and
EOG into a shared space; MID splits it into intent and artefact subspaces; LSC
aligns per-session statistics and runs adaptive conformal control; the SAS head
commits Walk/Stop only on unanimous agreement, else STOP.}"""),

    (r"\caption{MID across seven subjects: accuracy (left) and movement recoverability (right).}",
     r"""\caption{Motion-invariant disentanglement across seven subjects. \textbf{Left:}
balanced accuracy of the IMU-only baseline, the decoder without MID, and with MID.
\textbf{Right:} recoverability of the IMU descriptor from the intent code, under the
original cross-split probe; see Section~\ref{sec:reeval} for the corrected estimate.}"""),

    (r"\caption{Full CALM-Net per subject, mean over six held-out sessions.}",
     r"""\caption{Full CALM-Net per subject, mean over the six held-out test sessions with
training on sessions 1--3. Column definitions are given in the text; conformal
coverage targets $0.90$.}"""),

    (r"\caption{Conformal coverage (left) and calibration error (right) across the session gap.}",
     r"""\caption{Full CALM-Net across seven subjects. \textbf{Left:} adaptive conformal
inference holds coverage at the $0.90$ target under cross-session drift, where static
conformal misses it. \textbf{Right:} temperature scaling lowers calibration error
across the session gap.}"""),

    (r"\caption{CALM-Net versus baseline decoders, mean over seven subjects.}",
     r"""\caption{CALM-Net versus baseline decoders, mean over seven subjects,
longitudinal. Baseline accuracy is movement-inflated: it sits at the IMU-only
baseline of $0.84$, reachable with no EEG at all.}"""),

    (r"\caption{Encoder ablation under an identical downstream pipeline.}",
     r"""\caption{Encoder ablation under the same MID, calibration, and adaptive-conformal
pipeline, mean over seven subjects. The Riemannian--XFCA encoder is more accurate but
more movement-coupled; the band-power encoder disentangles more cleanly.}"""),
]

missing = []
for old, new in CAPS:
    if old in t:
        t = t.replace(old, new, 1)
    else:
        missing.append(old[:60])

P.write_text(t, encoding="utf-8")
print(f"rebalanced {len(CAPS) - len(missing)}/{len(CAPS)}")
for m in missing:
    print("  NOT FOUND:", m)
