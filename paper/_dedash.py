import sys
f = "calmnet_paper_journal.tex"
s = open(f, encoding="utf-8").read()

pairs = [
    ("and that---unlike prior exoskeleton BMIs---exploits",
     "and that, unlike prior exoskeleton BMIs, exploits"),
    ("report and hardest to argue with---decoding accuracy.",
     "report and hardest to argue with: decoding accuracy."),
    ("Accuracy alone is an inadequate---and, we argue, a \\emph{misleading}---target when",
     "Accuracy alone is an inadequate (and, we argue, a \\emph{misleading}) target when"),
    ("exactly this regime---seven participants,",
     "exactly this regime: seven participants,"),
    ("\\textbf{C1 --- A multimodal neuro-kinematic architecture.}",
     "\\textbf{C1: A multimodal neuro-kinematic architecture.}"),
    ("\\textbf{C2 --- Motion-invariant disentanglement (MID).}",
     "\\textbf{C2: Motion-invariant disentanglement (MID).}"),
    ("resolves---rather than apologises for---the movement confound",
     "resolves, rather than apologises for, the movement confound"),
    ("\\textbf{C3 --- Cross-frequency coupling attention (XFCA).}",
     "\\textbf{C3: Cross-frequency coupling attention (XFCA).}"),
    ("\\textbf{C4 --- Longitudinal self-calibration with a distribution-free",
     "\\textbf{C4: Longitudinal self-calibration with a distribution-free"),
    ("\\textbf{C5 --- Safety-asymmetric selective decision.}",
     "\\textbf{C5: Safety-asymmetric selective decision.}"),
    ("fuses three independent signals---",
     "fuses three independent signals: "),
    ("and kinematic contamination---and",
     "and kinematic contamination, and"),
    ("the cross-session drop---universal deep adaptation primed on earlier",
     "the cross-session drop (universal deep adaptation primed on earlier"),
    ("dual-selection transfer~\\cite{luo2023}---but outputs",
     "dual-selection transfer~\\cite{luo2023}) but outputs"),
    ("Selective classification---the ``reject option''---is",
     "Selective classification, the ``reject option'', is"),
    ("\\emph{two} IMUs---head-worn and exoskeleton-mounted---each with",
     "\\emph{two} IMUs, head-worn and exoskeleton-mounted, each with"),
    ("near-chance for at least one---direct evidence that",
     "near-chance for at least one: direct evidence that"),
    ("coupling---co-modulation across bands",
     "coupling: co-modulation across bands"),
    ("movement-invariant neural features---by construction, not by",
     "movement-invariant neural features, by construction, not by"),
]

missing = [o for o, _ in pairs if o not in s]
if missing:
    print("NOT FOUND:")
    for m in missing:
        print("   ", repr(m))
    sys.exit(1)

for o, n in pairs:
    s = s.replace(o, n)

open(f, "w", encoding="utf-8").write(s)
# report remaining em dashes (--- ), ignoring en dashes (--)
import re
rem = len(re.findall(r"(?<!-)---(?!-)", s))
print(f"replaced {len(pairs)} blocks | remaining '---': {rem}")
