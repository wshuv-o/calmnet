"""Build the final manuscript presentation from the compiled paper.

The proposal deck supplies the templates: every slide is a copy of one of its
slides, keeping shapes, fonts, sizes, colours and geometry, with only text and
images replaced. Tables are screenshots cropped from the compiled manuscript by
crop_paper_assets.py, so the deck shows the paper's own rendering rather than a
retyped copy. Figures are the paper's own figure files, and the architecture is
the hand-maintained drawio export.

    python presentation/crop_paper_assets.py
    python presentation/build_final_deck.py
"""

import copy
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
ASSETS = HERE / "paper_assets"
SRC = HERE / "CALM-Net_midterm_proposal.pptx"
import sys
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else "CALM-Net_final_presentation.pptx")

BLK = RGBColor(0x11, 0x11, 0x11)
GRY = RGBColor(0x55, 0x55, 0x55)
FONT = "Times New Roman"
MINUS = "−"

prs = Presentation(str(SRC))
N_ORIG = len(prs.slides)
T = {i + 1: s for i, s in enumerate(prs.slides)}


# ------------------------------------------------------------------ helpers
def clone(ti, skip=()):
    """New slide copying template slide ti, without pictures and tables."""
    src = T[ti]
    new = prs.slides.add_slide(src.slide_layout)
    for shp in list(new.shapes):
        shp._element.getparent().remove(shp._element)
    for shp in src.shapes:
        if shp.shape_type in (13, 19) or shp.name in skip:
            continue
        new.shapes._spTree.insert_element_before(copy.deepcopy(shp._element), "p:extLst")
    return new


def borrow(s, ti, name, left=None, top=None, width=None, height=None):
    """Copy one shape from template slide ti into slide s."""
    src = next(sh for sh in T[ti].shapes if sh.name == name)
    el = copy.deepcopy(src._element)
    s.shapes._spTree.insert_element_before(el, "p:extLst")
    sh = s.shapes[-1]
    for attr, v in (("left", left), ("top", top), ("width", width), ("height", height)):
        if v is not None:
            setattr(sh, attr, Inches(v))
    return sh


def get(s, name):
    return next(sh for sh in s.shapes if sh.name == name)


def place(sh, left=None, top=None, width=None, height=None):
    for attr, v in (("left", left), ("top", top), ("width", width), ("height", height)):
        if v is not None:
            setattr(sh, attr, Inches(v))


def R(text, size, bold=False, italic=False, color=BLK):
    return (text, size, bold, italic, color)


def P(*runs, align=PP_ALIGN.LEFT, sa=8, ls=1.1):
    return dict(runs=list(runs), align=align, sa=sa, ls=ls)


def write(shape, paras):
    """Replace a shape's text, keeping the shape and its frame settings."""
    tf = shape.text_frame
    body = tf._txBody
    for p in body.findall(qn("a:p"))[1:]:
        body.remove(p)
    first = tf.paragraphs[0]
    for child in list(first._p):
        if child.tag in (qn("a:r"), qn("a:br"), qn("a:fld")):
            first._p.remove(child)
    for i, spec in enumerate(paras):
        p = first if i == 0 else tf.add_paragraph()
        p.alignment = spec["align"]
        p.space_after = Pt(spec["sa"])
        p.line_spacing = spec["ls"]
        for text, size, bold, italic, color in spec["runs"]:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
            f.color.rgb = color


def title(s, text):
    write(get(s, "TextBox 2"), [P(R(text, 30, bold=True), sa=8, ls=1.0)])


def picture(s, path, left, top, width, height):
    """Add an image fitted inside the box, centred, aspect kept."""
    w, h = Image.open(path).size
    scale = min(width / w, height / h)
    pw, ph = w * scale, h * scale
    return s.shapes.add_picture(str(path), Inches(left + (width - pw) / 2),
                                Inches(top + (height - ph) / 2), Inches(pw), Inches(ph))


def lead(bold, rest, size):
    return P(R(bold, size, bold=True), R(rest, size, color=GRY), sa=8, ls=1.2)

# ------------------------------------------------------------------ 1 title
s = clone(1)
write(get(s, "TextBox 4"),
      [P(R("Tangent-space features for EEG movement-intent decoding", 24),
         align=PP_ALIGN.CENTER, sa=8, ls=1.15),
       P(R("Final manuscript presentation", 18, color=GRY),
         align=PP_ALIGN.CENTER, sa=8, ls=1.15)])
write(get(s, "TextBox 5"),
      [P(R("Md Wahiduzzaman Suva  [26-94088-2]", 15, color=GRY), align=PP_ALIGN.CENTER, sa=5),
       P(R("Esme Moula Chowdhury Abha  [26-94089-2]", 15, color=GRY), align=PP_ALIGN.CENTER, sa=5),
       P(R("Supervisor: Dr. Muhammad Hasibur Rashid Chayon", 15, color=GRY), align=PP_ALIGN.CENTER, sa=8)])

# ------------------------------------------------------------------ 2 problem
s = clone(2)
title(s, "Problem Statement")
write(get(s, "TextBox 4"), [
    P(R("A convolutional EEG decoder reduces each window to log band power, which is "
        "the diagonal of its spatial covariance. Everything off the diagonal, how the "
        "channels covary, is discarded before the first layer sees it.", 19, color=GRY),
      sa=12, ls=1.2),
    P(R("The question this paper answers is whether that discarded structure carries "
        "usable movement-intent information, and what it costs to read it.", 19, color=GRY),
      sa=8, ls=1.2)])
write(get(s, "Rectangle 5"), [
    P(R("51 participants, five cohorts, one training pipeline.", 18, bold=True), sa=6, ls=1.15),
    P(R("Two lower-limb exoskeleton cohorts, a treadmill cohort, a motor-execution "
        "cohort and BCI Competition IV-2a.", 16, color=GRY), sa=8, ls=1.18)])
write(get(s, "TextBox 6"), [
    P(R("The answer is not uniform. ", 18, bold=True),
      R("The structure helps on four cohorts and fails on the fifth, and the recording "
        "protocol says which case applies before any model is trained.", 18, color=GRY),
      sa=8, ls=1.18)])

# ------------------------------------------------------------------ 3 challenges
s = clone(3)
title(s, "Key Challenges")
for name, num, head, body in (
        ("Rectangle 4", "1", "Second-order structure is thrown away",
         "A log-power stem keeps the diagonal of the covariance and discards the rest. "
         "Whether that loses anything is an empirical question, not a settled one."),
        ("Rectangle 5", "2", "A reference that adapts can adapt wrongly",
         "Reading the off-diagonal needs a reference covariance. Updating it on the test "
         "stream tracks the session, but it also tracks whichever class is streaming."),
        ("Rectangle 6", "3", "Failure has to be predictable before training",
         "A component that helps on four cohorts and fails on the fifth is only useful "
         "if the protocol says in advance which case applies.")):
    write(get(s, name), [P(R(num, 30, bold=True), sa=6, ls=1.1),
                         P(R(head, 19, bold=True), sa=10, ls=1.05),
                         P(R(body, 15, color=GRY), sa=8, ls=1.22)])

# ------------------------------------------------------------------ 4 objectives
s = clone(4)
title(s, "Objectives and Outcomes")
for name, head, body in (
        ("TextBox 5", "Read the covariance, not just its diagonal",
         "Outcome: a tangent-space branch beside the log-power stem. It raises balanced "
         "accuracy on four of five cohorts, by +0.064 on the held-out cohort C where all "
         "20 of 20 participants improve (p = 8.8e-5)."),
        ("TextBox 7", "State when the adapting reference is admissible",
         "Outcome: Condition 1. The estimator memory must outlast a class block and stay "
         "below the drift timescale. Read from the protocol, it separates all five cohorts "
         "with nothing fitted."),
        ("TextBox 9", "Report what the component costs",
         "Outcome: the branch adds 238,028 parameters to a 19,505-parameter stem path, and "
         "on cohort A it triples the spurious-activation rate. Both are reported, not hidden.")):
    write(get(s, name), [P(R(head, 20, bold=True), sa=2, ls=1.1),
                         P(R(body, 15, color=GRY), sa=8, ls=1.12)])
write(get(s, "TextBox 10"),
      [P(R("Question answered:  ", 17, bold=True),
         R("does the covariance structure a log-power stem discards carry movement intent, "
           "and when can a test-time reference be trusted to read it?", 17, italic=True, color=GRY))])

# ------------------------------------------------------------------ 5 literature
s = clone(5)
boxes = {
    "Rectangle 4": ("Lower-limb exoskeleton BCI",
                    ["NeuroRex dataset  [Sarkar 2026]", "Deep-learning control  [Ferrero 2024]",
                     "Exo error potentials  [Soriano 2025]", "Robot-assisted gait  [Tortora 2023]"]),
    "Rectangle 5": ("EEG decoders",
                    ["EEGNet  [Lawhern 2018]", "ShallowFBCSPNet  [Schirrmeister 2017]",
                     "EEG-Conformer  [Song 2023]"]),
    "Rectangle 6": ("Riemannian and alignment",
                    ["Riemannian geom.  [Barachant 2012]", "Euclidean alignment  [He 2020]",
                     "Riemannian Procrustes  [Rodrigues 2019]"]),
    "Rectangle 7": ("Test-time adaptation",
                    ["Tent  [Wang 2021]", "Correlated streams  [Gong 2022]",
                     "Reject option  [Geifman 2019]"]),
}
for name, (head, items) in boxes.items():
    write(get(s, name), [P(R(head, 15, bold=True), sa=9, ls=1.05)] +
          [P(R(it, 12, color=GRY), sa=7, ls=1.1) for it in items])
write(get(s, "Rectangle 8"),
      [P(R("RESEARCH GAP", 14, bold=True), sa=4, ls=1.1),
       P(R("Riemannian methods estimate their reference over a complete recording, so the "
           "recording must exist before the decoder can run. Test-time adaptation removes "
           "that constraint but degrades on temporally correlated streams. No prior work "
           "states, from the protocol alone, when an in-network adapting reference is safe.",
           18), sa=8, ls=1.14)])
write(get(s, "TextBox 9"),
      [P(R("Citations shown as [first author, year]; full entries are on the References slide.",
           10.5, italic=True, color=GRY))])

# ------------------------------------------------------------------ 6 architecture
s = clone(6)
title(s, "Decoder Architecture")
picture(s, ASSETS / "fig_arch_paper.png", 0.87, 1.55, 11.60, 4.55)
write(get(s, "TextBox 5"),
      [P(R("Reported model, 290,943 parameters:  ", 16, bold=True),
         R("a multi-scale log-power stem reads the diagonal of each window's covariance, a "
           "tangent-space branch reads the rest of it at a running reference M, and the two "
           "are concatenated into one linear classifier. The reference updates without "
           "labels and stays active at inference.", 16, color=GRY), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 7 core idea
s = clone(7)
title(s, "Core Idea and Hypothesis")
bul = [("•  Read the off-diagonal:",
        " each window's covariance C is shrunk and mapped to the tangent space at a reference M, "
        "giving vec(log(M⁻¹ᐟ² C M⁻¹ᐟ²))."),
       ("•  The reference adapts:",
        " M is a running mean over incoming windows, label-free, still updating at test time."),
       ("•  Estimator memory:",
        " the momentum m sets a memory of μ = N/m windows, 160 at the default N = 32, m = 0.2."),
       ("•  The failure mode:",
        " when one class streams for longer than μ, M becomes that class and whitening removes "
        "the very variance that separates the classes.")]
write(get(s, "TextBox 4"),
      [P(R(b, 18, bold=True), R(t, 18, color=GRY), sa=12, ls=1.18) for b, t in bul])
write(get(s, "Rectangle 5"),
      [P(R("CONDITION 1   ", 14, bold=True),
         R("τ_blk  <  μ  <  τ_drift.  The branch is admissible only where the estimator "
           "memory outlasts a class block and stays below the session-drift timescale. Both "
           "sides are fixed before training.", 17, italic=True), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 8 five cohorts
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Datasets and Evaluation Protocol")
place(get(s, "TextBox 4"), left=0.70, top=1.75, width=5.60, height=4.60)
picture(s, ASSETS / "tab_blocks.png", 6.50, 1.90, 6.25, 4.30)
write(get(s, "TextBox 4"), [
    P(R("A  exoskeleton", 17, bold=True), R("   7 participants, 9 sessions over weeks", 15, color=GRY), sa=9, ls=1.15),
    P(R("B  treadmill", 17, bold=True), R("   8 participants, 3 trials, independent lab", 15, color=GRY), sa=9, ls=1.15),
    P(R("C  motor execution", 17, bold=True), R("   20 participants, held out from every design decision", 15, color=GRY), sa=9, ls=1.15),
    P(R("D  BCI IV-2a", 17, bold=True), R("   9 participants, two sessions on different days", 15, color=GRY), sa=9, ls=1.15),
    P(R("E  exoskeleton", 17, bold=True), R("   7 participants, DECODED / EUROBENCH", 15, color=GRY), sa=9, ls=1.15),
    P(R("Block length τ_blk is read from the label sequence alone, before training: "
        "A 18, B 214, C 5, D 1, E 13 windows.", 15, color=GRY), sa=10, ls=1.2)])

# ------------------------------------------------------------------ 9 what is decoded
s = clone(8)
title(s, "Artefact Control")
picture(s, ASSETS / "tab_artefact.png", 0.55, 2.15, 6.30, 4.00)
write(get(s, "TextBox 5"), [
    P(R("Walk and stop differ in movement, so the accuracy could be artefact.", 17, bold=True), sa=12, ls=1.15),
    P(R("Retraining the reported model after ICA cleaning [Pion-Tonachini 2019]: ", 16, color=GRY),
      R("0.897 to 0.859", 16, bold=True),
      R(" (p = 0.58, four of seven unchanged or better). The mean is carried by one "
        "participant; without sub-06 it is −0.010.", 16, color=GRY), sa=11, ls=1.18),
    P(R("The change does not follow the muscle components removed (ρ = +0.05), where "
        "dependence on muscle artefact would be negative.", 16, color=GRY), sa=11, ls=1.18),
    P(R("Constrains the artefact account. It does not exclude it.", 17, bold=True), sa=8, ls=1.18)])

# ------------------------------------------------------------------ 10 topography
s = clone(6)
title(s, "Scalp Topography of the Reference")
picture(s, RES / "fig_topography.png", 0.87, 1.60, 11.60, 4.40)
write(get(s, "TextBox 5"),
      [P(R("The branch's own reference:  ", 16, bold=True),
         R("it amplifies the frontocentral midline most (AFz, Fz at 2.9) and the temporal "
           "sites most exposed to jaw and neck muscle least (T7, T8 at 1.9). Band power falls "
           "over left sensorimotor cortex during walking (C3 −3.5 dB), the sign of mu and beta "
           "desynchronisation; gross artefact would raise it.", 16, color=GRY), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 11 activations
s = clone(6)
title(s, "Learned Representations")
picture(s, RES / "fig_activations.png", 0.87, 1.55, 11.60, 4.45)
write(get(s, "TextBox 5"),
      [P(R("Measured, not drawn:  ", 16, bold=True),
         R("forward hooks on a trained model, on held-out windows. The tangent map carries "
           "structure off the diagonal, which the stem discards. Class separation grows along "
           "both paths and is largest after fusion (2.41, against 1.22 for the branch and 0.74 "
           "for the stem), so the two are complementary in the model's own activations.",
           16, color=GRY), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 12 comparison
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Comparison with Published Decoders")
place(get(s, "TextBox 4"), left=0.70, top=1.48, width=12.00, height=1.00)
picture(s, ASSETS / "tab_compare.png", 0.55, 2.70, 12.20, 3.55)
write(get(s, "TextBox 4"),
      [P(R("One pipeline, shared splits, seeds and metric code, no per-model tuning. ", 17, bold=True),
         R("On cohort C the reported model reaches 0.843 against 0.783 for the strongest "
           "published decoder, and the stem it is built on reaches 0.779.", 17, color=GRY),
         sa=8, ls=1.18)])

# ------------------------------------------------------------------ 13 branch vs stem
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Branch versus Stem Across Cohorts")
place(get(s, "TextBox 4"), left=0.70, top=1.48, width=12.00, height=1.00)
picture(s, ASSETS / "tab_allcoh.png", 0.70, 2.80, 11.90, 3.20)
write(get(s, "TextBox 4"),
      [P(R("Four cohorts gain, one fails. ", 17, bold=True),
         R("The branch adds +0.045 on A, +0.064 on C, +0.040 on D and +0.021 on E, and costs "
           "0.175 on B. Cohort B is the one whose class blocks (214 windows) outlast the "
           "estimator memory (160), which Condition 1 names in advance.", 17, color=GRY),
         sa=8, ls=1.18)])

# ------------------------------------------------------------------ 14 cohort C
s = clone(6)
title(s, "Results on the Held-Out Cohort")
picture(s, RES / "fig_cohortC.png", 0.87, 1.65, 11.60, 4.30)
write(get(s, "TextBox 5"),
      [P(R("Consistent, above the noise, and ahead of the field.  ", 16, bold=True),
         R("(a) all 20 participants improve, mean +0.064 (p = 8.8 × 10⁻⁵). (b) the change "
           "exceeds that participant's own spread across seeds in 16 of the 20. (c) it leads "
           "the strongest published decoder on this cohort by +0.060, higher in 18 of 20 "
           "(p = 1.3 × 10⁻⁴). Calibration error falls from 0.075 to 0.062 on the same runs.",
           16, color=GRY), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 15 every cohort
s = clone(6)
title(s, "Effect Size Against Class-Block Length")
picture(s, RES / "fig_cohorts.png", 0.87, 1.65, 11.60, 4.30)
write(get(s, "TextBox 5"),
      [P(R("Nothing is fitted here:  ", 16, bold=True),
         R("block length comes from the protocol and the memory from the optimiser settings. "
           "Plotting the effect against τ_blk separates the four cohorts that gain from the "
           "one that fails, at the memory boundary μ = 160.", 16, color=GRY), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 16 rate by manipulation
s = clone(6)
title(s, "Direct Manipulation of the Estimator Memory")
picture(s, RES / "fig_rate.png", 0.87, 1.65, 11.60, 4.30)
write(get(s, "TextBox 5"),
      [P(R("One cohort, one variable:  ", 16, bold=True),
         R("moving the estimator memory across the block length switches the effect off and "
           "on. On cohort A it is +0.045 at μ = 160, +0.048 at 40 and −0.076 at 8, worse in "
           "every participant. Enlarging μ on the failing cohort recovers 74 % of the loss.",
           16, color=GRY), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 17 rate table
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Estimator Memory and Block Length: Summary")
place(get(s, "TextBox 4"), left=0.70, top=1.75, width=5.55, height=4.60)
picture(s, ASSETS / "tab_rate.png", 6.45, 1.60, 6.25, 4.70)
write(get(s, "TextBox 4"), [
    P(R("The condition gives an ordering and a direction.", 18, bold=True), sa=12, ls=1.15),
    P(R("It does not give a rate a practitioner could set. A stronger prescription, "
        "setting the rate from the block length with a margin of two, was pre-registered "
        "and falsified: at the memory it prescribes cohort B reaches 0.650, against 0.744 "
        "for the stem alone.", 16, color=GRY), sa=12, ls=1.18),
    P(R("The upper bound of the condition is not measured here.", 16, color=GRY), sa=8, ls=1.18)])

# ------------------------------------------------------------------ 18 ablation
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Ablation Study")
place(get(s, "TextBox 4"), left=0.70, top=1.48, width=12.00, height=1.00)
picture(s, ASSETS / "tab_ablation.png", 0.70, 2.70, 11.90, 2.15)
picture(s, ASSETS / "tab_tanref.png", 0.70, 5.10, 11.90, 1.30)
write(get(s, "TextBox 4"),
      [P(R("Every row is a choice the architecture makes. ", 17, bold=True),
         R("The branch raises accuracy and cuts the seed spread about nine-fold. Freezing "
           "the reference after fitting or removing it entirely both cost accuracy, so the "
           "adapting reference is where the gain comes from.", 17, color=GRY), sa=8, ls=1.18)])

# ------------------------------------------------------------------ 19 noise
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Measurement Noise and Ranking Criteria")
place(get(s, "TextBox 4"), left=0.70, top=1.75, width=5.55, height=4.60)
picture(s, ASSETS / "tab_noise.png", 6.55, 1.55, 6.15, 4.75)
write(get(s, "TextBox 4"), [
    P(R("What can be ranked, and what cannot.", 18, bold=True), sa=12, ls=1.15),
    P(R("Across three data-split seeds the reported model spans 0.007 to 0.022 depending "
        "on the cohort, and the stem spans 0.011 to 0.057. The stem is the noisier arm on "
        "all five cohorts, so the branch does not buy its accuracy with a less stable fit.",
        16, color=GRY), sa=12, ls=1.18),
    P(R("Differences below 0.02 are not interpreted anywhere in this work.", 16, bold=True),
      sa=8, ls=1.18)])

# ------------------------------------------------------------------ 20 deployment
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Control-Relevant Outcome Measures")
place(get(s, "TextBox 4"), left=0.70, top=1.75, width=5.55, height=4.60)
picture(s, ASSETS / "tab_deploy.png", 6.55, 1.55, 6.15, 4.75)
write(get(s, "TextBox 4"), [
    P(R("Balanced accuracy is not what a wearer experiences.", 18, bold=True), sa=12, ls=1.15),
    P(R("Calibration improves on four of five cohorts and the missed-onset rate falls on "
        "four, so the accuracy is not bought with a worse-behaved posterior.", 16, color=GRY),
      sa=12, ls=1.18),
    P(R("The exception is cohort A, where spurious walk commands rise from 0.70 to 2.09 "
        "per minute of standing. For a device that moves a wearer's legs that is a real "
        "cost, and it is reported rather than the accuracy alone.", 16, color=GRY), sa=8, ls=1.18)])

# ------------------------------------------------------------------ 21 new cohorts
s = clone(9, skip=("Table 5", "Rectangle 6", "Rectangle 7", "Rectangle 8",
                   "TextBox 9", "TextBox 10"))
title(s, "Results on Cohorts D and E")
place(get(s, "TextBox 4"), left=0.70, top=1.75, width=5.55, height=4.60)
picture(s, ASSETS / "tab_newbase.png", 6.55, 1.55, 6.15, 4.75)
write(get(s, "TextBox 4"), [
    P(R("The branch leads every published decoder on both.", 18, bold=True), sa=12, ls=1.15),
    P(R("0.741 against 0.682 for the strongest on cohort D, and 0.949 against 0.896 on "
        "cohort E. The stem alone also clears them, so the ordering does not depend on "
        "the branch; what the branch adds is the margin.", 16, color=GRY), sa=12, ls=1.18),
    P(R("Cohort D needs a caveat the margin hides: five of its eight decoders are left "
        "near chance on at least one seed, which its ~100 fitting windows per participant "
        "explain. Only the three that trained are load-bearing.", 15, color=GRY), sa=8, ls=1.18)])

# ------------------------------------------------------------------ 22 limitations
s = clone(7)
title(s, "Limitations")
bul = [("•  The boundary was found after the fact:",
        " it separates all five cohorts with nothing fitted, but was identified after A, B and C "
        "had been run. Only D and E were prospective."),
       ("•  No artefact rejection for this configuration:",
        " the ICA control constrains the artefact account on seven participants and does not "
        "exclude it. A power contrast is not an artefact rejection."),
       ("•  Cohort size:",
        " 7, 8, 20, 9 and 7 participants. Between-subject SD of 0.06 to 0.13 is large relative "
        "to the differences between arms."),
       ("•  The branch is not compact:",
        " 238,028 parameters against a 19,505-parameter stem path, so no parameter-efficiency "
        "claim is made.")]
write(get(s, "TextBox 4"),
      [P(R(b, 17, bold=True), R(t, 17, color=GRY), sa=11, ls=1.18) for b, t in bul])
write(get(s, "Rectangle 5"),
      [P(R("THE TEST THAT WOULD SETTLE IT   ", 13, bold=True),
         R("choose a protocol whose class blocks fall between 18 and 214 windows, register "
           "the prediction, and run it once.", 17, italic=True), sa=8, ls=1.15)])

# ------------------------------------------------------------------ 23 conclusion
s = clone(2)
title(s, "Conclusion")
place(get(s, "TextBox 4"), left=0.70, top=1.70, width=12.00, height=1.95)
write(get(s, "TextBox 4"), [
    P(R("The covariance structure a log-power stem discards does carry movement intent. "
        "A tangent-space branch raises balanced accuracy on four of five cohorts, and on "
        "20 of 20 participants of a cohort held out from every design decision.",
        18, color=GRY), sa=10, ls=1.18),
    P(R("Where it fails, it fails predictably: the estimator memory against the class-block "
        "length separates all five cohorts with nothing fitted.", 18, color=GRY),
      sa=8, ls=1.18)])
write(get(s, "Rectangle 5"), [
    P(R("The condition is the part worth building on, and the part least finished.", 18, bold=True),
      sa=6, ls=1.15),
    P(R("It yields no rate a practitioner could set, and its upper side has never been "
        "measured.", 16, color=GRY), sa=8, ls=1.18)])
write(get(s, "TextBox 6"), [
    P(R("Next: ", 18, bold=True),
      R("a protocol whose class blocks fall between 18 and 214 windows, with the "
        "prediction registered before the run. That is the experiment that would turn "
        "an ordering into a rate.", 18, color=GRY), sa=8, ls=1.18)])

# ------------------------------------------------------------------ 24 references
s = clone(11)
write(get(s, "TextBox 4"),
      [P(R("Works cited on these slides, keyed to their [author, year] tags.", 12.5,
           italic=True, color=GRY))])
left = [
    ("[Sarkar 2026]", "S. Sarkar et al., “EEG-controlled exoskeleton for walking and standing: a longitudinal multimodal dataset,” Sci. Data, 2026. (OpenNeuro ds007788)"),
    ("[Ferrero 2024]", "L. Ferrero et al., “Brain-machine interface based on deep learning to control a lower-limb robotic exoskeleton,” J. NeuroEng. Rehabil., 21(48), 2024."),
    ("[Soriano 2025]", "P. Soriano-Segura, M. Ortiz et al., “Characterization of error-related potentials during exoskeleton command via deep learning,” J. NeuroEng. Rehabil., 2025."),
    ("[Tortora 2023]", "S. Tortora et al., “Cortical and muscular activity under robot-assisted gait modes during exoskeleton walking,” 2023."),
    ("[Luu 2017]", "T. P. Luu et al., “Real-time EEG-based brain-computer interface to a virtual avatar enhances cortical involvement in human treadmill walking,” Sci. Rep., 7:8895, 2017."),
    ("[Schalk 2004]", "G. Schalk et al., “BCI2000: a general-purpose brain-computer interface (BCI) system,” IEEE TBME, 51(6):1034-1043, 2004."),
    ("[Brunner 2008]", "C. Brunner et al., “BCI Competition 2008 – Graz data set A,” Graz University of Technology, 2008. (BNCI2014-001)"),
    ("[Lawhern 2018]", "V. J. Lawhern et al., “EEGNet: a compact CNN for EEG-based brain-computer interfaces,” J. Neural Eng., 15(5), 2018."),
]
right = [
    ("[Schirrmeister 2017]", "R. T. Schirrmeister et al., “Deep learning with convolutional neural networks for EEG decoding and visualization,” Hum. Brain Mapp., 38(11):5391-5420, 2017."),
    ("[Song 2023]", "Y. Song et al., “EEG Conformer: convolutional transformer for EEG decoding and visualization,” IEEE TNSRE, 31:710-719, 2023."),
    ("[Barachant 2012]", "A. Barachant et al., “Multiclass brain-computer interface classification by Riemannian geometry,” IEEE TBME, 59(4), 2012."),
    ("[He 2020]", "H. He and D. Wu, “Transfer learning for brain-computer interfaces: a Euclidean space data alignment approach,” IEEE TBME, 67(2):399-410, 2020."),
    ("[Rodrigues 2019]", "P. L. C. Rodrigues et al., “Riemannian Procrustes analysis: transfer learning for brain-computer interfaces,” IEEE TBME, 66(8):2390-2401, 2019."),
    ("[Wang 2021]", "D. Wang et al., “Tent: fully test-time adaptation by entropy minimization,” ICLR, 2021."),
    ("[Gong 2022]", "T. Gong et al., “NOTE: robust continual test-time adaptation against temporal correlation,” NeurIPS, 2022."),
    ("[Pion-Tonachini 2019]", "L. Pion-Tonachini et al., “ICLabel: an automated electroencephalographic independent component classifier,” NeuroImage, 198:181-197, 2019."),
]
for name, refs in (("TextBox 5", left), ("TextBox 6", right)):
    write(get(s, name), [P(R(tag + "  ", 11, bold=True), R(txt, 10.5, color=GRY), sa=9, ls=1.06)
                         for tag, txt in refs])

# ------------------------------------------------------------------ drop the proposal slides
ids = prs.slides._sldIdLst
for sid in list(ids)[:N_ORIG]:
    prs.part.drop_rel(sid.rId)
    ids.remove(sid)
prs.save(str(OUT))
print("saved %s | %d slides" % (OUT.name, len(prs.slides)))
