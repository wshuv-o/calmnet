"""Updated-manuscript deck, same academic style as the midterm deck.

Serif type, black on white, no accent colour, plain ruled tables. Numbers are
read from results/*.json at build time, so rebuilding after an experiment
finishes updates the slides; anything not yet run prints as pending.
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

BLK = RGBColor(0x11, 0x11, 0x11)
GRY = RGBColor(0x55, 0x55, 0x55)
LINE = RGBColor(0x00, 0x00, 0x00)
BORD = RGBColor(0x44, 0x44, 0x44)
WHT = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Times New Roman"
W, H = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def slide(bg=WHT):
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(1, 0, 0, W, H)
    r.fill.solid(); r.fill.fore_color.rgb = bg
    r.line.fill.background(); r.shadow.inherit = False
    s.shapes._spTree.remove(r._element); s.shapes._spTree.insert(2, r._element)
    return s


def box(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tf = s.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    return tf


def para(tf, text, size, color=BLK, bold=False, italic=False, first=False,
         align=PP_ALIGN.LEFT, space=8, lh=1.1):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align; p.space_after = Pt(space); p.line_spacing = lh
    for seg in (text if isinstance(text, list) else [(text, {})]):
        t, o = seg if isinstance(seg, tuple) else (seg, {})
        r = p.add_run(); r.text = t; f = r.font
        f.size = Pt(o.get("size", size)); f.name = FONT
        f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
        f.color.rgb = o.get("color", color)
    return p


def rect(s, x, y, w, h, fill=None, line=None, lw=1.0):
    sh = s.shapes.add_shape(1, x, y, w, h)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh


def obox(s, x, y, w, h, lw=1.0):
    return rect(s, x, y, w, h, fill=WHT, line=BORD, lw=lw)


def hrule(s, x, y, w, pt=1.0):
    rect(s, x, y, w, Pt(pt), fill=LINE)


def header(s, text):
    tf = box(s, Inches(0.7), Inches(0.5), Inches(12), Inches(0.85))
    para(tf, text, 30, BLK, bold=True, first=True, lh=1.0)
    hrule(s, Inches(0.7), Inches(1.32), Inches(11.93), pt=1.4)


def no_style(gt):
    tblPr = gt._tbl.find(qn("a:tblPr"))
    if tblPr is None:
        return
    for a in ("firstRow", "firstCol", "lastRow", "lastCol", "bandRow", "bandCol"):
        if a in tblPr.attrib:
            tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = tblPr.makeelement(qn("a:tableStyleId"), {}); tblPr.append(sid)
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"


def tcell(gt, ri, ci, text, size, bold, align):
    cell = gt.cell(ri, ci); cell.fill.background()
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Pt(10); cell.margin_right = Pt(6)
    cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
    p = cell.text_frame.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; f = r.font; f.name = FONT
    f.size = Pt(size); f.bold = bold; f.color.rgb = BLK


def table(s, header_row, rows, x, y, widths, hh=0.5, dr=0.46, size=15,
          bold_rows=()):
    gt = s.shapes.add_table(1 + len(rows), len(header_row), Inches(x), Inches(y),
                            Inches(sum(widths)), Inches(hh + len(rows) * dr)).table
    no_style(gt)
    for ci, w in enumerate(widths):
        gt.columns[ci].width = Inches(w)
    gt.rows[0].height = Inches(hh)
    for ci, name in enumerate(header_row):
        tcell(gt, 0, ci, name, size, True, PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER)
    for ri, row in enumerate(rows, start=1):
        gt.rows[ri].height = Inches(dr)
        for ci, v in enumerate(row):
            tcell(gt, ri, ci, v, size - 0.5, ri - 1 in bold_rows,
                  PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER)
    tw = sum(widths)
    hrule(s, Inches(x), Inches(y), Inches(tw), pt=1.5)
    hrule(s, Inches(x), Inches(y + hh), Inches(tw), pt=1.0)
    hrule(s, Inches(x), Inches(y + hh + len(rows) * dr), Inches(tw), pt=1.5)


def picture(s, path, top, width):
    pic = s.shapes.add_picture(str(path), 0, Inches(top), width=Inches(width))
    pic.left = int((W - pic.width) / 2)
    return pic


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


def f3(v):
    return "%.3f" % v


# ---------------------------------------------------------------- data
EV = load("overnight_eval.json")
C3 = EV.get("cohort3", {})
P1, P2 = C3.get("P1", {}), C3.get("P2", {})
ICAC = EV.get("ica_control", {})


def done(r):
    return "mean_diff" in r


# ═══ 1 · TITLE ═══
s = slide()
tf = box(s, Inches(0.8), Inches(2.0), Inches(11.73), Inches(1.9))
para(tf, "Label-free covariance alignment for longitudinal EEG decoding", 40, BLK,
     bold=True, first=True, align=PP_ALIGN.CENTER, lh=1.1)
hrule(s, Inches(4.42), Inches(3.95), Inches(4.5), pt=1.5)
tf = box(s, Inches(1.4), Inches(4.25), Inches(10.53), Inches(0.6))
para(tf, "Updated manuscript", 22, GRY, first=True, align=PP_ALIGN.CENTER)
tf = box(s, Inches(1.4), Inches(5.55), Inches(10.53), Inches(1.4))
para(tf, "Esm E Moula Chowdhury Abha  [26-94089-2]", 15, GRY, first=True, align=PP_ALIGN.CENTER, space=5)
para(tf, "Md Wahiduzzaman Suva  [26-94088-2]", 15, GRY, align=PP_ALIGN.CENTER, space=5)
para(tf, "Supervisor: Dr. Muhammad Hasibur Rashid Chayon", 15, GRY, align=PP_ALIGN.CENTER)

# ═══ 2 · WHAT CHANGED SINCE THE MIDTERM ═══
s = slide(); header(s, "What Changed Since the Midterm")
items = [
    ("The movement-leakage measure was confounded. ",
     "A decoder carrying no movement information at all still scored R² = 0.863, because walk and stop differ in "
     "movement by definition. The invariance claim was withdrawn and replaced by a probe conditioned on the label."),
    ("Abstention did not work. ",
     "Declining the least-confident 10 % of windows lowers accuracy in 14 of 18 arms. It is reported as a negative result."),
    ("The contribution moved to longitudinal drift. ",
     "When an in-network alignment layer helps across sessions, and when it harms."),
    ("External validation was added. ",
     "An independent second cohort, and a third cohort tested against a prediction registered before any model was trained."),
]
tf = box(s, Inches(0.7), Inches(1.7), Inches(11.93), Inches(5.3))
for i, (b, d) in enumerate(items):
    para(tf, [(b, {"size": 18, "bold": True, "color": BLK}), (d, {"size": 17, "color": GRY})],
         17, first=(i == 0), space=16, lh=1.2)

# ═══ 3 · THE PROBLEM ═══
s = slide(); header(s, "A Decoder Must Keep Working for Weeks")
tf = box(s, Inches(0.7), Inches(1.75), Inches(11.93), Inches(1.9))
para(tf, "An exoskeleton decoder is fitted once and used for weeks. Electrode impedance, montage and skin contact "
     "change between sessions, and the second-order statistics every spatial filter depends on drift with them.",
     20, BLK, first=True, lh=1.25)
q = obox(s, Inches(0.7), Inches(3.85), Inches(11.93), Inches(1.4), lw=1.3)
qt = q.text_frame; qt.word_wrap = True; qt.vertical_anchor = MSO_ANCHOR.MIDDLE; qt.margin_left = Pt(18)
para(qt, "Correcting the drift inside the network, without labels, removes the per-session calibration block.",
     21, BLK, bold=True, first=True, lh=1.15)
tf = box(s, Inches(0.7), Inches(5.5), Inches(11.93), Inches(1.3))
para(tf, [("The question. ", {"size": 19, "bold": True, "color": BLK}),
          ("A running estimate adapts to whatever is currently streaming. Under which recording protocols is that safe?",
           {"size": 19, "color": GRY})], 19, first=True, lh=1.2)

# ═══ 4 · ARCHITECTURE ═══
s = slide(); header(s, "Architecture")
picture(s, HERE / "fig_arch_v2.png", 1.75, 12.3)
tf = box(s, Inches(0.7), Inches(5.55), Inches(11.93), Inches(1.5))
para(tf, [("Adaptive alignment ", {"size": 16, "bold": True}),
          ("whitens each window by a running covariance estimate, updated without labels at inference.  ",
           {"size": 16, "color": GRY}),
          ("Multi-scale power stem ", {"size": 16, "bold": True}),
          ("preserves log-power ratios.  ", {"size": 16, "color": GRY}),
          ("24,181 parameters ", {"size": 16, "bold": True}),
          ("in the proposed configuration.", {"size": 16, "color": GRY})], 16, first=True, lh=1.25)

# ═══ 5 · DATA ═══
s = slide(); header(s, "Three Cohorts")
table(s, ["", "Cohort A", "Cohort B", "Cohort C"],
      [["Source", "OpenNeuro ds007788", "Treadmill MoBI", "PhysioNet EEGMMIDB"],
       ["Task", "Exoskeleton walk / stop", "Treadmill walk / stand", "Rest / movement"],
       ["Participants", "7", "8", "20"],
       ["Channels", "60", "64", "64"],
       ["Class block (windows)", "18", "309", "5"],
       ["Role", "Development", "External validation", "Registered prediction"]],
      0.7, 1.75, [3.23, 2.9, 2.9, 2.9], hh=0.52, dr=0.52, size=16)
tf = box(s, Inches(0.7), Inches(5.55), Inches(11.93), Inches(1.4))
para(tf, "The class-block length, measured in windows, is the property that turns out to decide whether "
     "in-network alignment helps.", 17, GRY, italic=True, first=True, lh=1.2)

# ═══ 6 · DRIFT REDUCTION ═══
s = slide(); header(s, "Alignment Removes Session Drift")
picture(s, RES / "fig_drift.png", 1.6, 9.2)
tf = box(s, Inches(0.7), Inches(6.0), Inches(11.93), Inches(1.1))
para(tf, [("52.0 ± 4.5 % ", {"size": 18, "bold": True}), ("on cohort A and ", {"size": 18, "color": GRY}),
          ("83.9 ± 3.9 % ", {"size": 18, "bold": True}),
          ("on cohort B reduction in session-to-session covariance distance, in ", {"size": 18, "color": GRY}),
          ("15 of 15 participants", {"size": 18, "bold": True}), (", p ≤ 10⁻⁴.", {"size": 18, "color": GRY})],
     18, first=True, lh=1.2)

# ═══ 7 · ACCURACY ═══
s = slide(); header(s, "Decoder Accuracy on Cohort A")
table(s, ["Configuration", "Balanced acc.", "ECE", "Parameters"],
      [["Align + gate (proposed)", "0.901", "0.039", "24,181"],
       ["Align + context + gate", "0.878", "0.074", "618,997"],
       ["Context + gate", "0.827", "0.091", "n/a"],
       ["Stem only", "0.827", "0.082", "n/a"],
       ["ATCNet, same pipeline", "0.913", "n/a", "45,478"]],
      0.7, 1.75, [4.43, 2.5, 2.5, 2.5], hh=0.52, dr=0.52, size=16, bold_rows=(0,))
tf = box(s, Inches(0.7), Inches(5.15), Inches(11.93), Inches(1.9))
para(tf, [("0.901 at 53 % of ATCNet's parameters. ", {"size": 17, "bold": True}),
          ("Removing alignment costs 0.074 on this cohort. Seven participants, nine sessions each, "
           "fitted on sessions 1–3 and tested on sessions 4–9 recorded weeks later.", {"size": 17, "color": GRY})],
     17, first=True, lh=1.25)

# ═══ 8 · IS THE ACCURACY NEURAL ═══
s = slide(); header(s, "Is the Accuracy Neural")
tf = box(s, Inches(0.7), Inches(1.6), Inches(5.9), Inches(5.4))
para(tf, "Three separate lines of evidence", 18, BLK, bold=True, first=True, space=12)
para(tf, [("No motion input. ", {"size": 16, "bold": True}),
          ("The IMU is used only to score leakage after the fact.", {"size": 16, "color": GRY})], 16, space=10, lh=1.2)
para(tf, [("8–30 Hz band. ", {"size": 16, "bold": True}),
          ("Excludes the 1–2 Hz gait cycle and most muscle activity above 30 Hz.", {"size": 16, "color": GRY})],
     16, space=10, lh=1.2)
para(tf, [("Leakage probe. ", {"size": 16, "bold": True}),
          ("Injecting 2 % motion reaches 0.903 accuracy and moves the probe to +0.067. "
           "The proposed decoder reaches 0.901 with the probe at −0.068.", {"size": 16, "color": GRY})],
     16, space=10, lh=1.2)
box_ica = obox(s, Inches(6.85), Inches(1.6), Inches(5.78), Inches(5.2), lw=1.3)
bt = box_ica.text_frame; bt.word_wrap = True; bt.vertical_anchor = MSO_ANCHOR.TOP
bt.margin_left = Pt(16); bt.margin_right = Pt(16); bt.margin_top = Pt(14)
para(bt, "ICA artefact control", 18, BLK, bold=True, first=True, space=10)
para(bt, "ICLabel removes eye, muscle, heart, line-noise and channel-noise components, then the proposed decoder "
     "is retrained on the cleaned signal.", 15, GRY, space=12, lh=1.2)
if done(ICAC):
    para(bt, [("Uncleaned  ", {"size": 17, "color": GRY}), (f3(ICAC["mean_a"]), {"size": 17, "bold": True})],
         17, space=4)
    para(bt, [("Cleaned      ", {"size": 17, "color": GRY}), (f3(ICAC["mean_b"]), {"size": 17, "bold": True})],
         17, space=4)
    para(bt, [("Change       ", {"size": 17, "color": GRY}), ("%+.3f" % ICAC["mean_diff"], {"size": 17, "bold": True})],
         17, space=10)
    para(bt, "Paired over %d participants, training task only for both arms." % ICAC["n"],
         13, GRY, italic=True, lh=1.15)
else:
    para(bt, "Running overnight. Result pending.", 17, BLK, italic=True)

# ═══ 9 · EXTERNAL VALIDATION ═══
s = slide(); header(s, "External Validation")
table(s, ["Configuration", "Cohort A", "Cohort B"],
      [["Align + gate", "0.901", "0.581"],
       ["Align + context + gate", "0.878", "0.626"],
       ["Context + gate (no alignment)", "0.827", "0.792"],
       ["Stem only", "0.827", "0.737"]],
      0.7, 1.75, [5.93, 3.0, 3.0], hh=0.52, dr=0.52, size=16)
tf = box(s, Inches(0.7), Inches(4.55), Inches(11.93), Inches(2.4))
para(tf, [("The decoder transfers. ", {"size": 18, "bold": True}),
          ("Without alignment it reaches 0.792 on an independent laboratory's data, with different sensors and "
           "reversed class balance.", {"size": 18, "color": GRY})], 18, first=True, space=14, lh=1.2)
para(tf, [("The alignment layer inverts. ", {"size": 18, "bold": True}),
          ("It adds 0.074 on cohort A and costs 0.211 on cohort B.", {"size": 18, "color": GRY})],
     18, lh=1.2)

# ═══ 10 · WHY IT FAILS ON COHORT B ═══
s = slide(); header(s, "The Estimator Tracks the Streaming Class")
picture(s, RES / "fig_rate.png", 1.55, 10.2)
tf = box(s, Inches(0.7), Inches(5.9), Inches(11.93), Inches(1.2))
para(tf, [("Cohort B presents each class in blocks of 309 windows, ", {"size": 17, "color": GRY}),
          ("longer than the estimator's 160-window memory. ", {"size": 17, "bold": True}),
          ("The estimate converges to the current class and whitens away the difference between classes. "
           "Slowing adaptation recovers 0.090 to 0.098 on every arm that uses alignment and exactly 0.000 on "
           "the arms that do not.", {"size": 17, "color": GRY})], 17, first=True, lh=1.22)

# ═══ 11 · THE CONDITION ═══
s = slide(); header(s, "The Admissible Rate Band")
q = obox(s, Inches(2.2), Inches(1.75), Inches(8.93), Inches(1.25), lw=1.5)
qt = q.text_frame; qt.word_wrap = True; qt.vertical_anchor = MSO_ANCHOR.MIDDLE
para(qt, "τ_block   <   μ   <   τ_drift", 34, BLK, bold=True, first=True, align=PP_ALIGN.CENTER)
tf = box(s, Inches(0.7), Inches(3.3), Inches(11.93), Inches(3.8))
para(tf, [("Lower bound. ", {"size": 18, "bold": True}),
          ("The estimator's memory must outlast a class block, or it tracks the class.", {"size": 18, "color": GRY})],
     18, first=True, space=12, lh=1.2)
para(tf, [("Upper bound. ", {"size": 18, "bold": True}),
          ("It must adapt faster than the session drifts, or it has nothing to correct.", {"size": 18, "color": GRY})],
     18, space=12, lh=1.2)
para(tf, [("One model, one switch. ", {"size": 18, "bold": True}),
          ("Cohort A (18 < 160) enables alignment and reaches 0.901. Cohort B (309 > 160) disables it and reaches "
           "0.792. Both bounds come from the recording protocol, before any training.", {"size": 18, "color": GRY})],
     18, space=12, lh=1.2)
para(tf, "A learned gate cannot find this switch itself. The failure appears only at test time, so no training "
     "gradient contains it.", 15, GRY, italic=True, lh=1.2)

# ═══ 12 · REGISTERED PREDICTION ═══
s = slide(); header(s, "A Prediction Registered in Advance")
tf = box(s, Inches(0.7), Inches(1.6), Inches(11.93), Inches(1.3))
para(tf, [("Cohort C, PhysioNet EEGMMIDB. ", {"size": 17, "bold": True}),
          ("Class blocks of 5 windows, far inside the 160-window memory. The prediction and its thresholds were "
           "committed to version control before any model was trained on this cohort.", {"size": 17, "color": GRY})],
     17, first=True, lh=1.22)


def pred_row(r, name, rule):
    if not done(r):
        return [name, rule, "pending", "pending"]
    return [name, rule, "%+.3f" % r["mean_diff"], r["verdict"].capitalize()]


table(s, ["Prediction", "Rule", "Observed", "Verdict"],
      [pred_row(P1, "P1  Alignment costs little", "mean difference > −0.05"),
       pred_row(P2, "P2  Slowing does not help", "mean difference < +0.05")],
      0.7, 3.15, [4.13, 3.4, 2.2, 2.2], hh=0.52, dr=0.56, size=16)
tf = box(s, Inches(0.7), Inches(5.05), Inches(11.93), Inches(2.0))
para(tf, [("Why P2 matters. ", {"size": 17, "bold": True}),
          ("On cohort B, slowing adaptation recovered 0.090. The condition predicts the same intervention does "
           "nothing here, because there is no class tracking to stop.", {"size": 17, "color": GRY})],
     17, first=True, space=12, lh=1.22)
if done(P1) and done(P2):
    para(tf, "Paired over %d participants. Direction counts and Wilcoxon tests are reported but do not change the verdict."
         % P1["n"], 14, GRY, italic=True, lh=1.2)
else:
    para(tf, "Running overnight. Verdicts fill in when the runs finish.", 14, GRY, italic=True, lh=1.2)

# ═══ 13 · NEGATIVE RESULTS ═══
s = slide(); header(s, "Results Reported as Negative")
cards = [("Abstention", "Declining uncertain windows lowers accuracy in 14 of 18 arms, by 0.045 on average. "
                        "The trained gate ranks windows worse than no ranking at all."),
         ("The no-op control", "It is not inert. It moves distance by −28.8 % to +45.0 % per participant, because "
                               "trace normalisation after a fixed map breaks affine invariance."),
         ("Temporal smoothing", "Its gains are consistent in sign, 6 of 7 participants, but do not survive Holm "
                                "correction at n = 7. Only the shuffled control does.")]
x = 0.7; cw = 3.94; gap = 0.155
for t, d in cards:
    c = obox(s, Inches(x), Inches(1.85), Inches(cw), Inches(3.35))
    ct = c.text_frame; ct.word_wrap = True; ct.vertical_anchor = MSO_ANCHOR.TOP
    ct.margin_left = Pt(16); ct.margin_right = Pt(16); ct.margin_top = Pt(18)
    para(ct, t, 20, BLK, bold=True, first=True, space=12, lh=1.05)
    para(ct, d, 16, GRY, lh=1.25)
    x += cw + gap
tf = box(s, Inches(0.7), Inches(5.45), Inches(11.93), Inches(0.8))
para(tf, "Calibration (ECE 0.039) is a separate result and stands.", 15, GRY, italic=True, first=True)

# ═══ 14 · LIMITATIONS AND NEXT STEPS ═══
s = slide(); header(s, "Limitations and Next Steps")
tf = box(s, Inches(0.7), Inches(1.6), Inches(5.85), Inches(5.4))
para(tf, "Limitations", 19, BLK, bold=True, first=True, space=12)
for t in ["Headline ablation is a single seed.",
          "Condition 1 was formulated after cohort B; cohort C is its first advance test.",
          "No published decoder has been run on cohort B.",
          "The leakage probe cannot register artefact constant within a class."]:
    para(tf, [("•  ", {"size": 16, "bold": True}), (t, {"size": 16, "color": GRY})], 16, space=10, lh=1.2)
tf = box(s, Inches(6.8), Inches(1.6), Inches(5.85), Inches(5.4))
para(tf, "Next", 19, BLK, bold=True, first=True, space=12)
for t in ["Multi-seed headline, about 47 GPU hours.",
          "A cohort whose blocks outlast drift, to test the empty band.",
          "Published baselines on cohort B.",
          "Measure the drift timescale directly, without a decoder."]:
    para(tf, [("•  ", {"size": 16, "bold": True}), (t, {"size": 16, "color": GRY})], 16, space=10, lh=1.2)

# ═══ 15 · METHODS CORRECTIONS (backup) ═══
s = slide(); header(s, "Methods Corrected Since Submission")
table(s, ["Setting", "Submitted", "Corrected to match the code"],
      [["Band-pass", "0.5–40 Hz", "8–30 Hz, zero-phase"],
       ["Artefact step", "none", "EOG regression on cohort A"],
       ["Optimiser", "Adam, lr 10⁻³", "AdamW, lr 3×10⁻⁴, wd 10⁻², one-cycle"],
       ["Early stopping", "not stated", "patience 12"],
       ["Leakage bound", "stated for the decoder", "measured on tangent features"]],
      0.7, 1.75, [2.9, 3.6, 5.43], hh=0.52, dr=0.52, size=15)
tf = box(s, Inches(0.7), Inches(5.0), Inches(11.93), Inches(1.8))
para(tf, "Found while checking the pipeline overnight. None changes a reported number; each would have stopped a "
     "reviewer reproducing the work. The true band excludes more artefact than the submitted text claimed.",
     16, GRY, italic=True, first=True, lh=1.25)

out = HERE / "Updated_manuscript_presentation.pptx"
prs.save(str(out))
print("saved", out.name, "|", len(prs.slides._sldIdLst), "slides")
print("pending:", [k for k, r in (("ICA control", ICAC), ("P1", P1), ("P2", P2)) if not done(r)])
