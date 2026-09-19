"""Q&A preparation notes for the final presentation.

Writes QA_Preparation_Notes.docx and .md. Answers are grouped by the question
likely to prompt them, and every number is one the manuscript reports.

The previous version of these notes described a model the paper no longer
contains: 24,181 parameters, an adaptive alignment layer, a cross-epoch
transformer and a selection head, three cohorts, and a drift-reduction claim.
Walking into a viva with those numbers would be worse than having no notes, so
this file is generated rather than edited, and it is regenerated whenever the
results change.

    python presentation/qa_notes.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

COHORTS = [
    ("", "A, exoskeleton", "B, treadmill", "C, motor exec.", "D, BCI IV-2a",
     "E, exoskeleton"),
    ("Source", "OpenNeuro ds007788", "MoBI (Luu 2017)", "PhysioNet EEGMMIDB",
     "BNCI2014-001", "DECODED / EUROBENCH"),
    ("Participants", "7", "8", "20", "9", "7"),
    ("Recordings", "9 sessions over weeks", "3 trials", "6 motor-exec. runs",
     "2 sessions, different days", "runs within a session"),
    ("EEG", "60 ch, 100 Hz", "64 ch, 100 Hz", "64 ch, 160 to 100 Hz",
     "22 ch, 250 to 100 Hz", "27 ch, 200 to 100 Hz"),
    ("Task", "walk vs stop", "walk vs stand", "movement vs rest",
     "left vs right hand", "walk (MI) vs stand"),
    ("Fitted on", "sessions 1-3", "trial 1", "runs 3, 5, 7, 9", "session 1",
     "first 8 runs"),
    ("Tested on", "sessions 4-9", "trials 2-3", "runs 11, 13", "session 2",
     "remaining runs"),
    ("Window / stride", "4 s / 0.5 s", "2 s", "2 s / 0.5 s",
     "4 s, one per trial", "4 s / 0.5 s"),
    ("Class blocks", "789", "66", "3600", "734", "192"),
    ("Median block", "18 windows", "214 windows", "5 windows", "1 window",
     "13 windows"),
    ("Longest block", "486", "2211", "5", "6", "33"),
]

PARAMS = [("Multi-scale power stem path", "19,505"),
          ("Tangent-space branch", "238,028"),
          ("Fusion layer", "32,896"),
          ("Output layer", "514"),
          ("Total", "290,943")]

SECTIONS = [
    ("1. Datasets and trials", [
        ("table", COHORTS),
        "Cohort A in words: about 5031 fitting windows per participant.",
        "Cohort D in words: 288 trials per participant, but only about 100 fitting "
        "windows, the fewest of the five cohorts. That is why five of the eight "
        "published decoders are left near chance there under shared hyperparameters.",
        "Cohort E in words: each run is 15 s standing, 24 s walking under kinesthetic "
        "motor imagery, 22 s walking under a counting distractor, 14 s standing. We "
        "decode standing against walking under motor imagery and drop the counting "
        "condition rather than folding it into either class.",
        "If asked why the cohort B block length changed from earlier drafts: it is 214 "
        "windows, measured on the fitting recordings over a contiguous timeline. The "
        "earlier figure of 309 was measured on post-split subsets, which are not "
        "contiguous, so it was wrong.",
    ]),
    ("2. Model architecture", [
        "290,943 parameters. The input is a 60-channel, 4-second window on cohort A.",
        "There are two paths and nothing else.",
        "1. Multi-scale power stem. Three temporal resolutions (64, 128, 256 ms), each "
        "with depthwise spatial filters, then square, pool, logarithm. This is log band "
        "power per channel, which is the diagonal of the window's spatial covariance. "
        "The design follows ShallowFBCSPNet.",
        "2. Tangent-space branch. Form the window covariance C, shrink it toward a "
        "scaled identity at lambda = 0.1, and map it to the tangent space at a reference "
        "M: phi(X) = vec(log(M^-1/2 C M^-1/2)), with off-diagonal entries scaled by "
        "root 2 so the Euclidean norm equals the matrix Frobenius norm. 1830 dimensions "
        "for 60 channels, projected to 128. This reads what the stem discards.",
        "The two 128-dimensional outputs are concatenated and fused into one linear "
        "classifier.",
        "The reference M is a running mean over incoming window covariances. It uses no "
        "labels and keeps updating at inference.",
        ("table", [("Component", "Parameters")] + PARAMS),
        "If asked why the tangent map carries no gradient: backward through an "
        "eigendecomposition contains 1/(lambda_i - lambda_j) terms, which diverge on "
        "near-degenerate spectra, and covariances from 60 channels and 400 samples are "
        "routinely near-degenerate. The projection after it is learned.",
    ]),
    ("3. Contributions", [
        "1. A tangent-space branch beside the log-power stem, with a reference estimated "
        "from the input alone and no labels used at inference.",
        "2. Measured on five cohorts and 51 participants, three seeds each. Accuracy "
        "rises on four: cohort C 0.779 to 0.843, which is +0.064 (p = 8.8e-5, 20 of "
        "20); cohort A 0.864 to 0.909; cohort D 0.701 to 0.741 (p = 0.020); cohort E "
        "0.928 to 0.949 (p = 0.047). It costs 0.175 on cohort B, 0.744 down to 0.568.",
        "3. Condition 1, a two-sided rate condition: the estimator memory mu = N/m must "
        "be longer than a class block and shorter than the drift timescale. Read from the "
        "protocol alone, it separates all five cohorts with nothing fitted.",
        "4. A causal test of that condition, by moving the memory directly on one cohort "
        "and switching the effect off and on in both directions.",
        "5. A comparison with eight published decoders trained in one pipeline with "
        "shared splits, seeds and metric code, including the cohort where most of them "
        "fail to train.",
        "6. A stronger version of the condition, pre-registered and reported as falsified.",
    ]),
    ("4. Limitations, say these before you are asked", [
        "The block-length boundary was found after the fact. It separates all five "
        "cohorts with nothing fitted, but it was identified after cohorts A, B and C had "
        "run. Only its application to D and E was prospective.",
        "The artefact control rests on seven participants. ICA cleaning takes accuracy "
        "from 0.897 to 0.859, a change of -0.038 (p = 0.58, four of seven unchanged or "
        "better). The mean is carried by sub-06 alone at -0.207, and without it the "
        "change is -0.010. It constrains the artefact account; it does not exclude it.",
        "Small cohorts: 7, 8, 20, 9, 7. Between-subject SD of 0.06 to 0.13 is large "
        "relative to the differences between arms.",
        "The branch is not compact. 238,028 parameters against a 19,505-parameter stem "
        "path, so we make no parameter-efficiency claim.",
        "Offline only. No closed-loop testing on a real exoskeleton.",
        "The upper bound of Condition 1 is not measured. Our recordings bound the drift "
        "timescale only from above, at about one session.",
        "Cohort E tests task, not drift: only one participant recorded on two dates, so "
        "its split is temporal within a session. It is the closest match to the target "
        "application and the weakest test of longitudinal shift.",
        "Best-epoch model selection is upward-biased on small folds. It applies equally "
        "to every arm and baseline in a pipeline.",
    ]),
    ("5. Why this had not been done", [
        "Data: long multi-session exoskeleton EEG was rare. ds007788 gives 9 sessions "
        "over weeks per participant; earlier exoskeleton datasets were single-session.",
        "Separate fields: Riemannian alignment estimates its reference over a complete "
        "recording, offline, so the recording must exist first. Test-time adaptation "
        "removes that constraint but was developed for images. We put the two together "
        "inside one network layer.",
        "Hidden failure mode: the rate problem only appears on protocols whose block "
        "lengths differ by orders of magnitude. Ours span 1 to 214 windows. With a "
        "single dataset you would never see it.",
        "Say 'to our knowledge'. Never say 'nobody has done this'.",
    ]),
    ("6. Technical questions", [
        "Validation split. 30 per cent of the class blocks from the fitting data, chosen "
        "per class, with whole blocks kept together so overlapping windows cannot leak. "
        "Used for early stopping, best-epoch selection and temperature scaling. Test "
        "recordings are never used for training or selection. The branch sees test EEG "
        "but never test labels.",
        "Training. AdamW, lr 3e-4, weight decay 1e-2, one-cycle schedule. Batch 32, at "
        "most 30 epochs, early stopping after 12 without improvement, gradient clipping "
        "at 1.0, class-weighted cross-entropy. Hardware: RTX 2060 and RTX 5080, PyTorch "
        "with braindecode and MNE-Python. Do not quote a per-model training time; there "
        "is no clean measurement.",
        "Seed variance. Three data-split seeds. A seed changes the validation blocks, "
        "the weight initialisation and the batch order. Across the five cohorts the "
        "reported model spans 0.007 to 0.022 and the stem spans 0.011 to 0.057, so the "
        "stem is the noisier arm on all five. We do not interpret differences below "
        "0.02, and differences below the relevant seed spread are reported without a "
        "ranking.",
        "Statistics. Paired Wilcoxon signed-rank over participants, seed-averaged within "
        "participant, with Holm correction where there are multiple comparisons.",
        "Metrics. Balanced accuracy, because classes are imbalanced. Expected "
        "calibration error with 10 bins. Spurious activations per minute of standing. "
        "Missed onsets. Detection latency.",
        "Preprocessing. 8-30 Hz zero-phase band-pass, which excludes the 1-2 Hz gait "
        "rhythm and most muscle power above 30 Hz. EOG regression on cohort A, none on "
        "B to E. Amplitude normalisation globally per recording, before the squaring "
        "stage. No ICA in the main results; the ICA control is reported separately.",
        "Adaptation rate. Momentum m = 0.2 with batch N = 32 gives mu = N/m = 160 "
        "windows. Cohorts A (18), C (5), D (1) and E (13) have blocks shorter than 160, "
        "so the branch is admissible. Cohort B (214) does not.",
    ]),
    ("7. The questions that will actually be asked", [
        "'Isn't cohort C just your best result?' No, and it is the one cohort where that "
        "cannot be true. Every design choice was fixed on A and B before C was run. That "
        "is what held out means here, and it is why C carries the accuracy claim rather "
        "than A, where n = 7 and p = 0.22.",
        "'You failed on cohort B.' Yes, by 0.175, and we predicted it before training "
        "from the protocol. Cohort B presents walk in blocks of 214 windows against an "
        "estimator memory of 160, so the reference converges onto the walking covariance "
        "and whitens away the difference. Enlarging the memory recovers 74 per cent of "
        "the loss.",
        "'Are you decoding muscle, not brain?' Three things say probably not, none of "
        "them conclusive. ICA cleaning costs 0.038 and is not significant. The loss does "
        "not follow how much muscle was removed (rho = +0.05, where dependence would be "
        "negative). And band power falls over sensorimotor cortex during walking, which "
        "is the sign of mu and beta desynchronisation; gross artefact would raise it.",
        "'Why is the branch worth 238,028 parameters?' On the held-out cohort it is "
        "worth +0.064 and it beats the best published decoder by 0.060. We make no "
        "efficiency claim, and the parameter count is in the architecture figure.",
        "'Accuracy went up but is the controller better?' Not everywhere, and we report "
        "that. Calibration improves on four of five cohorts and missed onsets fall on "
        "four, but on cohort A spurious walk commands rise from 0.70 to 2.09 per minute "
        "of standing. On a device that moves someone's legs that is a real cost.",
        "'Is fusion doing anything, or is it just the branch?' Class separation measured "
        "in the trained model's own activations is 0.74 for the stem, 1.22 for the "
        "branch and 2.41 after fusion. Higher than either path alone, so they are "
        "complementary.",
        "If you do not know a number, say you will check it. Do not estimate one aloud.",
    ]),
]


def _flat(sec):
    return [x for x in sec if not (isinstance(x, tuple) and x[0] == "table")]


def write_md():
    out = ["# Q&A preparation notes", "",
           "For `CALM-Net_final_presentation.pptx`. Every number here is one the "
           "manuscript reports.", ""]
    for head, items in SECTIONS:
        out += ["## " + head, ""]
        for it in items:
            if isinstance(it, tuple) and it[0] == "table":
                rows = it[1]
                out += ["| " + " | ".join(str(c) for c in rows[0]) + " |",
                        "|" + "---|" * len(rows[0])]
                out += ["| " + " | ".join(str(c) for c in r) + " |"
                        for r in rows[1:]]
                out += [""]
            else:
                out += [it, ""]
    (HERE / "QA_Preparation_Notes.md").write_text("\n".join(out), encoding="utf-8")


def write_docx():
    from docx import Document
    from docx.shared import Pt, Cm
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    doc.add_heading("Q&A Preparation Notes", level=1)
    p = doc.add_paragraph()
    p.add_run("For CALM-Net_final_presentation.pptx. Every number comes from the "
              "results files or the manuscript. Regenerate with "
              "presentation/qa_notes.py whenever the results change.").italic = True
    for head, items in SECTIONS:
        doc.add_heading(head, level=2)
        for it in items:
            if isinstance(it, tuple) and it[0] == "table":
                rows = it[1]
                t = doc.add_table(rows=1, cols=len(rows[0]))
                t.style = "Table Grid"
                for cell, text in zip(t.rows[0].cells, rows[0]):
                    cell.text = ""
                    cell.paragraphs[0].add_run(str(text)).bold = True
                for r in rows[1:]:
                    cells = t.add_row().cells
                    for cell, text in zip(cells, r):
                        cell.text = ""
                        run = cell.paragraphs[0].add_run(str(text))
                        if r is rows[1] or len(rows[0]) == 2:
                            pass
                        run.font.size = Pt(9)
                for row in t.rows:
                    for cell in row.cells:
                        for par in cell.paragraphs:
                            par.paragraph_format.space_after = Pt(1)
                doc.add_paragraph()
            else:
                par = doc.add_paragraph(it, style="List Bullet")
                par.paragraph_format.space_after = Pt(4)
    doc.save(str(HERE / "QA_Preparation_Notes.docx"))


if __name__ == "__main__":
    write_md()
    write_docx()
    n = sum(len(_flat(v)) for _, v in SECTIONS)
    print("QA notes: %d sections, %d points" % (len(SECTIONS), n))
