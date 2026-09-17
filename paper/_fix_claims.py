"""Second pass: bring the body's claims into line with Section VI.

Every replacement below removes an assertion the corrected probe contradicts.
Where a claim was load-bearing for the narrative it is restated as an intent
plus a forward reference, not deleted, so the reader can see what was attempted
and what it cost.
"""
from pathlib import Path

P = Path("calmnet_paper_final.tex")
t = P.read_text(encoding="utf-8")

SUBS = [
    # --- Introduction: C2 ---
    ("The classifier reads only the\nintent subspace, so the walk/stop decision is provably movement-invariant. This\nresolves, rather than apologises for, the movement confound that inflates naive\naccuracy.",
     "The classifier reads only the\nintent subspace. The intent of this design is a walk/stop decision that does not\ndepend on movement; Section~\\ref{sec:reeval} reports that under a corrected\ninvariance probe the objective is \\emph{not} achieved, and that no architecture we\ntested achieves it."),

    # --- Results: MID paragraph ---
    ("we read $0.65$ as a conservative lower bound on\ntrue neural decodability, and, to our knowledge, the first movement-invariant such\nestimate reported for this paradigm. Together these results confirm that CALM-Net\ndecodes intent rather than the motion it commands.",
     "we read $0.65$ as a conservative lower bound on\ntrue neural decodability. We stress that this figure was obtained with the\ncross-split probe later found to be misspecified; Section~\\ref{sec:reeval} re-derives\nit and withdraws the accompanying invariance claim."),

    # --- Figure 2 caption ---
    ("without and with MID; MID drives it to zero\n(including for the movement-dominated subjects), confirming the decision is made from\nmovement-invariant neural features.}",
     "without and with MID, as measured by the\n\\emph{original} cross-split probe. Section~\\ref{sec:reeval} shows this probe\nconflates invariance with distribution shift; under the corrected estimator the same\nrepresentation retains recoverable movement.}"),

    # --- Discussion ---
    ("CALM-Net trades that\ninflated number for one that means what it says: a movement-invariant decode of\n$0.69$, calibrated confidence, and prediction sets whose coverage provably tracks its\ntarget as the recording drifts over weeks. For a device that moves a person's legs, an\nhonest $0.69$ that abstains when unsure is safer than a movement-inflated $0.83$ that\ncannot decline. This reframes the field's usual objective: the contribution is not a\nhigher number but a trustworthy one.",
     "CALM-Net was designed to trade that\ninflated number for one that means what it says. Section~\\ref{sec:reeval} shows the\ntrade was not achieved: the decode is not movement-invariant, and the selective head\nmeets its error bound largely by declining to commit. The argument we can still make is\nthe weaker but more durable one --- that on a movement-confounded paradigm an accuracy\nfigure is uninterpretable without a leakage measurement beside it, and that published\nrankings on this dataset, ours included, do not survive one. For a device that moves a\nperson's legs, the relevant contribution is not a higher number but a criterion for\nknowing when a number is meaningless."),

    # --- Conclusion ---
    ("We presented CALM-Net, a multimodal framework that decodes movement-invariant neural\nintent, remains calibrated across weeks, and abstains to a safe stop under a\ndistribution-free guarantee. Treating the exoskeleton's inertial signals as a modality\nto disentangle against, rather than a confound to hide, converts the paradigm's central\ndifficulty into its contribution: to our knowledge the first movement-invariant estimate\nof neural walk/stop decodability on the NeuroRex dataset, together with an\nadaptive-conformal coverage guarantee that holds on all seven subjects. Future work\nimplements and ablates the full Riemannian/attention encoder, validates on patient\npopulations, and closes the loop online.",
     "We presented CALM-Net, a multimodal framework intended to decode movement-invariant\nneural intent, remain calibrated across weeks, and abstain to a safe stop under a\ndistribution-free guarantee, and a re-evaluation showing that the first of those three\naims is not met --- by our model or by any of the $131$ architecture variants we tested.\n\nThe negative result is the finding. On this paradigm a movement-only classifier beats\nevery neural decoder; accuracy and movement leakage correlate at $r = +0.68$ across the\narchitecture space; the apparent best configuration is the leakiest; and\nbetween-architecture differences are no larger than seed noise. A $1{,}831$-parameter\nclosed-form pipeline is more invariant than every learned alternative we tried, which is\nconsistent with established transfer-learning practice rather than a discovery of ours.\n\nWhat we offer for reuse is the invariance probe, correctly specified: an IMU-referenced,\nwithin-distribution, session-grouped estimate of how much nuisance a decoder's\nrepresentation retains. It is cheap, it is model-agnostic, and applied here it\ninvalidates our own earlier conclusions --- which is the property one should want from a\ndiagnostic. Future work should establish published baselines (pyriemann MDM, MOABB\npipelines, published domain-adaptation comparators) before any architectural claim is\nadvanced, and should treat the Castermans--Nathan dispute over the cortical origin of\ngait EEG as an open question that decoder-level leakage measurement can inform."),
]

missing = []
for old, new in SUBS:
    if old in t:
        t = t.replace(old, new, 1)
    else:
        missing.append(old.split("\n")[0][:60])

P.write_text(t, encoding="utf-8")
print(f"applied {len(SUBS) - len(missing)}/{len(SUBS)} replacements")
for m in missing:
    print("  NOT FOUND:", m)
