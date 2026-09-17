# Red numbers in cas_calmnet.tex and what replaces them

Every value wrapped in `\pending{}` renders red. Each is listed here with the run
that will replace it. When all are final, change the macro in the preamble to
`\newcommand{\pending}[1]{#1}` (one line) and delete this file.

Updated 2026-09-17 13:05.

| Red item (where) | Current value | Replaced by | Machine | Result file |
|---|---|---|---|---|
| Drift reduction (abstract, highlights, contributions, §6.1, Discussion item 2) | 52.0 ± 4.5 %, 83.9 ± 3.9 %, t and p values, per-participant ranges | No-op control rerun without trace normalisation in the distance | other Claude session | to be named in its HANDOFF file |
| No-op control paragraph (§6.1) and Fig. 2 | control moves −28.8 % to +45.0 % | same rerun; paragraph rewritten, figure regenerated | other Claude session | same |
| tab:drift caption | t, p | same rerun; `python src/make_tables.py` | other Claude session | results/drift.json (regenerated) |
| Cohort B accuracies (abstract, §6.5 text and Table 12, Learnability, Discussion items 1–2, Limitations, Conclusion) | 0.792, 0.211, 0.581, 0.626, 0.166, 0.092 | align + gate against gate, 3 seeds, current estimator | 5080 queue job 1 | b_gate_aligngate_3seed.json |
| Rate recovery on cohort B (abstract, §5.4 noise, §6.8, tab:rate, prereg paragraph, Conclusion) | +0.090 to +0.098, −0.010 | slowed alignment m = 0.01, 3 seeds; intermediate m = 0.10 and 0.02, 3 seeds | 5080 job 2; other session | b_aligngate_m001_3seed.json; other session's files |
| Residual remark (§6.9) | 0.060, 0.723, 0.783 | same B runs | 5080 job 2 | same |
| "Error in either direction" and seed multiples (§5.4, §6.9, Limitations) | 0.060 to 0.092; 2.3 to 3.8 | recomputed from the B and A three-seed runs | both | both |
| Cohort A slow-rate cost, three seeds (§6.9 note) | seed 0 only (−0.085, 7 of 7, p = 0.016 are final for seed 0) | align + gate at m = 0.01, seeds 1–2 | other Claude session | its file |
| Component ablation (tab:ablation, Fig. ablation, cross-epoch context paragraph) | single seed, original estimator | ablation arms seeds 1–2, current estimator | 5080 job 4 | a_ablation_s12.json |
| Fig. rate panels (a) and (c) | single seed; panel (c) draws an upper bound | regenerate after the B runs; redraw (c) without an upper bound | this machine | src/figures.py |
| tab:compare "A Acc (3 seeds)" column and "seeds 1 and 2 in progress" notes (§6.4, Limitations) | pending | 8 published decoders, cohort A, seeds 1–2 (EEGNeX last) | this machine, running, ~16:35 | bd_pipeline_c.json |
| "No published decoder on cohort B" (Limitations) | untested | 8 published decoders on cohort B, 3 seeds | 5080 job 3 | bd_cohort_b.json |

## Final and no longer red

- Cohort C comparison: ours 0.796 (ECE 0.061); trained decoders 0.723–0.783 (ECE 0.077–0.083); Holm-corrected differences significant only against ShallowFBCSPNet (p = 0.044) and TSception (p < 0.001); EEGNet, Deep4Net and EEG-TCNet failed to train on cohort C.
- Cohort A seed 0 comparison: ours 0.884; published 0.823–0.871.
- Cohort A three-seed results: align + gate 0.868 ± 0.026; alignment +0.026 (p = 0.22); head −0.001 (p = 1.00).
- Cohort A slow rate at seed 0: −0.085, lower in 7 of 7, p = 0.016.

## Removed or rewritten (not red, because known to be wrong or out of scope)

- ATCNet and pipeline F numbers, the old comparison table and Fig. efficiency (user decision; pipeline F is a different training pipeline).
- The backbone screening paragraphs (they reported the screen and ATCNet's failure to run in it).
- "Cohort B's band is empty", the empty-band explanation of the residual, and the claim that the upper bound is bracketed: the drift timescale was not measured (tau_drift_session.json is marked invalid).
- Parameter-efficiency claims against published decoders (EEG-TCNet and FBLightConvNet are smaller).
- tab:norm caption "best for none of seven, worst for four": the computed counts are best for 2 and worst for 2 of six.
- tab:noise: align + gate marked as containing the transformer (it does not); pipeline F baseline spreads removed.

Audit: `python src/audit_overnight_numbers.py` checks 83 printed values against results/.
