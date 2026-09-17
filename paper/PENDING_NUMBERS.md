# Red numbers in cas_calmnet.tex and what replaces them

Every value wrapped in `\pending{}` renders red. Each is listed here with the run
that will replace it. When all are final, change the macro in the preamble to
`\newcommand{\pending}[1]{#1}` (one line) and delete this file.

Updated 2026-09-17 16:05.

| Red item (where) | Current value | Replaced by | Machine | Result file |
|---|---|---|---|---|
| Cohort B accuracies (abstract, §6.5 text and Table 12, Learnability, Discussion items 1–2, Limitations, Conclusion) | 0.792, 0.211, 0.581, 0.626, 0.166, 0.092 | align + gate against gate, 3 seeds, current estimator | 5080 queue job 1 | b_gate_aligngate_3seed.json |
| Rate recovery on cohort B (abstract, §5.4 noise, §6.8, tab:rate, prereg paragraph, Conclusion) | +0.090 to +0.098, −0.010 | slowed alignment m = 0.01, 3 seeds (5080); m = 0.02, 3 seeds (other session, partial); m = 0.10 is final | 5080 job 2; other session | b_aligngate_m001_3seed.json; other session's files |
| Residual remark (§6.9) | 0.060, 0.723, 0.783 | same B runs | 5080 job 2 | same |
| Discussion item 1 drift percentages (60 % against 25 %), measured with the earlier normalisation | 60 %, 25 % | rerun of the momentum sweep with drift_fixed measurement, or remove the numbers | other session | — |
| "Error in either direction" and seed multiples (§5.4, §6.9, Limitations) | 0.060 to 0.092; 2.3 to 3.8 | recomputed from the B and A three-seed runs | both | both |
| Component ablation (tab:ablation, Fig. ablation, cross-epoch context paragraph) | single seed, original estimator | ablation arms seeds 1–2, current estimator | 5080 job 4 | a_ablation_s12.json |
| Fig. rate panels (a) and (c) | single seed; panel (c) draws an upper bound | regenerate after the B runs; redraw (c) without an upper bound | this machine | src/figures.py |
| tab:compare "A Acc (3 seeds)" column and "seeds 1 and 2 in progress" notes (§6.4, Limitations) | pending | 8 published decoders, cohort A, seeds 1–2 (EEGNeX last) | this machine, running, ~16:35 | bd_pipeline_c.json |
| "No published decoder on cohort B" (Limitations) | untested | 8 published decoders on cohort B, 3 seeds | 5080 job 3 | bd_cohort_b.json |

## Final and no longer red

- Cohort C comparison: ours 0.796 (ECE 0.061); trained decoders 0.723–0.783 (ECE 0.077–0.083); Holm-corrected differences significant only against ShallowFBCSPNet (p = 0.044) and TSception (p < 0.001); EEGNet, Deep4Net and EEG-TCNet failed to train on cohort C.
- Cohort A seed 0 comparison: ours 0.884; published 0.823–0.871.
- Cohort A three-seed results: align + gate 0.868 ± 0.026; alignment +0.026 (p = 0.22); head −0.001 (p = 1.00).
- Cohort A slow rate at seed 0: −0.085, lower in 7 of 7, p = 0.016.
- Cohort B three-seed rate curve, current estimator (b_gate_aligngate_3seed.json dn_noctx, b_floor_m0.10.json, b_floor_m0.02.json; no-align reference averaged over its six runs, 0.763): alignment costs 0.183 at memory 160 (7 of 8, p = 0.016), 0.125 at 320 (7 of 8, p = 0.023), 0.068 at 1600 (6 of 8, p = 0.15). Gradual recovery, no step at 309, never beneficial. Replaces 0.792, 0.581, 0.211 in the abstract, external validation, limitations and conclusion.
- Drift reduction against rate, cohort A (drift_momentum.json): 45.9, 26.8, 12.1, 2.3, 0.2 % at m = 0.2, 0.1, 0.05, 0.02, 0.01. In Discussion item 1, the slow-adaptation section and the residual remark.
- Session drift (rerun without trace normalisation, drift_fixed.json): 45.9 ± 4.9 % (A, t = −9.46, p = 8e−5) and 62.5 ± 12.2 % (B, t = −12.33, p = 5e−6), 15 of 15; control changes distance by < 0.1 %. Fig. 2 and tab:drift regenerated from it.
- Cohort A slow rate on align + gate, three seeds (a_slow_noctx_3seed.json): 0.819 ± 0.017 against 0.868 ± 0.026, cost −0.050, lower in 7 of 7, p = 0.016; per seed −0.085, −0.056, −0.008.

## Removed or rewritten (not red, because known to be wrong or out of scope)

- ATCNet and pipeline F numbers, the old comparison table and Fig. efficiency (user decision; pipeline F is a different training pipeline).
- The backbone screening paragraphs (they reported the screen and ATCNet's failure to run in it).
- "Cohort B's band is empty", the empty-band explanation of the residual, and the claim that the upper bound is bracketed: the drift timescale was not measured (tau_drift_session.json is marked invalid).
- Parameter-efficiency claims against published decoders (EEG-TCNet and FBLightConvNet are smaller).
- tab:norm caption "best for none of seven, worst for four": the computed counts are best for 2 and worst for 2 of six.
- tab:noise: align + gate marked as containing the transformer (it does not); pipeline F baseline spreads removed.

Audit: `python src/audit_overnight_numbers.py` checks 95 printed values against results/.
