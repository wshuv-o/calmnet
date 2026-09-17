# Red numbers in cas_calmnet.tex and what replaces them

Every value wrapped in `\pending{}` renders red. Each is listed here with the run
that will replace it. When all are final, change the macro in the preamble to
`\newcommand{\pending}[1]{#1}` (one line) and delete this file.

Updated 2026-09-17 16:15. Eight red items remain.

| Red item (where) | Replaced by | Machine | Result file |
|---|---|---|---|
| Component ablation table (tab:ablation), Fig. ablation note, cross-epoch context paragraph (0.023; ECE 0.039 to 0.074; 1.64 to 1.26 FA/min; 23 %) | Ablation arms seeds 1-2, current estimator: dn_stem and dn_noalign (running, started 15:40, ~20:00-20:30); dn_full and dn_nogate | RTX 2060; RTX 5080 queue job 4 | a_ablation_s12_2060.json; a_ablation_s12.json |
| Rate section note and tab:ratecurve row for memory 3200 | align + gate at m = 0.01, three seeds | RTX 5080 queue job 2 | b_aligngate_m001_3seed.json |
| Fig. rate note: panel (c) draws an upper bound, panel (a) single seed | Redraw from src/figures.py | this machine | results/fig_rate.pdf |
| Limitations: published decoders on cohort B | Eight decoders on cohort B, three seeds | RTX 5080 queue job 3 | bd_cohort_b.json |

## Final (black) since the last update

- Cohort A comparison over three seeds (EEGNeX seeds 1-2 from the 5080, bd_eegnex_s12.json): published decoders 0.817-0.878, ours 0.868; no paired difference significant (EEG-TCNet +0.010, EEGNeX +0.008 above ours; others 0.003-0.051 below, smallest p = 0.22). ECE ours 0.058, published 0.058-0.076.
- Cohort B three-seed rate table (tab:ratecurve): cost of alignment 0.183 / 0.125 / 0.068 at memory 160 / 320 / 1600; slowing from 160 to 1600 recovers 0.115. Used in the abstract, slow-adaptation section, replication paragraph, Discussion items 1-2, Limitations and Conclusion.
- Single-seed items that no planned run replaces are now black and labelled single seed: tab:rate, tab:external, Fig. inversion, the seed-0 values 0.792 and 0.626.
- Earlier today: cohort C comparison; drift 45.9 % and 62.5 % with a null control; drift reduction against rate; cohort A slow rate over three seeds (−0.050).

## Removed or rewritten

- ATCNet and pipeline F; backbone screening paragraphs; parameter-efficiency claims.
- "Cohort B's band is empty", the bracketed upper bound, fitted tau_drift values (both invalid).
- tab:norm caption counts corrected; tab:noise transformer flag corrected.

Audit: `python src/audit_overnight_numbers.py` checks 105 printed values against results/.
