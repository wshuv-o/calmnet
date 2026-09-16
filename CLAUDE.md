# CLAUDE.md — CALM-Net / DriftNet Operating Manual

**Read this fully before the first action of any session.**

Three documents govern this project. They are vendored in `docs/` so they cannot
be lost, and they are binding, not advisory:

| doc | governs |
|---|---|
| [docs/RESEARCH_OPERATING_MANUAL.md](docs/RESEARCH_OPERATING_MANUAL.md) | how the project is run: phases, checkpoints, claim ledger, quality gates |
| [docs/WRITING_BIBLE.md](docs/WRITING_BIBLE.md) | how the prose is written: sentence rules, banned phrases, structure |
| [docs/FIGURE_RULES.md](docs/FIGURE_RULES.md) | how every figure is made |

`docs/tifs_style_reference.py` is the reference implementation of the figure
rules from a sibling project; `src/figstyle.py` is this project's version of it.

---

## 0. Project brief

- **Deliverable:** a journal manuscript on an EEG walk/stop intent decoder with
  in-network session adaptation, calibration and abstention.
- **Venue & format:** Applied Soft Computing (Elsevier). Template: **CAS**
  (`cas-dc.cls`, double column), vendored in `paper/`. A secondary IEEEtran
  version exists for coursework.
- **One-sentence thesis:** a test-time adaptation layer tracks whatever is
  currently streaming, so its adaptation rate must exceed the protocol's
  class-block timescale and stay below its drift timescale; where those cross,
  no valid rate exists and the layer should be omitted.
- **Data:** cohort A = OpenNeuro ds007788 (7 subjects x 9 sessions, exoskeleton);
  cohort B = MoBI treadmill (8 subjects x 3 trials). Results in `results/*.json`.
- **Success:** every number traceable to a JSON in `results/`; the paper states
  something specific and defends it.

---

## 1. Hard rules for this project

1. **No number enters the manuscript that is not in `results/`.** If two numbers
   exist for the same model, report both and state why they differ.
2. **Never compare across training harnesses.** ATCNet scores 0.913 in harness C
   (`exp_calmnetx.py`) and 0.881 in harness F (`fullbench.json`). The 0.032 gap
   is larger than most effects of interest. Groups are reported separately with
   a rule between them.
3. **Never compare across estimator versions.** The per-window covariance fix
   landed 2026-09-16 05:16; runs before and after are not comparable.
4. **Noise floor is not uniform.** Arms containing the cross-epoch transformer
   carry ~0.010 run-to-run spread; the stem-only arm is bit-identical across
   repeats. Differences below 0.02 in transformer arms are not interpreted.
5. **Single-seed results are marked as single-seed**, in the table and the
   caption.
6. **If the headline claim does not survive its own ablation, change the
   headline.** It has already happened twice here.

---

## 2. Figures

Every figure regenerates from `src/figures.py`, which imports `src/figstyle.py`.
Never restyle a figure in its own function. Before saying a figure is fixed,
**render it and look at the rendered image** — collisions are invisible in source.

Enforced by `src/figstyle.py`: serif matching the CAS body font, Okabe-Ito
palette, one series = one colour + one marker + one dash across the whole
document, our method visually privileged, no in-axes titles, panel letters in
the corner, vector PDF with `pdf.fonttype = 42`, figures generated at the exact
measured column width (COL = 3.30 in, FULL = 6.84 in for `cas-dc`).

---

## 3. Prose

Applied at ~80–85%, per the bible's own calibration note. The rules that bite
hardest here:

- Every claim sentence is backed **in the same paragraph** by a citation, a
  number, or a table/figure reference.
- Comparative language is always quantified. Never "outperforms" without a
  number and a test.
- **Banned:** novel, robust, comprehensive, synergistic, seamless, leverage,
  underscore, delve, harness, showcase; "significantly" without a test; "we
  believe / it is worth noting"; "moreover / furthermore / additionally".
- **Em-dashes for parenthetical asides are the single biggest tell.** Use
  parentheses or split the sentence.
- Abstract structure: Background → Problem → Solution → Results → Significance.
- Contribution list: 4–6 numbered items, each opening with an active verb.
- The Conclusion is forward-looking, not a summary.
- Per-subject **and** grand-mean reporting; parameter-count column in every
  architecture comparison.
- Write the reviewer's critique yourself and put it in Limitations.

---

## 4. Build

```bash
# figures (regenerates all seven as PDF + PNG)
cd src && python figures.py

# the paper
cd paper && ../tools/tex/tectonic.exe -X compile cas_calmnet.tex --outdir .
```

`tools/tex/tectonic.exe` is a self-contained LaTeX engine (gitignored, 19 MB);
re-download from the tectonic GitHub releases if missing. There is no system
LaTeX on this machine and `winget` fails on a certificate error.

---

## 5. Current state

Model: **align + gate**, 24 181 parameters. Cohort A 0.901 (seed 0, harness C)
against stock ATCNet 0.913 in the same harness. The architecture does **not**
transfer: on cohort B the bare stem beats the full model 0.737 to 0.626.

The paper therefore makes no accuracy claim. It claims parameter efficiency,
calibration and abstention outputs the baselines lack, a validated drift
mechanism (52% / 84%, 15/15 subjects), and the two-sided rate condition.
