# Handoff to the RTX 2060

Written 2026-09-19 from the 5080. Branch `representation-beats-architecture`.
Everything below is what a reviewer asked for and the 5080 could not finish.

Pull first. The paper, the deck, the speaking script and the Q&A notes are all
current as of the head of this branch and do **not** need rebuilding unless you
change a number.

---

## Why these are yours

Cohort C's recordings (PhysioNet EEGMMIDB) live on the 2060. The 5080 started
downloading them from PhysioNet and got 70 of 120 EDFs at roughly three files
per two minutes before the download was stopped as redundant. `data/eegbci/` on
the 5080 holds that partial copy; ignore it. Item 1 below is the one that
actually needs your disk.

---

## 1. Riemannian baseline on cohort C  (the highest-value item)

Cohort A is done and in the paper (Section 6.8). Cohort C is not, and cohort C
is the held-out cohort that carries the accuracy claim, so the comparison
matters most there.

```bash
CX_SEEDS=0,1,2 CX_WORKERS=8 CX_THREADS=2 \
CX_COHORT=eegbci CX_OUT=c_riemann.json \
python -u src/exp_riemann_baseline.py
```

Cohort A, for reference, over three seeds and seven participants:

| arm | cohort A |
|---|---|
| `riem_mdm` (MDM, Riemannian) | 0.706 |
| `riem_ts` (tangent space + logistic regression) | 0.793 |
| stem alone | 0.864 |
| stem + branch | 0.909 |

Paired against the reported model: +0.116, higher in 7 of 7, p = 0.016.

**Do not shortcut the Karcher mean.** Fitting it on a subsample of the training
covariances moved one participant's accuracy by −0.012 at 1000 and **+0.014** at
500, non-monotonic and the size of effects this paper interprets. Run it in full
and spread the work over workers. On the 5080 cohort A took about 20 minutes on
8 workers; cohort C has 20 participants and 2 s windows, so budget an hour.

When it lands, the number goes into the paragraph in Section 6.8 that currently
ends "This was run on cohort~A only." Update that sentence and the matching
entry in Limitations.

---

## 2. Holm correction across the five cohort tests  (a decision, not a run)

Table 11 reports per-cohort p-values with no family-wise correction, while other
parts of the paper do use Holm. A reviewer noticed the inconsistency.

Uncorrected, and Holm across the five as the reviewer computed them:

| cohort | raw p | Holm |
|---|---|---|
| C | 8.8e-5 | 0.00044 |
| B | 0.008 | 0.032 |
| D | 0.020 | 0.060 |
| E | 0.047 | 0.094 |
| A | 0.219 | 0.219 |

**Under Holm, D and E no longer cross 0.05.** That is why this is a decision and
not a calculation. Three defensible options:

1. Declare each cohort a separate prespecified hypothesis and say so explicitly.
2. Report corrected values and soften D and E to "directionally consistent".
3. Present the cohort-wise p-values as descriptive and lean on the effect sizes.

Whichever you choose, say it once and apply it everywhere. The paper currently
looks strict in Section 6.9 and permissive in Table 11.

Per-participant accuracies for all five cohorts are in `results/*_3seed.json`
under `per_subject`, so the correction is a two-minute computation once the
policy is chosen.

---

## 3. Median block length against the long blocks  (needs an argument)

Condition 1 uses the **median** single-class block, but Table 1 also reports the
longest:

| cohort | median | longest | memory |
|---|---|---|---|
| A | 18 | 486 | 160 |
| B | 214 | 2211 | 160 |
| C | 5 | 5 | 160 |
| D | 1 | 6 | 160 |
| E | 13 | 33 | 160 |

Cohort A is called safe because 18 < 160, yet it contains blocks of 486, which
exceed the memory. If one long block were sufficient to break the reference,
cohort A would fail, and it does not.

So the mechanism sentence and the statistic do not match, and the paper needs to
say which one it means: typical exposure, occupancy fraction, or repeated long
blocks. `tools/tau_blocks.py` already computes the block sequence, so the
occupancy distribution is available without new runs. This is the objection I
would expect a careful reviewer to press hardest after item 2.

---

## 4. Smaller items, all text

- **Zero-phase filtering is acausal.** Table 2 specifies zero-phase filtering
  and the paper describes an online decoder. The adaptation is causal and the
  normalisation is causal (verified: `exp_globalnorm.normalise` takes its
  statistics from the fitting split alone), but the band-pass is not. One
  sentence acknowledging that the evaluation uses offline preprocessing.
- **Table 13 uses Walk-command language for cohorts C and D**, which are
  rest-versus-movement and left-versus-right hand. FA/min should be named
  generically for those rows.
- **FA/min and missed onsets are not defined precisely enough** for a table
  called "what a wearer would experience": threshold, hysteresis, consecutive
  windows, merging of adjacent positives, onset tolerance. The code is in
  `onset_metrics` in `src/exp_calmnetx.py`.
- **Data availability names no repository.** Add the URL and the commit.
- **Title** overstates the longitudinal claim; only cohort A is longitudinal in
  the strong sense. The reviewer suggested "Tangent-Space Features for
  Cross-Session EEG Decoding Under Blocked Test Streams".
- **Braindecode, Ganin and DECODED citations** need checking against their
  proper records. AdamW and the `[10? ]` artifact are already fixed.

---

## Already fixed on the 5080, do not redo

- `[10? ]` was `\citep{schirrmeister,eegnet}` with no `eegnet` key. Now `lawhern`.
- Parameter arithmetic: 19,505 + 238,028 + 32,896 + **514** (classifier) =
  290,943. The text now states which counts include the classifier.
- AdamW cited Kingma and Ba; added Loshchilov and Hutter.
- "Helps wherever mu exceeds the block length" contradicted Table 12 and is now
  stated as predicting direction and severity, not sufficiency.
- "The off-diagonal entries are discarded" was wrong: a learned spatial filter
  has output power w'Cw. Now "represents that structure implicitly".
- "Held out from every design decision" (3 places), "pre-registered",
  "leads all eight", "nine times more reproducible", and the declared but never
  reported detection latency.
- Inference cost measured and added as Section 6.7 and Table 14.

## State at handoff

18 pages, 101 references, no unresolved cross-references, no em-dashes, abstract
257 words. `python tools/final_check.py` passes all 24 checks and
`python tools/check_layout.py` reports no table crossing a column. Deck is 25
slides; the script and Q&A notes match it.
