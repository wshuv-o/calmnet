# Morning briefing — read this first

Prepared overnight for the 10:00 presentation. Everything below is committed and
pushed to `representation-beats-architecture`. The full decision-by-decision
account is in `OVERNIGHT_LOG.md`.

## What to open

| file | what it is |
|---|---|
| `presentation/Updated_manuscript_presentation.pptx` | **15-slide deck for 10:00**, midterm style |
| `paper/cas_calmnet.pdf` | updated manuscript, 20 pages |
| `paper/cas_calmnet_overleaf.zip` | Overleaf package, verified by compiling from scratch |

## The four results that matter

**1. A prediction registered in advance came true, on an independent cohort.**
Before any model touched PhysioNet EEGMMIDB (20 participants), two predictions
were committed with their thresholds (`b7ca694`, 01:59:52). Both held:
enabling alignment cost **+0.002**, and slowing adaptation gained **−0.006**.
On cohort B the same slowing had recovered **+0.090**. Two further seeds agree.
This converts Condition 1 from an explanation into a prediction that has been
tested in advance, on its short-block side.

**2. Is the accuracy neural? Mostly yes, with an honest limit.**
Retraining after ICA artefact removal lowers accuracy by **0.051** (0.835 →
0.785, p = 0.22) and leaves it far above chance. The losses do not follow the
amount of muscle removed: the participant with the most muscle removed
*improves*. With seven participants that is evidence against muscle dependence,
and it falls short of proof. This is the answer to the question the midterm
deck invites.

**3. The selective head does nothing, for ranking or for accuracy.**
Alignment alone reaches **0.883**; with the head, **0.884**. **This reverses my
advice from last evening**, when I said the head should stay because it seemed
to add +0.016. That figure came from mixing estimators. On this evidence the
head could be removed.

**4. The methods section had four errors, now fixed.**
The submitted paper stated a 0.5-40 Hz band and Adam at lr 10⁻³. The code uses
**8-30 Hz** and **AdamW at 3×10⁻⁴**, plus EOG regression the paper called "no
artefact step". None changes a number; each would have defeated reproduction.

## Still running when this was written

Cohort A, gate against align + gate, three seeds each. Align + gate ends
~06:05 and gate ~07:40. It removes "single seed" from the headline. See the
last section of `OVERNIGHT_LOG.md` for whether it finished and was integrated.

## Decisions only you can make

1. **Remove the selective head?** The data now supports it (result 3). It
   needs a re-run without the head and changes to the architecture and figure.
2. **Headline estimator.** The paper leads with 0.901 (original estimator). The
   reviewer plan asks for the trace-normalised estimator throughout, which gives
   0.884. The ablation is now complete under both.
3. **Abstract format.** Structured headings were removed on the editorial
   advice. Check whether Applied Soft Computing requires them.

## Things you should know before presenting

- **Cohort B's data is not on this machine.** Its results are real but came
  from the other computer. Nothing on cohort B could be re-run tonight.
- **E: was not writable** from this account, even without restrictions, so
  everything went on C:. 3.85 GB of regenerable caches were removed to make
  room; the list is in `cache_cleanup_0203.txt`.
- **tau_drift is not in the paper.** The decoder-free measurement, as designed,
  is confounded by class switching (it returns cohort A's block length). A valid
  design is in `TODO_NEXT.md`.
- **I made and corrected errors overnight**, all recorded in the log: a
  mixed-estimator ECE comparison (caught by a scripted audit, now 45/45 values
  verified), two sentences broken by an earlier mechanical rewrite, and several
  log timestamps I had guessed (now reconciled against git).

## Likely questions at 10:00

- **"The midterm said the accuracy was confound. Why 0.901 now?"** The midterm's
  leakage measure scored R² = 0.863 on a decoder carrying no movement
  information; it was confounded with accuracy and was withdrawn. Slide 8 gives
  the current evidence.
- **"Isn't the rule post hoc?"** It was, on cohorts A and B. Slide 12: it has now
  predicted a third cohort correctly in advance, as the registration commit's
  timestamp shows.
- **"Does alignment help on the new cohort?"** No, and it doesn't hurt. The test
  predicted the absence of harm, and that is what was found.
