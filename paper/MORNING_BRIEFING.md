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

## The five results that matter

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

**5. Alignment's gain on cohort A is smaller than the submitted paper said.**
The submitted paper credits alignment with **+0.074** on cohort A. That was one
seed, and the comparison also removed the context module. The clean comparison,
align + gate against gate with nothing else changed, finished at 07:12 over
three seeds: **+0.026** (0.868 against 0.842), higher in 6 of 7 participants,
**p = 0.22**. Per seed it is +0.067, +0.029 and −0.019, so it changes sign once.
sub-04 gains 0.179 and sub-05 loses 0.127, each on every seed. More seeds would
not make this significant, because sub-05's consistent loss caps the test at
seven participants, so no further runs were started.

The paper now says this in four places (ablation, cross-cohort paragraph,
limitations, conclusion). Read together with cohort C (+0.002 registered,
+0.014 on replication seeds) the picture is consistent: **where the band holds,
alignment gains a little; where it fails, it costs 0.211.** The conclusion
now presents the condition as a design-time guard against that loss.

## Nothing is still running

Every queued job finished. The GPU is idle.

## Decisions only you can make

1. **Remove the selective head?** The data now supports it (result 3). It
   needs a re-run without the head and changes to the architecture and figure.
2. **Headline estimator.** The paper leads with 0.901 (original estimator). The
   reviewer plan asks for the trace-normalised estimator throughout, which gives
   0.884. The ablation is now complete under both.
3. **How to frame alignment.** The conclusion now calls the condition "most
   useful as a design-time guard" against the 0.211 loss, because the gain
   where the band holds is small (result 5). That is my wording; change it if
   you prefer another emphasis, but keep the +0.026 and p = 0.22.
4. **Abstract format.** Structured headings were removed on the editorial
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

- **"How much does alignment actually add?"** On cohort A, +0.026 over three
  seeds (p = 0.22), smaller than the single-seed +0.074 on slide 9, which now
  shows both. Its value is clearest on cohort B, where applying it against the
  band costs 0.211, and the band says so before training. Slide 7.
- **"The midterm said the accuracy was confound. Why 0.901 now?"** The midterm's
  leakage measure scored R² = 0.863 on a decoder carrying no movement
  information; it was confounded with accuracy and was withdrawn. Slide 8 gives
  the current evidence.
- **"Isn't the rule post hoc?"** It was, on cohorts A and B. Slide 12: it has now
  predicted a third cohort correctly in advance, as the registration commit's
  timestamp shows.
- **"Does alignment help on the new cohort?"** No, and it doesn't hurt. The test
  predicted the absence of harm, and that is what was found.
