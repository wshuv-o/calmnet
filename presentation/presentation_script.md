# CALM-Net final presentation: speaking script

Deck: `CALM-Net_final_presentation.pptx`, 24 slides, about 19 minutes in total.

## Slide 1. Title  (about 20 s)

Good morning. We are Md Wahiduzzaman Suva and Esme Moula Chowdhury Abha, and our supervisor is Dr. Muhammad Hasibur Rashid Chayon.

Our work is on tangent-space features for EEG movement-intent decoding, for the control of a lower-limb exoskeleton.

## Slide 2. Problem Statement  (about 45 s)

A convolutional EEG decoder reduces each window of signal to log band power. That quantity is the diagonal of the window's spatial covariance matrix.

Everything off that diagonal, which is how the channels vary together, is discarded before the first layer of the network ever sees it.

Our question is whether that discarded structure carries usable movement intent, and what it costs to read it.

We answer it on five cohorts and 51 participants, in one training pipeline.

## Slide 3. Key Challenges  (about 45 s)

There are three difficulties.

First, the second-order structure is thrown away by construction. Whether that loses anything is an empirical question, and it had not been answered for this task.

Second, reading the off-diagonal needs a reference covariance. If we update that reference on the test stream it follows the session, which is what we want, but it also follows whichever class is currently streaming, which is not.

Third, a component that helps on four cohorts and fails on the fifth is only useful if we can say in advance which case a new recording falls into.

## Slide 4. Objectives and Outcomes  (about 50 s)

Our first objective was to read the covariance rather than only its diagonal. The outcome is a tangent-space branch placed beside the log-power stem. It raises balanced accuracy on four of the five cohorts.

The second was to state when the adapting reference is admissible. The outcome is Condition 1, which compares the estimator's memory against the length of a class block. Read from the protocol alone, it separates all five cohorts with nothing fitted.

The third was to report what the component costs. The branch adds 238,028 parameters to a 19,505-parameter stem path, and on one cohort it triples the rate of spurious walk commands. We report both.

## Slide 5. Literature Review  (about 55 s)

Four bodies of work meet here.

Lower-limb exoskeleton BCI gives us the application and the datasets.

Convolutional EEG decoders, such as EEGNet and ShallowFBCSPNet, are the models we compare against.

Riemannian methods and alignment are where the tangent-space idea comes from. The limitation of that line is that the reference is estimated over a complete recording, so the recording has to exist before the decoder can run on it.

Test-time adaptation removes that constraint, but the vision literature has shown it degrades when the test stream is temporally correlated.

The gap is that no prior work states, from the recording protocol alone, when an in-network adapting reference is safe to use.

## Slide 6. Decoder Architecture  (about 55 s)

This is the reported model, 290,943 parameters in total.

A window of EEG, 60 channels by 400 samples, enters both paths.

The upper path is the multi-scale power stem. Three temporal resolutions, 64, 128 and 256 milliseconds, each with depthwise spatial filters, then square, pool and logarithm. That gives log band power per channel, which is the diagonal.

The lower path is the tangent-space branch. It forms the window's covariance, shrinks it toward a scaled identity, and maps it to the tangent space at a reference matrix M. That reads the off-diagonal.

The two 128-dimensional outputs are concatenated and fused into one linear classifier.

The dashed arrow is the important one. The reference M updates from the incoming data with no labels, and it keeps updating at inference.

## Slide 7. Core Idea and Hypothesis  (about 55 s)

The branch computes the logarithm of M to the minus one half, times the covariance C, times M to the minus one half, and vectorises it.

M is a running mean over the incoming windows. That is what lets the representation follow a session as it changes.

The momentum m sets how long that mean remembers. The memory is mu equals N over m, which is 160 windows at our default batch of 32 and momentum of 0.2.

And that is also the failure mode. If one class streams for longer than the memory, then M becomes the covariance of that class, and whitening a window of that class by its own covariance removes exactly the variance that separated the classes.

So Condition 1: the memory must be longer than a class block and shorter than the drift timescale. Both sides are fixed before training.

## Slide 8. Datasets and Evaluation Protocol  (about 55 s)

Five cohorts, 51 participants.

A and E are lower-limb exoskeleton cohorts. B is treadmill walking from an independent laboratory. C is motor execution, 20 participants. D is BCI Competition four, dataset 2a.

Every model is trained in the same pipeline: the same band, the same optimiser, the same early stopping, the same temperature scaling, and three data-split seeds.

The table gives the quantity Condition 1 needs. The class-block length is computed from the label sequence alone, before any training: 18 windows on cohort A, 214 on B, 5 on C, 1 on D and 13 on E.

Notice that cohort B is the one whose blocks are longer than our 160-window memory.

## Slide 9. Artefact Control  (about 55 s)

Walk and stop differ in movement, so a fair objection is that we are decoding muscle activity rather than brain activity.

We retrained the reported model after independent component analysis removed ocular, muscle, cardiac and line-noise components. Both arms are restricted to the same task, so they differ only in the cleaning step.

Balanced accuracy goes from 0.897 to 0.859. That difference is not significant, p equals 0.58, and four of the seven participants are unchanged or better.

The mean is carried by one participant, sub-06, which falls by 0.207. Without it the change is 0.010.

Importantly, the change does not follow how much muscle activity was removed. The correlation is plus 0.05, where dependence on muscle artefact would make it negative.

This constrains the artefact account. It does not exclude it, and we say so.

## Slide 10. Scalp Topography of the Reference  (about 50 s)

This is the branch's own reference, not a separate analysis.

Panel b is the per-channel gain that the tangent map applies. It amplifies the frontocentral midline most, AFz and Fz at 2.9, and it amplifies the temporal sites least, T7 and T8 at 1.9. Those are the sites most exposed to jaw and neck muscle.

Panel c is the band-power difference, walk minus stop. Power falls over left sensorimotor cortex, C3 by 3.5 decibels, and rises over occipital electrodes.

A decrease over sensorimotor cortex is the sign and the location of mu and beta desynchronisation. Gross movement artefact would raise power during walking, not lower it.

This is one participant, so it is supporting evidence and not an artefact rejection.

## Slide 11. Learned Representations  (about 50 s)

This figure is measured rather than drawn. We trained a model, then used forward hooks to record the real activations at the four points the architecture names, on held-out windows.

Panels a to c fold the tangent vector back into the 60 by 60 matrix it came from. You can see the structure sitting off the diagonal, which is precisely what the log-power stem discards.

Panel d measures class separation at each tap. The branch reaches 1.22, the stem 0.74, and after fusion it is 2.41.

Fusion being higher than either path alone is the point: the two representations are complementary inside the trained model, not only in the accuracy table.

## Slide 12. Comparison with Published Decoders  (about 55 s)

This is the table the rest of the results build toward.

The reported model and eight published decoders, all trained in the same pipeline with the same splits, the same seeds and the same metric code. No per-model tuning.

The reported model is highest on four of the five cohorts.

On the fifth, cohort B, it is the worst entry in the column. We are not going to hide that; we are going to explain it, and the explanation is the next part of the talk.

What makes cohort B a usable result rather than an unreliable one is that the stem alone still leads that column. The failure is specific to the branch, not to the pipeline.

## Slide 13. Branch versus Stem Across Cohorts  (about 50 s)

Here is the branch against the stem it is added to, on every cohort, three seeds throughout and paired within participant.

Plus 0.045 on cohort A, plus 0.064 on C, plus 0.040 on D, plus 0.021 on E.

And minus 0.175 on cohort B, lower in eight participants out of eight.

The regime column is the one to watch. It is not fitted to the accuracy. It is read from the class-block length against the estimator memory, before training, and it marks cohort B as at risk and the other four as safe.

## Slide 14. Results on the Held-Out Cohort  (about 60 s)

Cohort C carries the accuracy claim, because it was held out from every design decision. The architecture, the shrinkage, the momentum and the embedding width were all fixed on cohorts A and B before this cohort was ever run.

Three things make the number credible, and the three panels give one each.

Panel a: all 20 participants improve, from 0.779 to 0.843, a mean of plus 0.064.

Panel b: the whiskers are each participant's own spread across the three seeds. The effect is larger than that spread in 16 of the 20, so for most participants the gain is bigger than the noise the measurement already carries.

Panel c puts 0.843 in context. Against the eight published decoders on this same cohort, the branch leads the strongest of them by 0.060, higher in 18 of 20 participants. And the stem it is added to sits in the middle of that field.

Calibration error also falls, from 0.075 to 0.062, so the accuracy is not bought with a worse posterior.

## Slide 15. Effect Size Against Class-Block Length  (about 45 s)

Now the failure, and why it is predictable.

Panel a is the change from the branch on each cohort. Panel b plots that change against the class-block length, on a log axis.

Nothing here is fitted. The block length comes from the recording protocol and the memory comes from the optimiser settings, and the two were fixed before training.

The dashed line is the estimator memory, 160 windows. The four cohorts whose blocks are shorter than that memory all gain. The one cohort whose blocks are longer, cohort B at 214 windows, is the one that fails.

One boundary, no free parameter, and it separates all five.

## Slide 16. Direct Manipulation of the Estimator Memory  (about 50 s)

A boundary across cohorts could still be a coincidence of five datasets, so we tested it within a cohort by moving the memory directly and changing nothing else.

On cohort A, where blocks are 18 windows: plus 0.045 at a memory of 160, plus 0.048 at 40, and minus 0.076 at 8. At a memory of 8 windows, shorter than the block, the effect reverses, and it is worse in every participant.

On cohort B we go the other way. Enlarging the memory from 160 towards 3200 moves the cohort back monotonically, and it recovers 74 per cent of the loss.

So the same variable switches the effect off and on in both directions. That is causal support for the account, not just a correlation across datasets.

## Slide 17. Estimator Memory and Block Length: Summary  (about 45 s)

This is the same evidence as a table, and it is also where we report a negative result about our own claim.

The condition gives an ordering and a direction. It does not give a rate a practitioner could set.

We pre-registered a stronger version, which sets the adaptation rate from the block length with a fixed margin of two. It was falsified: at the memory that prescription gives, cohort B reaches 0.650, against 0.744 for the stem alone. Better, but still short of parity.

And the upper bound of the condition, the drift timescale, is not measured here. Our data bound it only from above, at about one session.

## Slide 18. Ablation Study  (about 45 s)

Every row here is a choice the architecture actually makes.

Adding the branch takes cohort A from 0.864 to 0.909, and it cuts the seed-to-seed spread by about nine-fold, from 0.032 to 0.004.

Freezing the reference after fitting costs accuracy. Removing it altogether also costs accuracy.

The lower table separates the two roles of that reference across cohorts. The adapting reference is the only setting that helps on both cohorts whose class blocks are short, so that is what we keep, and we let Condition 1 decide where it is admissible.

## Slide 19. Measurement Noise and Ranking Criteria  (about 40 s)

Before ranking anything we need to know what our own noise is.

Across three data-split seeds the reported model spans 0.007 to 0.022 depending on the cohort. The stem spans 0.011 to 0.057.

The stem is the noisier arm on all five cohorts, so the branch is not buying its accuracy with a less stable fit.

We therefore do not interpret any difference below 0.02 anywhere in this work, and differences smaller than the relevant seed spread are reported without a ranking.

## Slide 20. Control-Relevant Outcome Measures  (about 50 s)

Balanced accuracy is not what a wearer experiences, so this table reports what a controller would act on.

Calibration error improves on four of the five cohorts, and the missed-onset rate falls on four.

But look at cohort A. Spurious walk commands rise from 0.70 to 2.09 per minute of standing, even though accuracy went up.

Those two are different functions of the same posterior, and on a device that moves someone's legs the second one is what throws a wearer off balance.

It is a real cost of the branch, and we report it rather than the accuracy alone.

## Slide 21. Results on Cohorts D and E  (about 45 s)

These are the two cohorts added last, against the same eight published decoders.

The branch leads all of them on both: 0.741 against 0.682 on cohort D, and 0.949 against 0.896 on cohort E.

The stem alone also clears them, so the ordering does not depend on the branch. What the branch adds is the margin.

Cohort D needs a caveat that the margin itself hides. Five of its eight decoders are left near chance on at least one seed, which its roughly 100 fitting windows per participant explain. Only the three that trained properly are load-bearing, and all three hold after correction for multiple comparisons.

## Slide 22. Limitations  (about 50 s)

Four, honestly.

The block-length boundary was found after the fact. It separates all five cohorts with nothing fitted, but it was identified after cohorts A, B and C had been run. Only its application to D and E was prospective.

The artefact control rests on seven participants, and average re-referencing is part of the cleaning. It constrains the artefact account; it does not exclude it.

Our cohorts are small, 7, 8, 20, 9 and 7, and the between-subject standard deviation is large relative to the differences we are measuring.

And the branch is not compact. 238,028 parameters against a 19,505-parameter stem path, so we make no parameter-efficiency claim.

## Slide 23. Conclusion  (about 45 s)

The covariance structure that a log-power stem discards does carry movement intent. Reading it raises balanced accuracy on four of five cohorts, and on 20 of 20 participants of a cohort held out from every design decision.

Where it fails, it fails predictably. The estimator memory against the class-block length separates all five cohorts with nothing fitted, and moving that memory switches the effect off and on.

The condition is the part worth building on, and it is also the part least finished. It gives no rate a practitioner could set, and its upper side has never been measured.

The experiment that would settle it is small and specific: choose a protocol whose class blocks fall between 18 and 214 windows, register the prediction in advance, and run it once.

Thank you. We are happy to take questions.

## Slide 24. References  (about 15 s)

These are the works cited on the slides, keyed to the author-and-year tags used throughout.
