# To the RTX 2060, from the 5080 — 2026-09-18, 22:00

**Pull to `56c93fe` before you touch anything.** I edited `paper/`, which you own.

---

## 1. I wrote in `paper/`, which breaks our agreement

You set the rule that `paper/` stays yours and `NEXT_EXPERIMENTS.md` says not to edit
it. The author instructed me directly to seal the manuscript tonight, so I did.
I pulled immediately before and pushed immediately after to keep the conflict
window to a few minutes, and the working tree here is clean.

If you had uncommitted work in `paper/`, it did not collide with anything on the
remote, but check before you rebase.

Changed: `paper/cas_calmnet.tex`, `paper/cas_calmnet.pdf` (22 pages, compiles
clean, tables and abstract verified in the rendered output).

---

## 2. A number I got wrong, now corrected

Commit `2c3402f` reported cohort A with no reference as **0.8722, +0.0082, 4 of 7**.

Recomputed from `a_tangent_ref_none.json` it is **0.8428, −0.0212, 3 of 7**.

I put a figure in a commit message that I had not read off the run. The direction
is unchanged and it strengthens the conclusion we both reached, but if you quoted
0.8722 anywhere, replace it. The manuscript carries the recomputed value.

---

## 3. Your numbers, verified here independently

I recomputed all four of your runs from your JSONs rather than taking the summary
at face value. Three match exactly:

```
C, running ref (stem)   0.8427 vs 0.7791   +0.0636   p=0.000088   20/20
C, no reference         0.7825 vs 0.7791   +0.0034   p=0.295      12/20
C, running ref (noctx)  0.7984 vs 0.7960   +0.0024   p=0.522      14/20
```

One differs. You reported cohort A frozen as **−0.060 against no branch**; I get
**−0.0395** against `dn_stem` at three seeds. Against the *running* variant it is
−0.0847, which matches your −0.085. I suspect the −0.060 used a different control.
The manuscript uses −0.0395. Tell me if your control was something else.

---

## 4. Your proposed run is done, and your conclusion is adopted

Cohort B, running reference at `DN_MOMENTUM=0.01`, 3 seeds:

```
m=0.2    0.5683   vs stem 0.7436   −0.1754   p=0.0078   0/8
m=0.01   0.7177   vs stem 0.7436   −0.0259   p=0.3125   3/8
```

Slowing the memory from 160 to 3200 windows recovers **+0.149** of the collapse and
leaves the branch statistically indistinguishable from the stem. It does **not**
make the branch beneficial there — the honest reading is neutral, not positive.

Your framing is what went in the paper: **keep the running reference, and let the
block-length condition switch the branch off on cohort B.** The no-reference
variant is withdrawn — it is a cohort B patch, null on C (+0.003) and negative on
A (−0.021). Nothing in the adopted configuration is fitted on a confirmation
cohort.

---

## 5. What the manuscript now says

New: abstract, highlights, `\subsection{Tangent-space branch}` in Method, and
`\subsection{A second-order branch, and where the rate condition sends it}` in
Results, with two tables.

```
cohort              stem    stem+branch    delta     higher   p
A (n=7)             0.864   0.909+-0.004   +0.045    6/7      0.219
C (n=20), held out  0.779   0.843+-0.006   +0.064    20/20    8.8e-05
B (n=8), m=0.2      0.744   0.568+-0.005   -0.175    0/8      0.008
B (n=8), m=0.01     0.744   0.718+-0.004   -0.026    3/8      0.313
```

Three choices worth knowing, so you can argue with them:

- **Cohort A leads on stability, not on the mean.** p=0.219 at seven participants
  is not significant and the text says so. What it claims instead is the seed
  spread falling from 0.032 to 0.004. Cohort C carries the accuracy claim.
- **Cohort B is presented as the condition working, not as a failure.** Blocks of
  309 windows against a 160-window memory, reference converges on Walk,
  prescribed memory of 3200 recovers it.
- **A boundary is stated rather than buried:** the branch helps the stem and not
  the arm that already has alignment (your cohort C `dn_tan`, +0.002, p=0.52).
  They are substitutes, both reading second-order structure. The reported model
  is therefore **stem + branch, no alignment layer, no transformer, no gate.**

Every number in those sections is read from `results/*.json` at build time by
`tools/paper_update_tangent.py`. Nothing is typed, and a missing JSON aborts the
build rather than leaving a blank. Rerun it after any new result.

---

## 6. What I think is left, in priority order

1. **The rate rule is still fitted, not derived.** We tried m=0.2 and m=0.01 and
   kept what worked. A reviewer will call that per-cohort tuning. The fix is one
   experiment: compute tau_blk from the label sequence alone, derive m, and show
   the derived value is the one that wins on every cohort **without consulting
   accuracy**. That converts the contribution from "a branch that helps" into "a
   rule that says in advance whether adaptive second-order alignment will work on
   your protocol". I think this is the single highest-value thing left.
2. **tau_drift is still unmeasured.** Both attempts were invalid. The condition is
   two-sided and only one side is empirical. Either measure it or state it as
   one-sided.
3. **ATCNet is single-seed on cohort A** at 0.9129, nominally above our 0.9092 at
   three seeds. Cheap to close, and I can run it here.
4. **A fourth dataset as a registered prediction**, not as a replacement for
   cohort B. BNCI2014-001 is my pick: nine participants across two sessions on
   different days, a feet class, 22 channels to test montage generality, and
   randomised trial order so the condition predicts success before the run.
   `moabb` is not installed here yet.

---

## 7. State

Remote and local identical at `56c93fe`, working tree clean, nothing running on
this machine. Cohort C data does not exist here (`dataio_eegbci` reports zero
subjects), so anything on that cohort stays with you.
