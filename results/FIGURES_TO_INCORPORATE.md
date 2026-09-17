# Three mechanism figures — ready to paste

For the machine that owns `paper/`. The PDFs are committed; this file carries
the captions, placement and the claims each one supports or replaces. Nothing
in `paper/` has been touched.

All three are CPU-only, regenerate from a script, and use real recorded data
(`ds007788` via `dataio.build_epochs`, `sha256(X) = 2d80fc70…`, 2821 × 60 × 400,
9 sessions; cohort B from the MoBI loader). Nothing is simulated or schematic.

---

## A note on figure count first

The manuscript already carries 7 figures at 16 pages. These three make 10,
which is a lot for a paper whose claim list has shrunk. If only one goes in,
**it should be `fig_class_tracking`**: it is the only figure of the mechanism
the paper is actually about, and every other figure shows an outcome. The
topography is the second priority because it is the only direct evidence
against the motion-artefact objection.

---

## 1. `fig_class_tracking.pdf` — full width

**Place in** the rate-condition section, before the momentum sweep table.

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../results/fig_class_tracking.pdf}
\caption{The running covariance estimate tracks whichever class is currently
streaming. Top row: session and class covariances projected to two dimensions
by multidimensional scaling on the affine-invariant Riemannian distance, with
the estimate's path drawn through that space in recording order and coloured by
the streaming class; crosses mark the two class covariances. Bottom row:
distance from the estimate to each class covariance against window index, with
the background shaded by the streaming class at window resolution.
(a,b) Cohort A, $\tau_{\mathrm{blk}}=18$, $m=0.2$: the estimate never
approaches either class and the two curves never cross. (c,d) Cohort B,
$\tau_{\mathrm{blk}}=309$, $m=0.2$: the estimate begins near
$\Sigma_{\mathrm{stop}}$, migrates into the walk region, and the two curves
swap. (e,f) Cohort B at $m=0.01$: the migration collapses. The estimate is the
layer's own internal state, which depends only on the input and the momentum
and not on any trained weight, so this is computed without training and
verified against a live layer to $2.4\times10^{-7}$.}
\label{fig:tracking}
\end{figure*}
```

**Supports:** the central claim, which currently has no figure. **Quantitative
form:** 0 crossings (cohort A), 2 (cohort B fast), 1 (cohort B slow).

**Do not claim** that the slow setting fixes cohort B. It reduces the tracking;
alignment still costs $-0.031$ there against the no-alignment arm.

---

## 2. `fig_topography.pdf` — full width

**Place in** the limitations or the preprocessing discussion, next to the
motion-artefact paragraph citing Castermans and Kline.

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../results/fig_topography.pdf}
\caption{What the alignment layer does on the scalp, cohort A, one
participant, first three sessions, montage from the recorded channel names.
(a) Channel power before alignment. (b) Per-channel gain of
$\mathbf{M}^{-1/2}$: the transform amplifies frontocentral midline sites
(Fz 15.3, FCz 13.4) and suppresses the temporal channels most exposed to jaw
and neck muscle activity (T7 3.3, T8 3.3). (c) Walk minus Stop power. The
central decrease is the direction expected for mu and beta event-related
desynchronisation; gross movement artefact would raise power during walking
rather than lower it, so the sign of this contrast argues against an artefact
account without settling it.}
\label{fig:topo}
\end{figure*}
```

**Supports:** the ERD rationale for the power stem, and it is the only direct
evidence in the paper bearing on the artefact objection.

**Do not overclaim.** One participant, and the sign of a power contrast is not
an artefact rejection. The honest sentence is that it is *consistent with* ERD
and *inconsistent with* gross artefact, not that artefact is excluded. The ICA
control (`icactl_ica.json` 0.785 against `icactl_orig.json` 0.836) belongs in
the same paragraph.

---

## 3. `fig_manifold.pdf` — full width, or drop

**Place in** the drift-mechanism section, replacing or beside the per-
participant drift table.

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../results/fig_manifold.pdf}
\caption{Session covariances as points, positioned by multidimensional scaling
on the affine-invariant Riemannian distance and joined in recording order, with
a shared scale across panels. One participant, nine sessions recorded weeks
apart. (a) Raw. (b) The momentum-0 control, which whitens by a fixed estimate:
identical to (a), as affine invariance requires. (c) Aligned at $m=0.2$: mean
pairwise distance falls from 8.00 to 4.17. The control moving not at all while
adaptation halves the spread is what separates this from the tautology that
whitening reduces covariance distance by construction.}
\label{fig:manifold}
\end{figure*}
```

**Supports:** the answer to the reviewer's "the drift validation is close to
tautological". The null control is the whole argument and it now reads exactly
null.

**This is the one to cut if space is short** — the per-participant table
carries the same numbers.

---

## Regenerating

```bash
cd src
python fig_class_tracking.py          # figure 1
python fig_topo_manifold.py           # figures 2 and 3
```

CPU only, a few minutes each, no GPU contention with any queue.

---

## Two corrections these figures forced, which affect nothing already published

1. The trajectory replay first used `exp_drift.cov`, which does not
   trace-normalise each window, while `AdaptiveAlignment._cov` does. The
   estimate differed by 7.4 %. Fixed; now matches a live layer to 2.4e-07.
2. The manifold first used the layer's estimator, which left the control 3 %
   off instead of null, and MDS auto-scaled each panel so the collapse was
   invisible. Both fixed.

Neither affects any number already in the manuscript.
