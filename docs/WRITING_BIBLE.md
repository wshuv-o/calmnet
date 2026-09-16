# The Jahin Writing Bible (reusable, domain-general)

A distilled, field-independent version of Abrar Jahin's publication system for
writing strong empirical/method papers at Q1/Q2 venues. Originally tuned for BCI /
biomedical papers; the rules below are the parts that transfer to **any** technical
research writing. Field-specific examples are quarantined in the last section so you
can keep or swap them.

> **Calibration first — aim for 80–85%, not 100%.** Jahin applies his own system at
> roughly 70%. Applied at 100%, the prose reads *more disciplined than its author* —
> which itself reads as machine-written. Lean confident, not over-clean. Let the
> author's voice show through. This single calibration note overrides every rule below
> when they conflict.

---

## 1. Sentence-level rules (the ones that do the real work)

1. **Every claim sentence is backed in the same paragraph.** Backing = a citation, an
   equation/table/figure reference, a specific number, or a forward reference to a later
   section. No backing → delete the sentence or move it to the Discussion with a hedge.
2. **Hedge with precision, not vagueness.** Replace "may improve performance" with the
   quantified version: "improves accuracy by *X* points (paired test, *p* = *Y*,
   corrected *α* = *Z*)."
3. **First-mover claims carry scope qualifiers.** Not "the first method for X" but
   "to our knowledge, the first method evaluated on {A} and {B} under {protocol}."
4. **Comparative language is always quantified.** Not "significantly outperforms
   baselines" but "outperforms 9 of 10 benchmarked methods (paired test, *p* = *X*)."
5. **Causal language only in the Discussion, hedged.** Methods and Results use
   associational language ("is associated with", "improves by *X*%"); save causal
   claims for Discussion and hedge them.

---

## 2. Paragraph-level moves

- **Threat-then-response in Related Work.** State the existing approach → acknowledge
  what it does well (one sentence) → state its limitation → describe your response →
  forward-reference where you show it.
- **Every limitation acknowledgment converts to a contribution** ("we address this
  by…") rather than being left as an open problem hanging.
- **Triangulate load-bearing claims.** Aim for four independent supports per important
  claim — e.g. theory + ablation + per-unit breakdown + baseline comparison + external
  citation. A reviewer who knocks out one still faces three.

---

## 3. Structural defaults

- **Abstract: Background → Problem → Solution → Results → Significance**, with bold
  inline labels. Hits hard and stays scannable.
- **Numbered contribution list of 4–6 items**, each starting with an active verb
  ("We benchmark…", "We identify…", "We introduce…"), no bold-prefix labels.
- **Formalization as armor.** Where it fits, name a Lemma / Proposition / Remark for a
  key property. Even a standard result gains rhetorical weight once formalized.
- **The Conclusion is forward-looking, not a summary.** Position the work, name the most
  important open question it raises, and set up future work concretely. Every conclusion
  paragraph must add something not already said.

---

## 4. Phrases to avoid (these read as AI/filler tells)

- **Buzzwords:** *novel, robust, comprehensive, synergistic, seamless, leverage,
  underscore, delve, harness, showcase.* Strip them.
- **Unbacked intensifiers:** *significantly* without a stat test; *outperforms* without
  a number; *first* without a scope qualifier; *state-of-the-art* unless self-disclaiming.
- **Assertion markers:** *we argue / we believe / it is important to note / it is worth
  noting* — these flag an unsupported claim. Back it or cut it.
- **Vague hedges:** *may / might / could enhance* — replace with a quantified hedge or delete.
- **Vacuous transitions:** *moreover, furthermore, additionally* — use a strong topic
  sentence instead.
- **Em-dashes for parenthetical asides in prose** — the biggest single tell. Use
  parentheses or split the sentence.
- **"Rather than"** — easy to overuse; keep it under ~5 in a whole paper.

---

## 5. The meta-skill: knowing when to stop

The hardest discipline is not adding more. Each extra figure, sensitivity check, or
polish pass justifies itself in isolation; the sum is bloat ("a truck with a jet engine,
wings, and a machine gun"). Before any addition, ask: *is this load-bearing, or am I
just generating?* When the honest answer is "it's done, ship it," that is the answer.
Don't draft the fifth revision of an abstract that was fine three rounds ago.

Two corollaries paid for in real time:
- **If your headline claim does not survive its own ablation, change the headline** —
  don't prop up a property the evidence doesn't support. (A well-framed comparison or
  honest negative result can carry a paper that a failed novelty claim cannot.)
- **A reviewer critique you should have written yourself is the most painful kind.**
  Write it before the reviewer does, and put it in Limitations.

---

## 6. Domain-specific moves (example: BCI / biomedical — keep, swap, or delete)

These are concrete instantiations for small-cohort biomedical/BCI papers. Replace with
your field's equivalents.

- **Open with the clinical-stakes statistic** (e.g. ALS, locked-in syndrome, post-stroke
  numbers). A concrete number up front.
- **Per-subject + grand-mean reporting always.** Small-cohort fields gain credibility
  from per-unit tables, not just averages.
- **Parameter-count column** in any architecture comparison — a win at low cost is a
  separate axis to claim on.
- **Multi-seed when the headline margin is small** (< ~3 points). Single seed + small
  margin is the most attackable surface.
- **Disclose best-epoch vs final-epoch evaluation.** Best-epoch is conventional but
  upward-biased on small folds — acknowledge it in Limitations, don't hide it.
- **Cross-corpus replication if at all possible.** A finding that replicates on three
  independent datasets is the hardest thing to attack.
- **Reproduce ≥1 strong peer-reviewed baseline under your own pipeline** to show the
  comparison is fair; landing within ~0.1 point of the reported number is the credible target.

---

*Use at 80–85%. The goal is a disciplined paper that still sounds like a person wrote it.*
