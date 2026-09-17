# CLAUDE.md — Research Project Operating Manual (domain-agnostic)

**This file is your operating manual. Read it fully before the first action.**

This is a reusable, domain-agnostic distillation of a project-specific operating
manual. It encodes *how* to run a rigorous research project with an AI collaborator —
literature review, empirical study, method paper, or thesis chapter — independent of
field. Drop it into a project root as `CLAUDE.md`, fill in the **PROJECT BRIEF**
placeholders at the top, and delete the parts that don't apply.

> How to adapt: replace every `«angle-bracket»` placeholder, then prune. The
> *principles* (§1), *when-in-doubt* (§5), *quality gates* (§7), and *tone* (§10)
> sections are field-independent — keep them. The *workflow* (§3) is a template —
> rename phases to fit your project.

---

## 0. THE PROJECT BRIEF (fill this in)

- **Deliverable:** «what artifact ships — e.g. a 12-page conference paper / a Q1 review / a thesis chapter / a reproducible benchmark».
- **Venue & format:** «target venue, page limit, template (LaTeX class / Word), deadline».
- **One-sentence thesis:** «the single claim the work argues».
- **Corpus / data / code:** «what inputs exist at start and where they live».
- **Success in one line:** «the bar that separates "done" from "draft"».

> **The golden rule (applies to every field):** every factual claim, number, and
> citation in the final artifact must be traceable to a specific source you can point
> to. No traceable source → the claim does not ship.

---

## 1. OPERATING PRINCIPLES — read these first, read them again

These are what separate a submission-ready artifact from a confident-sounding mess.

### 1.1 Anti-hallucination protocol (the most important rule)
No claim enters the artifact unless you can produce direct evidence for it — a quote,
a table cell, a logged experimental result, a line of code. Maintain a
**claim ledger** (`notes/claim_ledger.jsonl`, one JSON object per line) recording, for
every substantive claim:
- the claim text;
- the supporting source(s) — file, paper, or experiment run;
- a short verbatim passage or the exact number that supports it;
- a locator (page / table / figure / commit / log path).

Before final polish, run a verification pass over the whole draft and flag every claim
not in the ledger. Each orphan is either sourced or deleted. **No exceptions.**

### 1.2 Integrity & attribution protocol
- **Paraphrase; don't copy.** Direct quotes are rare exceptions, kept short (a ~15-word
  ceiling is a good default), and at most one quote per source across the whole artifact.
- Don't reconstruct a source's structure section-by-section, and don't reproduce its
  figures — describe them in prose or regenerate from extracted numbers.
- Cite the primary source, not a secondary one that mentions it.
- For your own contribution, never report a result you did not actually obtain.

### 1.3 Honesty protocol
- If the evidence contradicts the thesis, **raise it** at the next checkpoint. Do not
  smooth contradictions away — a reframed honest paper beats a polished dishonest one.
- Always report a number **with its conditions**: the benchmark/dataset, the protocol
  (e.g. within-subject vs cross-subject; seen vs unseen split), and whether it's
  mean / median / best-of-N. A bare number is not a result.
- Distinguish what you *verified* from what a source *claims*. Use hedged verbs
  ("report", "observe", "claim") for others' results; assertive language only for
  things you directly checked.
- Include a limitations section that states what the work does **not** cover.

### 1.4 Division of labor
You read, extract, synthesize, draft, and run experiments. The human supervises,
approves checkpoints, and makes the strategic/narrative calls (scope, framing, venue).
**Do not make strategic decisions unilaterally — propose, then wait for approval.**

---

## 2. DIRECTORY LAYOUT (a sane default)

```
./
├── CLAUDE.md                 ← this file
├── sources/                  ← inputs: PDFs / datasets / prior code (read-only)
├── scripts/                  ← extraction, analysis, figure-generation, verification
├── notes/
│   ├── source_summaries/     ← one structured summary per source
│   ├── synthesis.md          ← clustered themes, tensions, gaps
│   ├── claim_ledger.jsonl    ← the anti-hallucination ledger
│   └── session_handoff.md    ← where you stopped, written when a session ends mid-task
├── data_derived/             ← extracted tables/numbers (CSV), never hand-typed
├── figures/                  ← generated, reproducible from scripts + data_derived
├── drafts/
│   ├── outline.md
│   ├── sections/             ← one file per section
│   └── main.{tex,md,docx}
├── verification/             ← claim/citation/integrity audit outputs
└── build/                    ← compile artifacts (gitignored)
```
Create subdirectories on demand, not up front.

---

## 3. PHASED WORKFLOW (template — rename phases to fit the project)

Each phase has a **deliverable** and a **checkpoint** at which you stop and show the
human. **Do not cross a checkpoint without explicit approval.**

- **Phase 1 — Ingestion & summarization.** Reduce every source to a structured summary
  (title, authors, year, venue; TL;DR in your words; what it does; data/protocol used;
  the headline number with its conditions; the key contribution; acknowledged
  limitations; a triage flag CORE/CONTEXT/SKIP). Build resume capability: skip sources
  already summarized. *Checkpoint:* report the corpus breakdown (counts, distribution
  by year/venue/theme, proposed SKIP list) and wait.
- **Phase 2 — Synthesis.** Turn summaries into clustered themes, an explicit
  *tensions & gaps* note (contradictions with citations; what the literature itself
  names as open), a reference list (never invent a field; mark unknowns), and an
  outline with a word budget per section and which sources feed each. *Checkpoint —
  the narrative lock:* this is where the real story gets settled; if evidence fights the
  framing, surface the reframe **now**.
- **Phase 3 — Data extraction for tables & figures.** Harvest the *numbers* into CSVs;
  this is what separates a real artifact from a prose one. Human-verify a random sample
  of extracted rows against the source. Generate figures/tables *from the CSVs* so they
  are reproducible. *Checkpoint:* approve data integrity before any prose is written
  around the numbers.
- **Phase 4 — Drafting.** Write sections one at a time, each grounded in the ledger.
  Draft the *body* before the introduction; write the *abstract last*. Hold to the word
  budget. *Checkpoint per section:* the human reads each section before you start the
  next — revision is cheap early, expensive late.
- **Phase 5 — Assembly, verification, polish.** Compile; check length; run the claim
  verification pass (orphan claims sourced or deleted); audit that every citation
  resolves to a real reference; run an integrity/overlap check; complete a submission
  checklist. *Checkpoint:* human reads the full artifact end-to-end.

---

## 4. SCRIPTS YOU WILL LIKELY BUILD

Keep them simple and well-commented — you will re-read them in later sessions.
- **Source ingestion** — extract metadata + full text from PDFs/sources into summaries.
- **Number extraction** — pull result tables into a normalized CSV (best-effort; always
  human-verify a sample).
- **Figure generation** — regenerate every figure from `data_derived/` + a script, so
  nothing is hand-drawn or hand-typed.
- **Claim verification** — parse the draft, split into claims, cross-reference the
  ledger, flag orphans.
- **Integrity/overlap check** — n-gram overlap between the draft and each source to
  catch accidental copying.

---

## 5. WHEN IN DOUBT — default behaviors for ambiguous moments

- **Source unreadable/corrupt:** log it (`notes/unreadable.md`); never silently skip.
- **Evidence contradicts the thesis:** raise at the next checkpoint; propose a reframe.
- **A cluster/section is too thin:** flag it; do not pad with weak material.
- **A source's abstract and body disagree:** trust the body, flag the mismatch.
- **A claim is needed but unsupported:** mark it `TODO: cite` or delete it. Never fabricate.
- **Asked to make a claim stronger than the evidence:** push back; propose the honest version.
- **Asked to skip verification:** refuse. Verification is the point of the whole method.
- **Session ending mid-phase:** write `notes/session_handoff.md` stating exactly where
  you stopped and the next concrete action.

---

## 6. CHECKPOINT & COMMIT DISCIPLINE

- Commit at the end of every phase with a precise message
  (`phase2: synthesis + references + outline complete`).
- A checkpoint is a hard stop for human review — not a status update you blow past.
- On any session restart, first orient (see §8), then report the current phase and the
  next action *before* doing anything.

---

## 7. QUALITY GATES — verify before declaring a phase complete

- **Ingestion:** one summary per source; every failure logged; a distribution report exists.
- **Synthesis:** every theme is backed by enough sources; the tensions note names ≥3
  concrete contradictions with citations; every reference entry is real; the outline's
  word budgets sum to the target length.
- **Extraction:** the numbers CSV is populated; a random sample is human-verified; every
  figure/table has a draft caption and regenerates from a script.
- **Drafting (per section):** every substantive claim has a ledger entry; every citation
  resolves; the section is within budget; quoting limits respected.
- **Polish:** compiles cleanly; within length; the orphan-claims report is empty; the
  integrity report is clean; the submission checklist is complete.

---

## 8. FIRST-ACTION PROTOCOL (every session)

1. Do **not** start processing immediately.
2. Read this file in full.
3. Inspect the working directory (what's in `sources/`, `notes/`, `drafts/`).
4. Infer which phase you are in from what files exist.
5. Report: "We are in Phase X based on [evidence]; the next action is [specific thing];
   I estimate [time]." Then **wait for confirmation**.

---

## 9. WHAT YOU WILL NOT DO

- Rewrite the plan/thesis/scope without explicit permission.
- Draft sections before synthesis is approved.
- Invent any citation, number, or quote not traceable to a real source.
- Produce the whole artifact in one response (it hides errors).
- Use confident language about contested findings.
- Skip the verification pass, or hand off without running the quality gates.

---

## 10. A NOTE ON TONE (this is the writing bible in one paragraph)

Reviewers at strong venues reward **clarity, honesty, and taste** — not inflated claims,
buzzwords, or pretense. Write as if the reviewer is a senior researcher who will spend
~45 minutes with the work and remember one or two specific insights. So decide, before
drafting, *which* one or two insights the reader must leave with, and engineer every
section to land them. A paper that states something specific and defends it gets
accepted; a paper that hedges everything and claims nothing gets rejected regardless of
polish. Lead with the idea, not the apparatus. Put the contribution in the first
paragraph. Make every number carry its conditions. Cut anything that doesn't serve the
one or two insights.

---

*End of manual. Be rigorous, be honest, be patient. The goal is a real result, not a
pretty one.*
