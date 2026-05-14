# V3 Company Research State — Decision Boundary Plan (Claude)

Author: Claude Code (planning daemon)
Date: 2026-05-14
Purpose: Define the decision boundary, schema sketch, and staged plan for the
v3 vertical slice. v3 introduces a persistent **CompanyResearchState** that
sits between source ingestion and memo generation, so the memo becomes one
output view over a durable, cited, reusable research artifact.

Inputs reviewed:
- `docs/research_platform/implementation_map.md`
- `docs/research_platform/module_plans/CONSOLIDATED_PLAN.md`
- `docs/research_platform/module_plans/V1_DECISION_BOUNDARY_CHECKLIST.md`
- `docs/research_platform/module_plans/V1_CLAUDE_BOUNDARY_GRILL.md`
- `docs/research_platform/README.md`
- `docs/research_platform/xiaohongshu_librarian_2026-05-12/librarian_xhs_extracted_report.md`
- `src/research_platform/memo_run.py`
- `src/research_platform/prompts/memo_user.md`
- `src/vault/ingest.py`
- `src/vault/memo_flow.py`
- `src/harness/context_types.py`
- `src/vault/contracts.py`
- `tests/unit/research_platform/test_memo_run.py`

## 1. Where v1/v2 stand today

The v1 vertical slice (already coded under `src/research_platform/memo_run.py`)
is intentionally a **one-shot, run-scoped pipeline**:

```text
ResearchTask -> deterministic retrieval -> vault ingest
  -> render source_index.md + question_list.md
  -> render MemoContextPackage JSON (run-scoped)
  -> assemble memo_user.md prompt + context files
  -> existing harness -> final.md + manifest + status
```

Persistent durable state today is limited to:

- `vault/companies/<TICKER>/sources/...` — raw originals + extracted text
- `documents` + `document_companies` + `companies` SQLite rows
- `vault/companies/<TICKER>/research/source_index.md` (last-run snapshot)
- `vault/companies/<TICKER>/research/question_list.md` (last-run caveats)
- `vault/companies/<TICKER>/research/context_packages/<run_id>.json`
- `vault/companies/<TICKER>/research/memos/<run_id>/` — final.md/manifest/status

What is **not** durable across runs today:

- analyst-usable synthesized company picture (business, guidance, risks,
  thesis, valuation, catalysts) — the v1 source_index is metadata only;
- open questions that should outlive a single retrieval pass — current
  `question_list.md` is "deterministic caveats from this batch", not a
  living question backlog;
- conflicts between sources/runs;
- thesis state, position-relevant claims, expectation tracking;
- diffs between runs (what changed about the company);
- LLM-derived candidate claims (deferred from CONSOLIDATED_PLAN §6 v2).

v1 made one deliberate, Ted-approved choice: **no canonical
`llm_research_state.md`**. The v2 lifecycle plan in CONSOLIDATED_PLAN §1.3 was
meant to replace that file with source-grounded structured objects rendered to
markdown views. v3 as Ted has now described it is the first vertical slice
that pulls that lifecycle work into the user-facing flow: the memo writer
*reads* the durable state, the run *updates* the durable state, and the next
run starts from that state instead of starting from scratch.

## 2. v3 product proposition in one paragraph

Given a user investment prompt, a company, and a batch of source materials,
AlphaSeeker first synthesizes / updates a durable **CompanyResearchState** for
that company — a cited, reusable analyst working file that holds business
summary, guidance, risks, thesis, key metrics, open questions, and conflicts
— then runs the existing memo harness *over* that state to produce the final
memo. The state persists. The next run reads it, updates it where new
evidence demands, and produces a new memo. The memo is a derived view; the
research state is the durable product object.

Sketch of v3 flow:

```text
ResearchTask + manual files
  -> retrieval (reuse v1 adapters)
  -> deterministic vault ingestion (reuse v1)
  -> CompanyResearchStateSynthesizer
       reads prior state (if any) + new DocumentRefs + open questions
       writes new CompanyResearchState (JSON + rendered Markdown view)
       produces ResearchStateDelta (what changed, why, with evidence)
  -> MemoContextPackage now points at the state, not raw documents
  -> existing memo harness consumes state as primary context
  -> final.md  +  CompanyResearchState updated on disk
  -> next run picks up where this left off
```

## 3. Decision boundaries — what `CompanyResearchState` owns vs not

This is the central architectural question. Get this wrong and the artifact
becomes either a "freeform wiki blob we cannot trust" (what
`llm_research_state.md` was) or a "claims DB nobody reads" (premature v2
lifecycle).

### 3.1 What CompanyResearchState owns

- A **company-level** durable view of the most current research picture:
  business summary, segment/unit economics, key guidance, risks, thesis,
  catalysts, open questions, and a small set of headline metrics.
- Section-level **citations** back to vault `DocumentRef`s (with quote
  snippets and grade carried through).
- A **provenance stamp** per section: extractor (LLM model+version or
  deterministic+rule), run_id, ingested-at, last-updated-at.
- A **candidate vs accepted** mark per claim/section. LLM-synthesized claims
  default to `candidate`; promotion to `accepted` is an explicit human (or
  rule-based) act.
- An **open_questions** list that survives across runs, with question_id,
  status, evidence_ref, last-seen run.
- A **diff log** of what changed in the state on this run (section, prior
  text/claim, new text/claim, triggering evidence, decision).
- A **freshness/staleness** marker per section (e.g. last evidence date,
  whether a more recent source has been ingested without the section being
  rerun).
- An optional **valuation snapshot** block (point-in-time, B-grade vendor
  numbers, *not* a model). The Librarian post separates "today's price" from
  "should the model change." v3 should make the same split (see §7).

### 3.2 What CompanyResearchState does NOT own

- **Raw source originals or extracted text** — those remain in the vault
  document store. State holds only `DocumentRef`/`EvidenceRef` pointers and
  snippets.
- **The investment memo itself** — `final.md` is an output view derived from
  state plus user prompt plus harness reasoning.
- **The deterministic source_index/question_list snapshot per run** — those
  remain as `memos/<run_id>/source_index.md` for reproducibility. State is
  *current-truth*; snapshots are *frozen-at-run*.
- **The analyst's investment thesis as commitment** — the human owns the
  call. The state can carry a `thesis_candidate` section that the user reads
  and either edits or accepts; it must not silently equal the user's
  position.
- **Cross-company portfolio state** — v3 is per company, not per portfolio.
- **The model** (analyst spreadsheet / valuation model) — out of scope; a
  later workflow can introduce a `model_registry`. State may *reference* one.
- **Daily/automated information flow** — that is the v4 "continuous
  workflows" line; v3 is one-run-at-a-time.
- **Conflict adjudication as truth** — conflicts are recorded as
  `ConflictRecord`s the human resolves. The LLM may propose a resolution as
  a candidate.

### 3.3 How it differs from existing artifacts

| Concern | `source_index.md` | `question_list.md` | `documents` table | `MemoContextPackage` | `CompanyResearchState` (v3) |
|---|---|---|---|---|---|
| Scope | Per-run snapshot | Per-run caveats | Per-document | Per-run prep | **Per-company, durable** |
| Lifetime | Frozen at run | Frozen at run | Append-only | Frozen at run | **Updated each run** |
| Owner | Retrieval/ingest | Retrieval/ingest | Vault registry | Orchestrator (run) | **Synthesis layer** |
| Content | Doc metadata | Missing-source warnings | Originals + checksums | Selected refs + caveats | **Synthesized claims + questions + risks + thesis_candidate, all cited** |
| Cite source? | n/a (is the index) | n/a | Yes (canonical) | Yes (references docs) | **Yes — every claim has at least one EvidenceRef** |
| Memo reads? | Indirect | Indirect | Indirect | Direct | **Primary context** |
| Memo writes back? | No | No | No | No | **Yes (as proposed deltas)** |

This is the crucial change: today, the memo harness *generates* prose that is
not persisted in any structured form. In v3, the memo harness *consumes*
durable structured state and *proposes* deltas back to it. The memo prose is
optional cherry on top; the state is the cake.

## 4. Proposed minimal v3 schema (artifact-first)

The v3 boundary policy: stay artifact-first (JSON + Markdown), do not
introduce new SQLite tables. Reuse existing `documents`/`companies`. This
matches v1's discipline (V1_DECISION_BOUNDARY_CHECKLIST §7) and keeps v3
reversible.

### 4.1 On-disk layout

```text
vault/companies/<TICKER>/research/
  state/
    research_state.json        # canonical machine-readable state
    research_state.md          # rendered analyst view
    diff_log.jsonl             # append-only diffs across runs
    open_questions.json        # durable open questions across runs
    conflicts.json             # durable conflict ledger
  context_packages/<run_id>.json  # unchanged (v1)
  memos/<run_id>/
    final.md, manifest.json, status.json,
    source_index.md, question_list.md   # frozen-at-run, unchanged (v1)
    state_snapshot.json        # NEW: snapshot of research_state.json
                                # as the memo saw it, for reproducibility
    proposed_state_delta.json  # NEW: candidate deltas the memo proposed
```

Rationale for the snapshot+delta split:

- `state_snapshot.json` is the "what the LLM read" record. Required for
  reproducible memos.
- `proposed_state_delta.json` separates "what the LLM proposed" from "what
  the system accepted." Per CONSOLIDATED_PLAN §3.3, LLM must not silently
  mutate durable truth. The post-run merge of deltas into `research_state.json`
  is a separate, auditable step (rule-based or explicit human accept).

### 4.2 `research_state.json` shape (v3)

Keep it small and concrete. Loosely modeled on the Librarian post's 14-section
structure, but pruned to a v3 minimum.

```text
{
  "state_id": "state-<ticker>-<created>",
  "ticker": "XOM",
  "company_name": "Exxon Mobil",
  "schema_version": 1,
  "created_at": "...",
  "last_run_id": "...",
  "last_updated_at": "...",
  "freshness": {
    "newest_evidence_at": "2026-05-01",
    "stale_sections": ["valuation"]      # rule-flagged, not auto-mutated
  },
  "sections": {
    "business_overview":    {<SectionBlock>},
    "key_guidance":         {<SectionBlock>},
    "key_metrics":          {<SectionBlock>},
    "risks":                {<SectionBlock>},
    "thesis_candidate":     {<SectionBlock>},   # explicitly candidate, not truth
    "catalysts":            {<SectionBlock>},
    "valuation_snapshot":   {<SectionBlock>}    # point-in-time, B-grade
  },
  "open_questions": [<QuestionRecord>],
  "conflicts": [<ConflictRecord>],
  "evidence_index": {
    "<evidence_id>": <EvidenceRef>            # dedupe across sections
  },
  "metadata": {
    "synthesizer_model": "claude-opus-4-7",
    "synthesizer_version": "v3.0",
    "input_document_ids": ["doc_..."],
    "carried_question_ids": ["q-..."]
  }
}
```

`SectionBlock`:

```text
{
  "section_id": "business_overview",
  "claim_status": "candidate" | "accepted",
  "summary_markdown": "Short markdown paragraph(s), one section worth.",
  "claims": [
    {
      "claim_id": "c-...",
      "statement": "Upstream segment contributed 70% of FY2025 operating earnings.",
      "field_kind": "quantitative" | "narrative" | "mixed",
      "claim_status": "candidate",
      "extraction_method": "llm" | "deterministic" | "manual",
      "source_grade": "A" | "B" | "C" | "unknown",
      "evidence_refs": ["<evidence_id>", ...],
      "extractor_model": "claude-opus-4-7",
      "extractor_version": "v3.0",
      "extraction_run_id": "memo-xom-...",
      "confidence": 0.0-1.0 | null,
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "last_updated_at": "...",
  "last_updated_run_id": "...",
  "warnings": []   # e.g. "single B-grade source", "evidence older than 90d"
}
```

Notes:

- `claims` is **the** payload. `summary_markdown` is a rendered prose view
  the memo can quote, but the contract truth is the claim list. This avoids
  the `llm_research_state.md` mistake of putting freeform prose at the
  canonical layer.
- Every claim **must** carry at least one `evidence_ref`. Match the existing
  `ClaimRecord._evidence_required` validator in `src/vault/contracts.py:144`.
- LLM-extracted quantitative claims **must not** default to `accepted` — keep
  the existing v0 contract rule (`contracts.py:149-151`).
- `confidence` is optional; do not put it on the critical path in v3.

### 4.3 Rendered `research_state.md` shape

A small, predictable Obsidian-readable view, not a freeform wiki. Suggested
template (v3, intentionally short):

```markdown
# <TICKER> Research State

Last updated: <iso>   Last run: <run_id>

## Freshness
- Newest evidence: <date>
- Stale sections: <comma list or "none">

## Business overview
<summary_markdown>
Cited claims:
- <statement> [S1, S3]
- ...

## Key guidance
... (same shape)

## Key metrics
... (same shape, prefer quantitative claims)

## Risks
... (same shape)

## Thesis (candidate)
... (same shape; flagged "candidate; not analyst-confirmed")

## Catalysts
... (same shape)

## Valuation snapshot (point-in-time, B-grade)
... (same shape; show as-of date)

## Open questions
- [ ] (q-...) Question text — last seen: <run_id> — evidence: <ref or none>

## Conflicts (open)
- (conflict-...) Summary — between S1 and S3 — severity: medium
```

Citations use the run's `S1, S2, ...` keys mapped through
`evidence_index` to `EvidenceRef`s, so the prose layer is auditable against
the JSON.

### 4.4 `diff_log.jsonl`

One JSON object per line, append-only:

```text
{
  "diff_id": "diff-...",
  "run_id": "memo-xom-...",
  "created_at": "...",
  "section_id": "key_guidance",
  "change_type": "added" | "updated" | "removed" | "promoted" | "demoted",
  "before": {<ClaimRecord or null>},
  "after":  {<ClaimRecord or null>},
  "triggering_evidence": ["<evidence_id>", ...],
  "decision": "auto_accepted" | "candidate_recorded" | "human_accepted" | "human_rejected",
  "rule_id": "..."  # which rule made the decision, if any
}
```

This is the audit trail. Cheap to write, easy to inspect, no DB.

### 4.5 `open_questions.json` and `conflicts.json`

Per CONSOLIDATED_PLAN §1.3, lifecycle objects are first-class. v3 promotes
the run-scoped `question_list.md` into a durable `open_questions.json` with
lifecycle states (`open`, `proposed_close`, `answered`, `rejected` —
existing `QuestionStatus` literal). Run-scoped question_list.md continues to
exist as a frozen snapshot per run.

Update semantics:

- A new run reads `open_questions.json` and includes still-`open` items in
  the memo context.
- A new run may propose closing a question (`proposed_close` with evidence
  ref) or add new open questions.
- Auto-close is **off by default** in v3 (see §6, Q4). Closure is either
  rule-based ("A-grade evidence with matching key directly answers question")
  or human.

### 4.6 Citation requirements

Hard rule: **no claim without at least one evidence_ref**. Soft rules:

- Quantitative claims should have at least one A-grade source.
- Single-source B/C claims must carry a `warnings` entry on the section
  block.
- Snippet length cap (e.g. 600 chars) so state stays grep-able.
- `evidence_index` is keyed by `evidence_id` and dedupes evidence used across
  multiple sections.

## 5. v3 read/write protocol

To keep boundaries tight, write down who can read and who can write each
artifact.

| Artifact | Readers | Writers |
|---|---|---|
| `research_state.json` | synthesizer (read prior), memo harness (read snapshot), human | synthesizer (full rewrite of `state.json` is fine; deltas tracked in `diff_log.jsonl`); accept-merge step |
| `research_state.md` | human, memo harness, future workflows | renderer only, never hand-edited |
| `diff_log.jsonl` | synthesizer (to know prior diffs), human, future audit | append-only writes by synthesizer + accept-merge step |
| `open_questions.json` | synthesizer, memo harness, human | synthesizer (propose); accept-merge step (apply) |
| `conflicts.json` | synthesizer, human | synthesizer (propose); human (resolve) |
| `state_snapshot.json` (per run) | reproducibility tooling | orchestrator at memo run start |
| `proposed_state_delta.json` (per run) | accept-merge step, human | memo harness output adapter |
| `final.md` | human | memo harness, frozen at end-of-run |

Two key bottlenecks for safety:

1. **Synthesizer is the only LLM writer to the durable state file** — and
   even it writes through `diff_log.jsonl` + `proposed_state_delta.json` for
   transparency.
2. **Memo harness never mutates durable state** — it can *propose* deltas
   into `proposed_state_delta.json`, which feed into the next run's
   synthesizer or into an explicit accept-merge step. This preserves the
   "harness does not adjudicate truth" rule from CONSOLIDATED_PLAN §1.5.

## 6. Decision questions for Ted

These are concrete, blocking-before-coding questions. They are written as
binary or small-N choices so a reply can be a one-line ruling.

1. **Run-level draft vs company-level durable truth.** Should
   `research_state.json` be (a) **company-level durable truth** that the
   memo run updates in place via deltas (Claude's recommendation, matches
   "living wiki" Librarian model), or (b) **run-level artifact** that is
   regenerated from scratch each run with prior state as input but never
   "owned" across runs? Choice (a) means we maintain identity across runs;
   choice (b) is simpler but loses the diff/freshness story.

2. **Auto-persist vs candidate-only for LLM-derived state.** Should
   LLM-derived synthesizer output (a) **persist automatically as
   `candidate` records** into `research_state.json` (Claude's
   recommendation; matches CONSOLIDATED_PLAN §9 Ted ruling on v1
   LLM-derived claims), or (b) **stay in `proposed_state_delta.json` until
   explicit accept-merge** (stricter, requires either a rule engine or a
   human review step before state mutates)?

3. **Accept-merge step: implicit-rule-based vs explicit human.** If (2)(a):
   what merges deltas into `state.json`? Options: (i) **rule-based
   auto-merge**: "always accept candidate; promote to `accepted` only on
   second confirming A-grade source"; (ii) **explicit accept command**:
   `python -m src.research_platform.accept_state ...`; (iii) **interactive
   review queue** rendered to markdown. v3 is simplest with (i); v3.1 can
   add (ii)/(iii).

4. **Open-question auto-carry vs explicit carry.** Should `open_questions`
   that were `open` in the prior state automatically appear in the next
   run's MemoContextPackage (Claude's recommendation), or only carry forward
   if explicitly re-asked? Auto-carry matches "living wiki" but means an
   un-answered question can persist across many runs until evidence kills
   it.

5. **Question auto-close.** Should a question auto-close to `answered` when
   a new claim with an A-grade evidence ref directly responds to it, or only
   `proposed_close` and wait for human? CONSOLIDATED_PLAN §8 Q5 left this
   open; v3 must pick.

6. **Valuation snapshot in state.** Should `valuation_snapshot` (point-in-time
   B-grade vendor numbers — market cap, EV, PE) live inside
   `research_state.json` or in a sibling `valuation_snapshot.json`? Claude
   leans **sibling file** because it is high-frequency (daily-ish) and the
   rest of state is low-frequency, so co-locating churns diffs needlessly.
   The Librarian post explicitly separates these update cadences.

7. **Strictness of citation requirement.** v3 minimum rule: every claim in
   `state.json` has ≥1 `evidence_ref`. Should we also require: (a) at least
   one A-grade source for any quantitative claim, (b) snippet length cap
   (suggest 600 chars), (c) reject claims whose only evidence is a `manual`
   (B-default) source? Claude recommends (a) yes, (b) yes, (c) no.

8. **Thesis ownership.** Should `thesis_candidate` ever be promoted to
   `thesis` in the state, or stay `candidate` forever and treat the user's
   real thesis as out-of-band (in a memo, in a portfolio file)? Claude
   recommends **stay candidate forever in v3**; promotion to durable thesis
   is a portfolio concern.

9. **Memo input shape.** Should the memo harness receive (a) the *rendered*
   `research_state.md` plus `state.json` as context files (Claude's
   recommendation, mirrors v1's source_index.md + JSON pattern), or (b)
   only the JSON, or (c) only the markdown? (a) gives the LLM both
   readable prose and structured citations.

10. **Memo write-back.** Should the memo harness be allowed to *propose*
    deltas into `proposed_state_delta.json` (Claude's recommendation —
    creates a feedback loop), or stay strictly a consumer in v3 with deltas
    only generated by the synthesizer pass? (a) is more powerful but adds a
    second LLM mutator to track; (b) is simpler.

11. **Schema and migration.** v3 stays artifact-first (no new SQLite tables).
    Confirm: no v3 writes to `questions`, `facts`, `metrics`, `conflicts`,
    or `wiki_versions` tables. Schema migration deferred to v3.1+.

12. **Backward compatibility with v1 `memo_run.py`.** Should the v1
    `run_research_memo()` entrypoint stay callable for the simple
    no-state path (Claude recommends keeping it as the legacy/escape hatch
    until v3 is dogfooded), or be replaced outright by v3 once v3 ships?

13. **Reuse vs new module path.** Where should v3 code live? Options: (i)
    extend `src/research_platform/memo_run.py` with a `--use-state` flag;
    (ii) add `src/research_platform/state_synthesizer.py` +
    `src/research_platform/state_memo_run.py`; (iii) new package
    `src/research_platform/state/`. Claude recommends (iii) — packages
    naturally separate the synthesizer, render, accept-merge, and memo
    runner.

14. **Synthesizer model and budget.** Should the synthesizer use the same
    model as the memo harness (Claude Opus 4.7), or a cheaper one (Sonnet)
    given it is a smaller scoped task with strict JSON output? Claude
    leans toward starting on Opus to maximize cite quality; revisit after
    one demo.

15. **Acceptance gate for v3 demo.** What is the demo target? Options: (i)
    one XOM run that creates new state from scratch + memo; (ii) two XOM
    runs back-to-back showing state carryover and a non-trivial diff; (iii)
    XOM + one other ticker to show isolation. Claude recommends (ii) —
    that is the minimal proof the durable-state idea works.

## 7. Source-grade and freshness handling in state

Carry through v1's A/B/C grading discipline (CONSOLIDATED_PLAN §4, Ted v1
rulings) into state, and add freshness:

- `SectionBlock.last_updated_at` = max(evidence published_at, run created_at)
- A section is `stale` if its newest evidence is older than its
  policy-defined cadence (e.g. valuation: 7 days; guidance: 90 days;
  business_overview: 365 days).
- Staleness is **flagged**, never auto-mutated. The synthesizer reads
  staleness on input and decides whether to re-synthesize; the memo harness
  reads staleness flags so the memo can disclose them.

Valuation cadence specifically (per §6 Q6) is best handled in a sibling
file with its own cadence — daily refresh OK, no diff log entries.

## 8. Update / diff semantics

v3 should make the *change* between runs first-class, not the new state.
The simplest implementation:

1. Synthesizer loads `prior = research_state.json` (or empty).
2. Synthesizer loads new `DocumentRefs` from this run's ingestion + still-open
   questions.
3. Synthesizer outputs a new `proposed_research_state.json` (full state,
   not diff) constrained by the schema.
4. Diff helper computes the delta vs `prior` at the claim level (added,
   updated, removed, promoted, demoted).
5. Diff entries with `decision` per the chosen accept-merge policy (§6 Q3)
   are appended to `diff_log.jsonl`.
6. If the policy says "auto-merge", `research_state.json` is replaced with
   the new state. Otherwise the new state is written only to
   `memos/<run_id>/proposed_state_delta.json` for review.

This keeps the synthesizer's job simple ("write the new state") and the
trust boundary inside the diff/accept-merge step where it belongs.

## 9. Tests and validation gates

Keep test surface modest; v3 should not need 30 new tests. Suggested set:

**Schema and round-trip**
- `research_state.json` round-trips through Pydantic models without loss.
- Every `SectionBlock.claims` validates the `evidence_refs` non-empty
  invariant.
- LLM quantitative claims default to `candidate`, never `accepted`.
- `evidence_index` dedupes evidence_ids referenced from multiple sections.

**Synthesizer plumbing (mock the LLM)**
- Empty prior state + 2 mock documents → state with N sections, each cited.
- Existing prior state + 1 new document → diff_log has 1+ entry, decision
  field present.
- Synthesizer LLM mock returns invalid JSON → run fails cleanly with a
  caveat in `status.json`, durable state untouched.
- Synthesizer LLM mock returns claims without evidence → rejected by
  schema, recorded as warning, durable state untouched.

**Lifecycle**
- Open question with `status=open` in prior state appears in next
  MemoContextPackage when auto-carry is on.
- Auto-close (if enabled) requires an A-grade evidence ref; rule is unit-tested.

**Memo bridge**
- Memo harness receives `research_state.md` + `state_snapshot.json` as
  `context_files`. Existing assemble_memo_prompt template renders without
  unresolved placeholders.
- Memo harness write-back (if §6 Q10 = yes) produces
  `proposed_state_delta.json`; the next synthesizer pass actually reads it.

**Reproducibility**
- `state_snapshot.json` exactly matches the `research_state.json` value the
  synthesizer wrote in this run (modulo timestamps).
- `diff_log.jsonl` is append-only — running the same run with the same
  `run_id` twice raises.

**Live demo gate**
- One mocked-providers XOM run produces a non-empty `research_state.json`
  with at least one A-grade evidence ref in `key_guidance`.
- A second mocked-providers XOM run with an added new document produces a
  non-empty `diff_log.jsonl` entry.

## 10. Staged coding plan (v3)

Following Ted's preference for minimal functional implementation, and the
v1/v2 discipline of contracts-then-implementation.

### v3.0 — Contracts and skeleton (no LLM yet)

- Add Pydantic models in `src/research_platform/state/contracts.py`:
  `CompanyResearchState`, `SectionBlock`, `StateDiffEntry`, etc.
  Reuse existing `ClaimRecord` / `QuestionRecord` / `ConflictRecord` /
  `EvidenceRef` / `DocumentRef` from `src/vault/contracts.py` where they
  already match.
- Add renderer `src/research_platform/state/render.py` that emits
  `research_state.md` from a `CompanyResearchState` instance. Pure function.
- Add diff helper `src/research_platform/state/diff.py` that emits a
  `list[StateDiffEntry]` between two states. Pure function.
- Add filesystem helpers in `src/research_platform/state/storage.py`:
  `load_state(ticker, vault_root)`, `write_state(state, vault_root)`,
  `append_diff(entries, vault_root)`, `read_open_questions`, etc.
- Unit tests for contracts, render, diff, storage. **No LLM calls.**

Validation: golden render test, diff invariants, storage idempotency.

### v3.1 — Synthesizer (LLM in, JSON out)

- Add `src/research_platform/state/synthesizer.py`:
  - input: `prior_state`, `new_documents`, `open_questions`
  - output: `CompanyResearchState`
  - implementation: strict-JSON LLM call against a prompt template at
    `src/research_platform/prompts/state_synthesizer.md`
  - hard validation post-call: every claim has evidence; quantitative LLM
    claims default to `candidate`
- Add accept-merge function with **rule-based default** (per §6 Q3 (i)) so
  v3.1 is self-contained.
- Unit tests with mocked LLM only.

### v3.2 — Wire into a memo run

- Add `src/research_platform/state/memo_run.py` (or extend v1 `memo_run.py`
  with a `--use-state` flag, depending on §6 Q13):
  1. retrieval (reuse v1)
  2. ingest (reuse v1)
  3. synthesizer → new state + diff_log
  4. write `research_state.{json,md}`, `state_snapshot.json`,
     `proposed_state_delta.json`
  5. memo harness consumes state as `context_files`
  6. final.md preserved, manifest gains `research_state_path`,
     `state_snapshot_path`, `diff_log_path`
- Update `memo_user.md` template (or add `memo_user_state.md`) to instruct
  the harness to read state as primary context.
- Unit tests: full mocked-pipeline test mirroring
  `tests/unit/research_platform/test_memo_run.py` shape.

### v3.3 — Demo and tighten

- Run mocked-providers double-run XOM demo (per §9 acceptance gate).
- Optional live XOM demo if Ted approves provider calls.
- Tighten only where the demo reveals real gaps. Do not pre-build accept-merge
  UX, question auto-close ergonomics, or valuation cadence automation here.

### Explicitly out of scope for v3

- New SQLite tables / schema migrations.
- Multi-company portfolio state.
- Daily monitor / cron workflows.
- Excel model registry.
- Conflict adjudication UI beyond the markdown ledger.
- Deck/briefing output variants.
- Cross-company claim sharing.

## 11. Risks and what could go wrong

1. **State turns into freeform Markdown again.** Discipline: `state.json` is
   canonical, `state.md` is rendered. Lint rule: no human edits to `state.md`.
2. **Synthesizer drift between runs.** Two runs on the same evidence
   produce subtly different prose, generating noise in `diff_log.jsonl`.
   Mitigation: pin synthesizer model+version, treat prose-only diffs as
   non-events (compare on `claims` list, not on `summary_markdown`).
3. **Memo harness re-fetches sources and writes uncited prose into final.md.**
   Same risk as v1 (per V1_CLAUDE_BOUNDARY_GRILL §8). Keep `equity` skill
   pack with soft-preference policy; rely on prompt to cite from state.
4. **Question backlog grows forever.** Mitigation: required staleness on
   open questions ("not seen in 5 runs → demote to rejected"); revisit in
   v3.1+ if it becomes painful.
5. **Schema churn.** Use `schema_version` and bump on breaking changes; old
   state files survive as read-only history.
6. **Citation laundering.** LLM cites a snippet that does not exist in the
   referenced document. Mitigation: post-synthesis sanity check that every
   `evidence_ref.quoted_snippet` substring-matches the referenced
   `extracted_text_path`.
7. **State and memo disagree.** Acceptable in v3 — the memo is one
   perspective; the state is durable. Future v4 can introduce reconciliation
   pass.

## 12. Concise summary

v3 introduces a per-company, durable, cited **CompanyResearchState** artifact
that sits between vault ingestion and memo generation. It is artifact-first
(JSON + Markdown + diff log), reuses v1 ingest/retrieval, and turns the
memo harness from a one-shot prose generator into a consumer of structured
state that can optionally propose deltas back. The memo becomes a derived
view; the state becomes the durable product object. Implementation should
proceed in three small slices (contracts → synthesizer → memo wiring) with
mocked-LLM tests at each gate.

## 13. Decision questions for Ted (compact list)

1. Run-level draft vs **company-level durable** state? (Claude: company-level)
2. LLM output **auto-persists as candidate** vs candidate-only-until-merge?
3. If auto-merge: **rule-based** vs explicit command vs human-interactive?
4. Open questions **auto-carry across runs**?
5. Question **auto-close** on A-grade match, or `proposed_close` only?
6. Valuation snapshot **inside state.json** or sibling `valuation_snapshot.json`?
7. Citation strictness: enforce ≥1 A-grade source for quantitative claims?
   snippet length cap? reject manual-only claims?
8. `thesis_candidate` ever promoted to `thesis` in state, or candidate-forever?
9. Memo harness reads **state.md + state.json**, only JSON, or only markdown?
10. Memo harness allowed to **write proposed deltas** back into state?
11. Confirm v3 stays artifact-first (no new SQLite tables)?
12. Keep v1 `run_research_memo()` as legacy entrypoint?
13. v3 code at `src/research_platform/state/` package, or extend `memo_run.py`?
14. Synthesizer model: Opus 4.7 or cheaper Sonnet for v3?
15. Demo target: single run / **two-run carryover** / multi-ticker?
