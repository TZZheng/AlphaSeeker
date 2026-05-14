# AlphaSeeker Research Platform — v3 Decision Boundary Checklist

Last updated: 2026-05-14 14:25 CDT
Owner: `codex`
Purpose: clear v3 CompanyResearchState boundaries before implementation.

## Status

**V3 design-doc gate is cleared for planning docs.** Ted approved writing design docs after the second Claude Code addendum confirmed the markdown-first / StateOwner direction.

Load-bearing planning reports:

- `docs/research_platform/module_plans/V3_COMPANY_RESEARCH_STATE_BOUNDARY_CLAUDE.md`
- `docs/research_platform/module_plans/V3_MARKDOWN_FIRST_STATE_OWNER_CLAUDE_ADDENDUM.md`

This checklist is the final decision layer over those two reports. The first report is still useful for product boundaries, state-vs-memo separation, tests, and artifact-first discipline. The second report supersedes the first report on the canonical shape of state: **v3 is markdown-first, not JSON-first.**

## Final Ted rulings and accepted defaults

1. **State scope:** company-level durable state, not run-level draft.
2. **Canonical state shape:** markdown-first `research_state.md`, not rigid `research_state.json`.
3. **Structured sidecars:** use JSON/JSONL only for lifecycle invariants: section index, evidence index, open questions, conflicts, valuation snapshot, diffs, and proposal audit.
4. **Valuation:** use a separate `valuation_snapshot.json` from v3, because valuation is long, point-in-time, and time-sensitive.
5. **Open questions:** carry across runs automatically.
6. **Question closing:** do not casually auto-close; use `proposed_close` or a clearly cited answer path.
7. **Citation strictness:** every state update must cite resolvable evidence; quantitative-looking claims require same-sentence citation or warning/revision; snippets are capped.
8. **Write-back boundary:** memo work may propose durable state updates, but durable state is written only through a StateOwner / StateUpdater boundary.
9. **StateDelta mechanism for v3.0:** use D1 post-memo `proposals.jsonl` with small structured envelopes and markdown bodies.
10. **True in-flight tool loop:** defer D2 `propose_state_update(...) -> StateOwner result -> memo continues` until harness support exists.
11. **StateOwner v3.0:** rule-based Python post-pass, not LLM-mediated.
12. **Storage:** artifact-first; no new SQLite tables in v3.
13. **Module location:** new `src/research_platform/state/` package.
14. **Legacy compatibility:** keep existing v1 `run_research_memo()` behavior reachable; v3 state should be opt-in/controlled until proven.
15. **Demo gate:** two-run XOM proof: run 1 populates state from empty; run 2 updates with new evidence; integrity check passes and `diff_log.jsonl` records accepted change.

## Intended v3 vertical slice

Target: first useful durable company research-state loop without over-constraining LLM prose.

```text
ResearchTask / ticker / source bundle
  -> v1 retrieval + vault ingestion + context package
  -> load or initialize per-company state/ folder
  -> snapshot state/ into memos/<run_id>/state_snapshot/
  -> memo harness reads research_state.md + sidecars before raw source files
  -> memo harness writes final.md and proposals.jsonl
  -> StateOwner post-pass validates proposals
       ACCEPT / REJECT / REVISE
       apply accepted/revised changes to research_state.md + sidecars
       append diff_log.jsonl and accepted_changes.jsonl
  -> integrity check
       pass: status complete
       fail: rollback state from state_snapshot and status records failure
```

The v3 distinction from v1: v1 turns sources into a memo and preserves run artifacts. v3 turns memo-discovered durable insights into a reusable company-level state that future runs read first.

## Cleared v3 boundaries

### 1. CompanyResearchState is an artifact bundle, not a giant JSON document

`CompanyResearchState` means the durable per-company state folder:

```text
vault/companies/<TICKER>/research/state/
  research_state.md
  state_index.json
  evidence_index.json
  open_questions.json
  conflicts.json
  valuation_snapshot.json
  diff_log.jsonl
```

The concept boundary matters more than a class name: these files are the durable company research state. `final.md` is a memo output view, not the state center. Raw source files and the vault document registry remain owned by v1/vault ingestion.

### 2. `research_state.md` is canonical and markdown-first

`research_state.md` is the analyst-facing durable artifact. It should be readable in Obsidian and useful to both humans and LLMs.

Required rules:

- H2 sections (`##`) are the top-level state sections.
- The section set is open: standard sections are encouraged, company-specific sections are allowed.
- Each section carries a stable in-band HTML anchor immediately after the H2:

```markdown
## Guyana growth engine
<!-- key: guyana_growth_engine -->

<prose with [S1] citations>
```

- The anchor key is canonical identity. The H2 text can be renamed by a human; the section can move; the anchor still identifies it.
- `state_index.json` byte ranges are a cache derived from anchors, not ground truth.
- Section bodies are markdown prose, bullets, and tables. Do not force every analyst claim into a Pydantic object.

Recommended-but-not-required initial sections:

```markdown
# <TICKER> Research State

## Current research summary
<!-- key: current_research_summary -->

## Source coverage / evidence map
<!-- key: source_coverage -->

## Durable facts
<!-- key: durable_facts -->

## Key guidance / metrics
<!-- key: key_guidance_metrics -->

## Risks / bear cases
<!-- key: risks_bear_cases -->

## Thesis (candidate)
<!-- key: thesis_candidate -->

## Catalysts
<!-- key: catalysts -->

## Open questions
<!-- key: open_questions -->
<!-- derived: rendered from open_questions.json -->

## Conflicts / uncertainty
<!-- key: conflicts_uncertainty -->
<!-- derived: rendered from conflicts.json -->

## Valuation snapshot pointer
<!-- key: valuation_snapshot_pointer -->
```

Company-specific sections such as `guyana_growth_engine`, `permian_inventory`, `downstream_margin_cycle`, or `lng_chemicals_optionality` are expected, not exceptions.

### 3. Structured sidecars are for invariants only

Markdown is canonical for analyst prose, but several objects must remain structured because they need IDs, status, equality semantics, or reproducible lifecycle behavior.

| File | Purpose |
|---|---|
| `state_index.json` | Section metadata: key, heading, byte range cache, freshness, cite keys, warnings, provenance. |
| `evidence_index.json` | Append-only mapping from stable cite keys (`S1`, `S2`) to `EvidenceRef`. |
| `open_questions.json` | Durable question backlog with lifecycle states. |
| `conflicts.json` | Durable conflict ledger with evidence on both sides and resolution state. |
| `valuation_snapshot.json` | Time-sensitive valuation data with `as_of`, source keys, assumptions, and fields. |
| `diff_log.jsonl` | Append-only audit of accepted/rejected/revised proposals. |
| `memos/<run_id>/proposals.jsonl` | Every proposed update emitted by the memo LLM for this run. |
| `memos/<run_id>/accepted_changes.jsonl` | Subset StateOwner applied or revised/applied. |

Rule of thumb: if it has an ID/status/lifecycle, it is JSON. If it is analyst prose, it is markdown. If it is provenance for prose, it lives in `state_index.json` and `evidence_index.json`.

### 4. Cite keys are stable and append-only

State markdown cites evidence using `[S1]`, `[S2]`, etc. These keys resolve through `evidence_index.json`.

Rules:

- Once `S2` points to an evidence record, it does not get reassigned.
- New evidence gets a new higher key.
- Per-run `state_snapshot/evidence_index.json` freezes the mapping a memo saw.
- Archived `final.md` and archived `research_state.md` snapshots must remain auditable.
- If the canonical evidence index is ever compacted manually, archived snapshots remain the fallback truth.

### 5. StateDelta is a proposal envelope, not full-state JSON

Ted challenged whether `StateDelta` and `CompanyResearchState` need complete JSON control. Final decision:

- Do **not** use full rigid state JSON as the canonical state.
- Do **not** try to avoid structure entirely.
- Use a small structured proposal envelope with markdown bodies.

Minimum proposal types:

```text
propose_section_update(
  section_key: str,
  action: "replace" | "append" | "create" | "remove",
  body_markdown: str,
  evidence_keys: list[str],
  rationale: str,
)

propose_question(
  question_id: str | null,
  text: str,
  priority: "low" | "normal" | "high",
  related_section_key: str | null,
)

propose_close_question(
  question_id: str,
  evidence_keys: list[str],
  proposed_answer: str,
  rationale: str,
)

propose_conflict(
  summary: str,
  left_evidence_key: str,
  right_evidence_key: str,
  severity: "low" | "medium" | "high",
)

propose_valuation_snapshot(
  as_of: str,
  fields: dict[str, number | string | null],
  source_keys: list[str],
  assumptions_markdown: str | null,
)

propose_no_op(rationale: str)
```

Physical v3.0 representation: JSON Lines in `memos/<run_id>/proposals.jsonl`, one record per proposal. The content field for section updates is markdown.

Why this boundary: StateOwner must know which section to change, which action to take, which evidence backs the change, and why. A pure prose handoff is too ambiguous to validate. The JSON envelope is the minimum reliable surface; the analyst content remains markdown.

### 6. v3.0 uses D1 post-memo proposals, not in-flight tool calls

Desired long-term shape:

```text
memo LLM -> propose_state_update(...) -> StateOwner returns ACCEPTED/REJECTED/REVISED -> memo LLM continues
```

But current harness does not expose a structured function-call channel with typed round trips. It supports generic write/edit/patch/status style operations. Therefore v3.0 uses D1:

```text
memo LLM writes final.md + proposals.jsonl
StateOwner runs after memo completion
StateOwner applies accepted/revised proposals
next run sees updated state
```

This preserves the same trust boundary and can ship against existing harness behavior. True in-flight D2 tool calls are a v3.2 target after harness work.

### 7. StateOwner is the durable write boundary

Memo output does not directly mutate durable state files. StateOwner owns durable writes.

v3.0 StateOwner is a deterministic Python post-pass:

```text
load state_snapshot/
load proposals.jsonl
for each proposal:
  validate schema
  validate evidence keys
  apply rule-based policy
  ACCEPT / REJECT / REVISE
  apply accepted/revised change to state/ files
  append diff_log.jsonl
write accepted_changes.jsonl
run integrity check
rollback on failure
```

A later LLM-mediated StateOwner can be explored after the deterministic path proves useful. Do not start with an LLM StateOwner: it adds cost and makes v3.0 harder to test.

### 8. Initial StateOwner policy

Minimum policy rules:

- `evidence_keys` must resolve in `evidence_index.json` for evidence-bearing proposals.
- `section_key` must match a slug pattern such as `[a-z][a-z0-9_]*`.
- Direct edits to derived sections (`open_questions`, `conflicts`) are rejected; use dedicated proposal types.
- `body_markdown` for section updates must not include a fresh H2 heading; StateOwner owns section boundaries.
- Quantitative-looking sentences without same-sentence `[S]` citation are revised or warning-marked, not silently accepted.
- Contradictions against higher-grade evidence should become `propose_conflict`, not overwrite.
- `remove` on a section with current-run citations is rejected unless an explicit safe path is added later.
- Soft cap 20 proposals/run with warning; hard cap 50 proposals/run.
- New section creation is allowed, but near-duplicate section keys should be revised toward existing sections.

### 9. Open questions are structured but rendered into markdown

`open_questions.json` owns question lifecycle. `research_state.md` may include a derived Open Questions section for readability.

Question lifecycle vocabulary for v3.0:

```text
open -> proposed_close -> answered
open -> rejected / stale / deferred
```

v3.0 should be conservative:

- A memo can propose a new question.
- A memo can propose closing a question with cited answer.
- StateOwner accepts only if citations resolve and answer is specific.
- Derived markdown is regenerated from JSON; manual edits to the derived Open Questions section are overwritten.

### 10. Conflicts are structured but rendered into markdown

`conflicts.json` owns conflict lifecycle. The markdown section is derived/readable.

A conflict needs:

- stable conflict ID;
- summary;
- left/right evidence keys;
- severity;
- status (`open`, `resolved`, `stale`, etc.);
- optional resolution rationale.

StateOwner should reject silent overwrite of contradictory evidence unless a conflict proposal is present.

### 11. Valuation uses sibling `valuation_snapshot.json`

Valuation is separate from `research_state.md` because it is long, high-churn, and point-in-time.

`valuation_snapshot.json` should include:

- `as_of` date/time;
- source keys;
- market price / market cap / enterprise value where available;
- multiples and yield fields where available;
- DCF/SOTP/peer-comparison assumptions where available;
- assumptions as markdown or structured fields;
- warnings about stale inputs or weak source grade.

The markdown state should point to the latest valuation snapshot rather than inline all valuation details.

### 12. Per-run snapshots are required

Every memo run using v3 state must freeze the state it saw:

```text
vault/companies/<TICKER>/research/memos/<run_id>/state_snapshot/
  research_state.md
  state_index.json
  evidence_index.json
  open_questions.json
  conflicts.json
  valuation_snapshot.json
```

This snapshot is required for:

- reproducibility;
- audit of `[Sn]` cite keys at memo time;
- rollback if StateOwner integrity check fails;
- comparing what the memo saw with what StateOwner changed.

### 13. Post-run integrity check is a hard gate

Before marking the v3 run complete:

1. Every `[Sn]` in `research_state.md` resolves in `evidence_index.json`.
2. Every section in `state_index.json` corresponds to an anchored H2 in `research_state.md`.
3. Every anchored H2 has a `state_index.json` record.
4. Derived Open Questions / Conflicts sections match rendered sidecars.
5. `accepted_changes.jsonl` is a subset of `proposals.jsonl` by proposal ID.
6. Quantitative-looking prose without citation is warning-marked or rejected/revised according to policy.
7. No state write occurs without a diff log entry.

Failure behavior: rollback state files from `state_snapshot/`, mark run status with failure, and keep proposals for debugging.

### 14. Concurrency policy

v3.0 should use a per-ticker file lock around StateOwner apply. Two concurrent XOM runs must not interleave writes.

Acceptable v3.0 behavior:

- both runs may generate proposals from their own snapshots;
- StateOwner apply serializes;
- second apply should either revalidate against current state or fail safely with status warning if assumptions changed.

Do not attempt complex merge resolution in v3.0.

## Intended module boundaries

Suggested package layout:

```text
src/research_platform/state/
  __init__.py
  contracts.py   # Propose* schemas, StateIndex, EvidenceIndex, OpenQuestion, Conflict, ValuationSnapshot
  storage.py     # load/save state folder, snapshots, anchors, atomic writes, file lock
  apply.py       # StateOwner: apply proposals, mutate markdown/sidecars, append diff_log
  policy.py      # rule-based ACCEPT / REJECT / REVISE logic
  render.py      # render derived Open Questions and Conflicts sections
  lint.py        # cite resolution, number-without-cite heuristic, integrity check
```

Likely later modules:

```text
src/research_platform/state/synthesizer.py    # v3.1 bootstrap from empty state, if needed
src/research_platform/state/memo_bridge.py    # v3.2 true tool-loop bridge, if harness supports it
```

Keep `src/research_platform/memo_run.py` as the cross-layer orchestrator home. Add v3 state calls there or behind a new helper, without breaking the v1 path.

## Implementation order

### v3.0 — deterministic markdown-first state foundation

No real LLM calls in tests.

1. Add `src/research_platform/state/contracts.py`.
2. Add state folder storage and snapshot helpers.
3. Add markdown section parser/anchor maintenance.
4. Add `state_index.json` and `evidence_index.json` models.
5. Add proposal schemas and JSONL read/write.
6. Add rule-based StateOwner policy.
7. Add apply logic for section create/append/replace/remove.
8. Add open questions/conflicts/valuation sidecar support.
9. Add diff log and accepted changes output.
10. Add integrity check and rollback from snapshot.
11. Add memo prompt fragment requiring `proposals.jsonl` output.
12. Wire v3 state post-pass into memo run behind a safe flag/config.

### v3.1 — bootstrap / synthesizer refinement

Only after v3.0 is stable:

- decide whether empty-state bootstrapping needs a synthesizer pre-pass;
- optionally create a skeleton `research_state.md` before memo run;
- improve section recommendations from the Librarian section list;
- refine lint heuristics from actual XOM output.

### v3.2 — true in-flight StateOwner tool loop

Requires harness capability work:

```text
memo LLM -> structured propose_state_update call -> StateOwner result -> memo LLM continues
```

Only pursue after D1 proves useful and the harness can support typed tool round trips.

### v3.3 — proof demo

Two-run XOM demo:

1. Run 1 from empty or near-empty state.
2. Verify state folder populated with cited sections and sidecars.
3. Run 2 with one new/changed source.
4. Verify `proposals.jsonl`, `accepted_changes.jsonl`, and `diff_log.jsonl` show non-trivial accepted change.
5. Verify open question carryover and any proposed close behavior.
6. Verify valuation snapshot remains separate.
7. Verify post-run integrity passes both runs.

## Test checklist

### Contracts / schema

- Proposal schemas accept valid examples and reject missing required fields.
- `state_index.json` round-trips through Pydantic.
- `evidence_index.json` enforces cite-key uniqueness.
- `open_questions.json`, `conflicts.json`, and `valuation_snapshot.json` round-trip.
- Proposal IDs are stable enough to compare proposals vs accepted changes.

### Markdown storage

- Anchored H2 sections are parsed correctly.
- `byte_range` cache updates after section body grows/shrinks.
- Heading rename with same anchor preserves identity.
- Section reorder with same anchor preserves identity.
- Missing anchor logs warning and avoids unsafe mutation.

### StateOwner policy

- Unresolved cite key -> REJECT.
- Direct edit to derived section -> REJECT.
- Quantitative-looking sentence without same-sentence cite -> REVISE/warning.
- Prose-only churn with same evidence and weak rationale -> REJECT.
- Near-duplicate section key -> REVISE toward existing key.
- Proposal cap warning/hard cap behavior works.

### Apply / audit

- Valid `propose_section_update` updates markdown and state index.
- Accepted change appends `diff_log.jsonl` and `accepted_changes.jsonl`.
- Rejected proposal appears in proposal audit but does not mutate state.
- Open question proposal updates JSON sidecar and rendered section.
- Conflict proposal updates JSON sidecar and rendered section.
- Valuation proposal writes `valuation_snapshot.json`, not main markdown body.

### Integrity / reproducibility

- Snapshot equals run-start state.
- Every `[Sn]` resolves.
- Every indexed section has an anchor.
- Derived sections match rendered sidecars.
- Failed integrity check rolls back from snapshot.
- Two mocked runs produce deterministic accepted changes.

### Integration

- v1 memo path remains available.
- v3 memo context files put `research_state.md` and sidecars before raw sources.
- Memo run writes `proposals.jsonl` in the expected run folder.
- StateOwner post-pass runs after memo completion and updates status.

## Explicit non-goals for v3.0

- No new SQLite tables.
- No full rigid `research_state.json` canonical state.
- No LLM-mediated StateOwner yet.
- No in-flight structured tool calls yet.
- No complex concurrent merge resolution.
- No perfect quantitative-claim parser; use conservative heuristics and warnings.
- No final analyst-owned investment recommendation commitment. `thesis_candidate` remains candidate until Ted/human process promotes it outside v3.0.

## Remaining uncertainties to watch during implementation

These should not block v3.0 design, but should be revisited after the first mocked/live outputs:

1. **One-phase vs two-phase D1.** v3.0 starts one-phase: memo writes final.md and proposals.jsonl in one pass. If proposals quality is poor, A/B a two-phase prompt.
2. **Hand-edit reconciliation.** Anchors should survive normal edits. If humans delete anchors often, v3.1 needs explicit reconciliation UX.
3. **Concurrent runs.** File lock is enough for v3.0. If multiple same-ticker runs become common, revisit merge semantics.
4. **Lint quality.** Number-without-cite detection should start conservative; avoid blocking on noisy regex.

## Decision gate

Implementation may begin after this checklist and the two Claude planning docs are committed.

First implementation target: **v3.0 deterministic foundation** in `src/research_platform/state/`, with mocked/no-LLM tests only.
