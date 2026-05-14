# V3 Markdown-First State + StateOwner Tool-Loop — Addendum (Claude)

Author: Claude Code (planning daemon)
Date: 2026-05-14
Companion to: `docs/research_platform/module_plans/V3_COMPANY_RESEARCH_STATE_BOUNDARY_CLAUDE.md`
Status: Planning only — no implementation.

This addendum responds to Ted's pushback on the JSON-first design in the
companion document. The objection in his words:

> A rigid full JSON schema on `CompanyResearchState` constrains the LLM and
> will miss company-specific sections. The canonical analyst artifact should
> be markdown; JSON sidecars only for invariants. The memo LLM should call a
> `propose_state_update` tool, and a separate **StateOwner** loop should
> accept/reject/revise. StateDelta is the hardest part — maybe we can avoid
> JSON entirely; no clean answer yet.

The companion doc was good on **boundaries** (state vs memo, durable vs
run-scoped, citations required) and on **storage discipline** (artifact-first,
no new SQLite). It was wrong on **shape**: it modeled the canonical state as a
fixed 7-section JSON object, which is precisely the constraint Ted flagged.

This addendum keeps the boundary work, replaces the shape, and proposes the
StateOwner loop.

---

## 1. Markdown-first canonical — is this the right pivot?

**Short answer: yes, with one caveat.** "Markdown-first" does not mean
"unstructured." It means the *unit of analyst meaning* (a section of prose
about a company, with cited claims inside) is expressed as markdown rather
than as a list of JSON claim objects. The schema lives at the **section
boundary** and at the **lifecycle-object boundary**, not inside the prose.

### 1.1 Why JSON-first failed the test

The companion doc's `research_state.json` (V3 §4.2) hard-codes the section
keys (`business_overview`, `key_guidance`, `key_metrics`, `risks`,
`thesis_candidate`, `catalysts`, `valuation_snapshot`). Three things break:

1. **Company-specific sections vanish.** "Tanker fleet age curve" for FRO,
   "China gaming approval cadence" for NTES, "SaaS net retention by cohort"
   for a software co — none have a slot. Either the LLM stuffs them into
   `risks` (loses fidelity) or the schema grows per company (loses
   testability).
2. **The Librarian post explicitly uses a 14-section *standard*, not a rigid
   superset.** Sections like "Known Truth" / "Key Guidance" / "Open
   Questions" co-exist with topic-specific sections that exist only for that
   company. We are designing toward that pattern; the JSON schema fought it.
3. **Claim-level JSON granularity is the wrong unit.** An analyst writes "the
   thesis is X because of Y [S2] and Z [S4], with the caveat that [S5]" as a
   paragraph, not as three claim objects. Forcing claim objects either
   (a) generates over-atomized fragments the LLM mid-runs to write, or
   (b) generates one "claim" per paragraph that contains the same prose —
   structure for structure's sake.

### 1.2 What the markdown-first pivot buys

- **Open section set.** New section emerges naturally; the schema does not
  need a migration.
- **Natural authoring unit.** LLM writes a paragraph. Done. No reshaping.
- **Human readability is the default.** `research_state.md` is the artifact
  the analyst opens in Obsidian. JSON-first treated it as a rendered byproduct
  that could drift from `state.json`; markdown-first eliminates the drift
  surface.
- **Cite tags survive.** `[S1]` / `[S3]` in prose are still machine-parsable
  and still resolve through a structured `evidence_index`.

### 1.3 The caveat — what we lose and how to compensate

JSON-first gave us **per-claim provenance**: every quantitative number had
its own `extractor_model`, `extraction_method`, `confidence`, and a hard
schema guarantee that a quantitative LLM claim could not silently be
"accepted." We do not want to lose those.

The compensation is to push provenance down to **two anchors**:

- **Per-section provenance** in `state_index.json` (one record per section:
  who wrote it, in which run, with which model, citing which evidence_ids,
  freshness).
- **Per-cite provenance** via the `evidence_index` — `[S1]` resolves to an
  `EvidenceRef` that already carries `source_grade`, `document_id`, snippet.

Claims do not need their own object layer if every prose claim is bound to
an evidence_id and every section has a header record. The `ClaimRecord` model
in `src/vault/contracts.py:122` stays useful but moves to a **secondary**
role — used only by lint passes that extract quantitative claims from prose
for QC (e.g. "this paragraph has a number but no `[S]` cite"). It is no
longer the canonical write target.

**Verdict: pivot to markdown-first. The pivot is correct in principle; the
risk is implementation laziness collapsing it into "freeform wiki blob." The
mitigations in §2 and §6 are what make it safe.**

---

## 2. What must remain structured (and why)

These objects need **lifecycle states, IDs, equality semantics, or
machine-readable cadence**. Markdown can encode none of those reliably.

| Sidecar | Why structured | Cadence |
|---|---|---|
| `open_questions.json` | Questions have IDs and a state machine (`open` / `proposed_close` / `answered` / `rejected`); carry across runs; must be diffable | Updated each run |
| `conflicts.json` | Same — IDs, severity, two-sided refs, resolution state | Updated each run |
| `valuation_snapshot.json` | Numbers with as-of date; high-frequency churn (daily) that must not pollute prose diffs | Daily-ish |
| `state_index.json` | Maps section keys → file location / last_updated_at / last_run_id / cite count / synthesizer model+version | Updated each run |
| `evidence_index.json` | Maps cite keys (`S1`, `S2`, …) → `EvidenceRef` (document_id, snippet, source_grade) | Updated each run |
| `diff_log.jsonl` | Append-only audit; one record per accepted/rejected proposal | Append each run |
| `proposals/<run_id>.jsonl` | Per-run record of every `propose_*` tool call the memo LLM emitted, accepted or not | Frozen at end of run |

Everything else is markdown:

- `research_state.md` — the canonical analyst artifact, open section set,
  prose with `[Sn]` cite tags, H2 per section.
- Per-section bodies — paragraphs, bullet lists, tables. No fixed schema.

**The rule of thumb:** *if it has an ID and a status, it is JSON. If it is
prose a human reads, it is markdown. If it is provenance for prose, it is
the index pair (state_index + evidence_index).*

This is also why `evidence_index` matters more than the companion doc gave
it credit for. In markdown-first, the entire correctness story rests on
`[Sn]` cites being resolvable. The index is non-optional.

---

## 3. StateDelta — design alternatives

This is the hard part. The companion doc punted: it had the synthesizer write
a "new state, full re-render" and computed deltas mechanically. In
markdown-first that does not work — the synthesizer cannot rewrite the entire
markdown from scratch each run, both because the prose would churn (Risk #2
in companion doc §11) and because it gives the LLM no way to express
"add this paragraph to the China exposure section without touching the rest."

Six candidate designs:

### A. Pure JSON proposal (status quo from companion doc)

Memo LLM emits `proposed_state_delta.json` with claim-level adds/updates/removes.

- Pro: fully testable, schema-validated, every change has a typed envelope.
- Con: exactly the rigidity Ted objected to. Claim objects force re-shaping
  prose into structured fragments. Section set must be fixed.

**Reject for canonical use.** Keep JSON-as-transport (see D), not
JSON-as-content.

### B. Markdown patch with frontmatter

LLM emits one or more markdown chunks like:

```
---
action: replace_section
section_key: china_regulatory_exposure
evidence_keys: [S2, S4]
rationale: "Q1 call mentions new approval pause"
---
## China regulatory exposure

Management acknowledged a pause in new title approvals through Q2 [S2],
which together with the WeChat circular [S4] suggests a 60–90 day hit
to release cadence.
```

- Pro: LLM writes naturally; humans can read the patch directly; section_key
  is structured but the body is free.
- Con: Multi-section patches in one response need a separator/manifest. Diff
  semantics get awkward for partial edits ("add a sentence to paragraph 3
  of risks"). Frontmatter parsing has edge cases (escaped `---` in body).

**Keep as a fallback for human-authored patches; not the primary path.**

### C. Command-block DSL (text-only)

LLM emits commands in a DSL:

```
@update_section china_regulatory_exposure
@cite S2 S4
@body
Management acknowledged ... [S2] ... [S4] ...
@end
```

- Pro: parseable; explicit; reads close to natural authoring.
- Con: yet another DSL to specify, document, and lint. We invent a tool API
  inline in prompts. Failure modes (forgotten `@end`) are real.

**Reject. The provider already gives us a tool-call mechanism — use it.**

### D. Structured proposals file with markdown bodies (RECOMMENDED)

Memo LLM emits proposals as **records in a file**, not as native function
calls. Each record has structured args (schema-validated) and a markdown
body string. This is realized in one of two physical forms depending on
harness capability (see §5.4 for the capability check):

- **D1 (v3.0 primary): post-memo proposals file.** After writing final.md,
  the memo LLM writes `memos/<run_id>/proposals.jsonl` — one JSON record
  per line. StateOwner runs as a post-pass over that file.
- **D2 (v3.1+, if the harness supports interleaved structured tool calls):**
  same record shape, but emitted as tool calls during memo generation, with
  StateOwner applying them in-line and returning the updated section anchor
  to the LLM as the tool result.

The schema is bounded but small (identical for D1 and D2):

```text
propose_section_update(
  section_key: str,              # arbitrary, open set
  action: "replace" | "append" | "create" | "remove",
  body_markdown: str,            # freeform; can be empty for "remove"
  evidence_keys: list[str],      # cite keys this body relies on
  rationale: str,                # 1–2 sentences, why
)

propose_question(
  question_id: str | null,       # null = new; else update existing
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
  as_of: str,                    # ISO date
  fields: dict[str, number|null],# market_cap, ev, pe, ev_ebitda, …
  source_keys: list[str],
)

propose_no_op(rationale: str)    # explicit "no state change this run"
```

- Pro: JSON only at the envelope; content is markdown. Schema is small,
  testable, stable. Open section set is preserved (section_key is a string).
  Audit trail is the proposals file itself. D1 works against the existing
  harness without new tool plumbing — the LLM already writes files via
  `write`/`edit`/`patch` (see `src/harness/presets.py:8`). D2 is a natural
  upgrade path.
- Con: D1 means accepted proposals do not feed back to the memo LLM during
  generation — the memo cannot cite "the section I just updated this run"
  without a small two-phase loop (memo → propose → optional memo revision).
  D2 needs harness work to add structured tool calls. Schema maintenance is
  still real but small.

**Recommended as the v3 primary mechanism, in form D1 for v3.0.** This is
the smallest JSON surface that gives us provenance while letting the LLM
author markdown freely, and it ships against the harness as it stands.

### E. Conversational two-LLM protocol

Memo LLM writes the memo and *describes in prose* what state should change.
A second LLM (StateOwner agent) reads that narration plus the prior state
and emits the actual writes.

- Pro: maximum flexibility; matches how a junior analyst hands work to a
  senior. No DSL.
- Con: two LLM calls per run minimum (more if revisions). Cost. Hard to
  pin "what was proposed" without log gymnastics. The memo LLM's
  descriptions are easy to underspecify ("update the risks").

**Defer. Reasonable v3.2+ refinement once the tool-call path is stable.**

### F. Append-only event log

LLM emits a stream of events: `CITE`, `CLAIM`, `QUESTION`, `CONFLICT`.
State is rebuilt by replaying events.

- Pro: append-only is auditable by construction; fits the diff_log model.
- Con: requires a replay engine that is correct (event sourcing
  bugs are awful); does not solve the "where in the markdown does this
  claim live" problem; pushes complexity onto consumers.

**Reject for v3. Possible future.**

### Decision matrix

| Design | Open section set | LLM-native | Schema cost | Audit trail | v3 recommendation |
|---|---|---|---|---|---|
| A. Pure JSON | No | Bad | Medium | Excellent | Reject |
| B. Markdown patch + frontmatter | Yes | Good | Low | Adequate | Fallback / human edits |
| C. Command DSL | Yes | Good | Medium | Adequate | Reject |
| **D1. Proposals file (post-memo)** | **Yes** | **Good** | **Low** | **Excellent** | **PRIMARY v3.0** |
| D2. Tool-call + markdown body (in-flight) | Yes | Excellent | Low | Excellent | Defer to v3.2 (needs harness) |
| E. Two-LLM conversational | Yes | Excellent | None | Weak | Defer to v3.2 |
| F. Event log | Yes | Mixed | High | Excellent | Reject for v3 |

**Can we avoid JSON entirely?** Practically, no — Ted's question deserves a
direct answer. *Some* structured envelope is needed for: (1) which section
this changes, (2) which evidence_ids back it, (3) what kind of action
(replace/append/create/remove). Without those, the StateOwner has no way to
locate the change or check that cited evidence exists. The smallest
envelope is the tool-call schema in (D), which is ~50 lines of Pydantic
total. That is the floor.

---

## 4. Recommended minimal v3.0 design

### 4.1 On-disk layout

```text
vault/companies/<TICKER>/research/
  state/
    research_state.md            # CANONICAL — markdown, open section set
    state_index.json             # section_key → metadata (last_updated, run_id, model, cite_count, freshness)
    evidence_index.json          # cite_key (S1, S2, …) → EvidenceRef
    open_questions.json          # durable questions with lifecycle
    conflicts.json               # durable conflict ledger
    valuation_snapshot.json      # point-in-time numbers
    diff_log.jsonl               # append-only audit
  context_packages/<run_id>.json # unchanged (v1)
  memos/<run_id>/
    final.md, manifest.json, status.json
    source_index.md, question_list.md             # unchanged (v1)
    state_snapshot/                               # frozen copy of state/ at run start
      research_state.md, state_index.json, evidence_index.json,
      open_questions.json, conflicts.json, valuation_snapshot.json
    proposals.jsonl                               # every propose_* tool call, accepted or not
    accepted_changes.jsonl                        # subset that StateOwner applied
```

Key change from companion doc §4.1: **`research_state.md` is canonical, not
rendered**. `state_index.json` is the structured *index over* the markdown,
not a parallel representation of it.

### 4.2 `research_state.md` shape

```markdown
# <TICKER> Research State

<!-- as-of run: memo-xom-2026-05-14-1407 -->
<!-- Headings and the key anchors below them are managed by StateOwner. -->
<!-- Body prose is markdown; cite tags [Sn] resolve via evidence_index.json. -->

## Business overview
<!-- key: business_overview -->
<prose with [Sn] cite tags>

## Key guidance
<!-- key: key_guidance -->
<prose>

## Known truth
<!-- key: known_truth -->
<prose>

## Risks
<!-- key: risks -->
<prose>

## Thesis (candidate)
<!-- key: thesis_candidate -->
<prose, flagged "candidate; not analyst-confirmed">

## Catalysts
<!-- key: catalysts -->
<prose>

## China regulatory exposure
<!-- key: china_regulatory_exposure -->
<prose>

## Open questions
<!-- key: open_questions -->
<!-- derived: rendered from open_questions.json. Body edits will be overwritten. -->
- [ ] q-2026-05-001 — <text> — last seen run memo-xom-…
- [ ] q-2026-04-018 — <text> — proposed_close in this run, evidence [S5]
```

Rules:

- Sections are H2 (`##`). No deeper nesting at the section boundary.
- Each section is identified by a stable `section_key`, carried **in-band**
  as an HTML comment anchor `<!-- key: <section_key> -->` immediately after
  the H2. StateOwner locates sections by anchor first, by `byte_range`
  second, by H2 title last (best-effort fallback for hand-edits). The
  anchor is the canonical identity; H2 text is human-facing and may be
  reworded.
- Cite tags `[Sn]` resolve through `evidence_index.json`. `Sn` is **stable
  per evidence_id for the life of the state** (append-only); a given S2
  always points to the same `evidence_id`. New evidence gets a new Sn.
  Frozen `memos/<run_id>/state_snapshot/evidence_index.json` keeps a copy
  of the mapping the memo saw, so archived `final.md` files remain
  resolvable even if the canonical index is later compacted.
- "Open questions" and "Conflicts" sections are *rendered* from the JSON
  sidecars at write time, not authored as freeform prose. They carry a
  `<!-- derived: ... -->` comment marking them as overwriteable. Policy
  rejects any `propose_section_update` whose `section_key` is in the
  derived set (`open_questions`, `conflicts`).

### 4.3 `state_index.json` shape

```text
{
  "schema_version": 1,
  "ticker": "XOM",
  "last_run_id": "memo-xom-…",
  "last_updated_at": "…",
  "sections": [
    {
      "section_key": "business_overview",
      "heading": "Business overview",
      "byte_range": [124, 1893],          // where in research_state.md
      "last_updated_at": "…",
      "last_updated_run_id": "…",
      "synthesizer_model": "claude-opus-4-7",
      "synthesizer_version": "v3.0",
      "cite_keys": ["S1", "S3", "S7"],
      "freshness": {
        "newest_evidence_at": "2026-04-12",
        "policy_max_age_days": 365,
        "is_stale": false
      },
      "warnings": []                       // e.g. "single B-grade source"
    },
    …
  ]
}
```

- The in-band anchor `<!-- key: <section_key> -->` (see §4.2) is the
  canonical identity. `byte_range` is a cache, recomputed by re-locating
  the anchor at each write — so it survives hand-edits, section reorder,
  and length changes in adjacent sections.
- `section_key` is stable across heading renames; `heading` is the display
  text and may drift. This is the equivalent of Obsidian's block IDs
  without the ergonomic mess, and it survives Obsidian's preview rendering
  (HTML comments are preserved).

### 4.4 `evidence_index.json` shape

```text
{
  "schema_version": 1,
  "ticker": "XOM",
  "entries": {
    "S1": {<EvidenceRef>},
    "S2": {<EvidenceRef>},
    …
  }
}
```

The `EvidenceRef` model already exists at `src/vault/contracts.py:71`. Reuse
it. Cite keys (`Sn`) are **append-only and permanent** within a state file:
once `S2 → evidence_xyz` is allocated, S2 forever points to that
evidence_id. New evidence gets a fresh higher-numbered key. This makes
archived `final.md` files (which embed `[Sn]` in prose) durable even after
many runs. The frozen per-run `memos/<run_id>/state_snapshot/evidence_index.json`
copy is the second line of defense if the canonical index is ever
re-keyed (e.g. after a manual compaction).

### 4.5 Module boundaries (planning only)

If/when this gets implemented, suggested package layout:

```text
src/research_platform/state/
  __init__.py
  contracts.py            # tool-call schemas (Propose*), StateIndex, EvidenceIndex
  storage.py              # load/save research_state.md + sidecars; byte_range maintenance
  apply.py                # StateOwner: take a Proposal, mutate files, append diff_log
  policy.py               # rule-based accept/reject logic for v3.0
  render.py               # render Open Questions + Conflicts sections from JSON sidecars
  lint.py                 # post-write checks: cite resolution, evidence presence, etc.
  synthesizer.py          # (v3.1) bootstrapping synthesizer — creates state when none exists
  memo_run.py             # (v3.2) wires the tool-loop into the memo harness
```

This mirrors the companion doc's §10 staging, with `synthesize_state.json`
replaced by `apply.py` (StateOwner) and `render.py` reduced to two
derived-section renderers instead of a full state renderer.

### 4.6 Storage discipline carried over

v3 stays artifact-first. **No new SQLite tables.** Reuse `documents` /
`companies` via the existing vault registry (`src/vault/contracts.py`,
`src/vault/store.py`). The state-layer files in §4.1 are the only new
on-disk artifacts. This matches companion doc §6 Q11 — confirm
unchanged under markdown-first.

### 4.7 What v3.0 deliberately defers

- LLM-mediated StateOwner (option E in §3): v3.0 uses rule-based `policy.py`.
- Synthesizer for *creating* state from scratch on first run: v3.1. v3.0
  accepts an empty state and lets the tool-loop populate it incrementally
  through the memo run. (This is genuinely the smallest version that
  proves the idea; see §7 Q1.)
- Cross-section structural edits ("split risks into two sections"): not in
  v3.0. The only actions are replace / append / create / remove a whole
  section.

---

## 5. Memo LLM → StateOwner proposals pipeline

(Called the "tool-loop" in earlier framing — that name is accurate only for
the v3.2 D2 variant where the LLM emits in-flight tool calls. v3.0 D1 is a
one-way pipeline: memo writes `proposals.jsonl`, StateOwner consumes it.)

### 5.1 Two distinct roles

- **Memo LLM** = the existing harness model. Reads context. Writes the memo
  prose. Calls `propose_*` tools when it sees a state-relevant insight.
- **StateOwner** = a separate process (rule-based in v3.0; possibly a small
  LLM in v3.2). Receives proposals, validates, decides accept/reject/revise,
  writes state files, appends diff_log. Never writes the memo.

This split mirrors the companion doc's §5 "synthesizer is the only LLM
writer to durable state." StateOwner is the same trust boundary, but pulled
out of the synthesizer and made explicit.

### 5.2 Run lifecycle (D1 — v3.0 primary)

```text
1. Orchestrator snapshots state/ → memos/<run_id>/state_snapshot/
     (this snapshot is the rollback source used in step 5 if the
     integrity check fails)
2. Orchestrator builds MemoContextPackage that points memo LLM at:
     - research_state.md (the analyst view)
     - state_index.json (section keys + freshness)
     - open_questions.json (open items only)
     - evidence_index.json (cite key → evidence resolution)
     - new documents this run (DocumentRefs)
   The prompt instructs the LLM: "When finishing, write
   memos/<run_id>/proposals.jsonl with one record per state change you
   recommend; use the schema below."
3. Memo LLM runs against the existing harness. It writes:
     - publish/final.md (the memo)
     - memos/<run_id>/proposals.jsonl (proposals — see schema in §3 D)
4. After the memo run returns, StateOwner runs as a post-pass:
     a. Loads proposals.jsonl.
     b. For each proposal: policy.py decides ACCEPT | REJECT | REVISE;
        lint.py validates cites, evidence, derived-section rule.
     c. ACCEPTED/REVISED records are applied to state/ files in order.
        REJECTED records are logged with reason but do not mutate state.
     d. diff_log.jsonl gains one entry per ACCEPTED/REVISED change.
        accepted_changes.jsonl is written alongside.
5. Orchestrator runs the post-run integrity check (§6.4). On failure,
   accepted changes are rolled back from state_snapshot/ and status.json
   records the failure.
```

Two design subtleties worth being explicit about:

- **D1 is apply-after-run.** The memo LLM does not see its own accepted
  changes mid-generation; it sees state as it was at run start. This is
  acceptable because the memo and the state updates are conceptually one
  authoring act — the memo declares "here is what I noticed worth recording"
  in the same pass it writes the memo prose. If a future run needs the
  state to reflect *this run's* findings, that is what the next run sees.
  Apply-during-run (D2) is deferred until the harness supports structured
  tool-call round-trips and the in-band anchor design (§4.2) is proven
  out under concurrent edits.
- **Prompt-side discipline.** The proposals.jsonl format must be specified
  explicitly in the memo prompt (or in a referenced schema file the LLM
  reads). The harness writes-files tool path is well-trodden — see
  `src/vault/memo_flow.py:24-30` and the existing memo prompt at
  `src/research_platform/prompts/memo_user.md`. The new prompt fragment is
  small: ~30 lines specifying schema + an example.

### 5.3 StateOwner policy (v3.0, rule-based)

Minimum policy set:

- ACCEPT iff: every `evidence_key` resolves in `evidence_index`; rationale
  is non-empty; body_markdown is non-empty (except for `remove` action);
  section_key matches the slug pattern `[a-z][a-z0-9_]*`; section_key is
  not in the **derived set** (`open_questions`, `conflicts`); body does not
  itself contain a fresh `## ` H2 heading (sections are bounded by
  StateOwner, not by LLM-authored headings inside the body).
- REVISE iff: a quantitative-looking claim in body_markdown lacks a cite tag
  in the same sentence — StateOwner appends a warning to the section's
  `warnings` entry and marks the section as having an unresolved
  quantitative claim in `state_index.json`.
- REJECT iff: evidence_key does not resolve; section_key targets a derived
  section; body contradicts a claim in the same section that has higher
  source_grade *without* declaring a `propose_conflict` first; action is
  `remove` on a section that has any cite from the current run (defensive
  — refuse to silently delete fresh work); soft cap of 20 proposals
  exceeded (warn) or hard cap of 50 exceeded (reject excess).

`open_questions` and `conflicts` mutations go through `propose_question` /
`propose_close_question` / `propose_conflict` only. Their markdown is
rendered, never authored.

This is small enough to fit in one Python file (~150 lines). v3.2 can swap
in an LLM-mediated StateOwner if rule-based feels too brittle.

### 5.4 Harness capability — why D1 (file-based), not D2 (tool-call), in v3.0

A quick scan of the existing harness (`src/harness/runtime.py`,
`src/harness/presets.py:8-54`) shows the agent-facing tool set is generic
shell-style: `delegate`, `agents`, `bash`, `write`, `edit`, `patch`,
`status`. There is no structured function-calling channel where an
Anthropic-style tool call would return a typed result mid-generation. That
means D2 (the LLM emits `propose_*` as in-flight tool calls and StateOwner
responds with ACCEPTED/REJECTED in the tool result) is *not* a v3.0-shippable
design — it requires harness work first.

D1 sidesteps this: the LLM writes a `proposals.jsonl` file via the existing
`write` tool, and StateOwner runs as a post-pass invoked by the orchestrator.
It uses only mechanisms the harness already supports. D2 becomes an
optimization for v3.2+, conditional on adding a structured tool channel.

### 5.5 Why this is better than synthesizer-then-memo

The companion doc's flow was: synthesizer first runs to update state, then
memo runs as a consumer. Two issues:

1. The synthesizer never reads the memo it is preparing for. It updates
   state based on raw evidence and prior state. If the memo prompt or user
   question would direct attention to a specific facet (e.g. "focus on
   pricing power"), the state update misses it.
2. Two LLM passes always, even when no state-relevant insight arose.

The tool-loop inverts this: state updates happen *because* the memo LLM
encountered something worth recording, in the same pass it generates the
memo. One pass, opportunistic updates, no synthesizer-vs-memo drift.

(The synthesizer still exists in v3.1 — for bootstrapping when no state
exists yet. After that, the tool-loop carries the load.)

---

## 6. Failure modes and tests

### 6.1 Failure modes specific to markdown-first

| Failure | Mechanism | Detection | Mitigation |
|---|---|---|---|
| Cite tag drift | `[S2]` in prose does not resolve in `evidence_index.json` | `lint.py` cite-resolution check | Reject any proposal that introduces unresolved cite |
| Section sprawl | LLM creates a new section every run (`china_exposure_v2`, `china_risk`, …) | `state_index.json` slug count growth ratio | Policy: max N new sections per run (suggest 1); REVISE to existing key if levenshtein-close |
| Prose-only churn | LLM rewords a section without new evidence; diff_log fills with noise | Pre-apply check: if `evidence_keys` identical to prior section and rationale doesn't mention new evidence | REJECT prose-only edits unless rationale explicitly justifies |
| Section silently dropped | `remove` action proposed without rationale, or `remove` on a section with current-run cites | Policy guard in §5.3 (explicit REJECT rules) | REJECT |
| Heading rename loses identity | Analyst (or LLM) edits H2 text after a manual reword | The `<!-- key: ... -->` anchor (§4.2) is the canonical identity; H2 text is decorative and may drift freely | Anchor-first lookup; H2-text-match only as last-resort fallback |
| Anchor accidentally deleted | Analyst deletes a section including its anchor while editing | StateOwner fails to locate the section by anchor or by H2-text match | Log unresolved; skip mutations to that section_key for this run; emit a status warning |
| Citation laundering | Cite tag resolves, but `quoted_snippet` is not in the source extracted_text | Substring check (existing concern from companion doc §11.6) | `lint.py` runs on accept |
| Open Questions section out-of-sync with `open_questions.json` | Tool path bug | Post-run integrity check renders Open Questions deterministically from JSON and diffs against markdown | Auto-re-render the section if mismatch found |
| Cross-run collision on same section | Two memo runs against the same ticker write proposals concurrently; both want to mutate `risks` | Per-ticker file lock around StateOwner (§9) | Concurrent runs serialize; second sees first's writes. Within a single run, StateOwner applies proposals sequentially, so intra-run "collision" is structurally impossible under D1 |

### 6.2 Failure modes shared with companion doc

These remain real and need the same mitigations the companion doc named:

- Synthesizer drift across runs (companion §11.2) — pin model+version per
  section; treat prose-only diffs as non-events.
- Memo harness re-fetches and writes uncited prose (companion §11.3) — same
  prompt-level discipline, plus `lint.py` cite check on final.md.
- Question backlog grows forever (companion §11.4) — staleness rule on
  questions ("not surfaced in N runs → demote").
- Schema churn (companion §11.5) — `schema_version` on every JSON sidecar.

### 6.3 Test surface (mocked-LLM)

Schema and round-trip:
- `state_index.json` round-trips through Pydantic without loss.
- `evidence_index.json` cite-key uniqueness invariant.
- Every `Propose*` tool call schema rejects empty `evidence_keys` for any
  `action != "remove"`.

Storage:
- `storage.apply_section_replace` is idempotent given the same input.
- `byte_range` is recomputed correctly after a replace that grows or
  shrinks a section.
- Append to a missing section creates it (with policy gate).

StateOwner policy:
- Unresolved cite_key → REJECT with the offending key in the reason.
- Prose-only edit (same evidence, no new rationale) → REJECT.
- `remove` on section with current-run cites → REJECT.
- Quantitative claim without inline cite → REVISE with warning.
- Section_key levenshtein-close to existing (`china_exposure` vs
  `china_regulatory_exposure`) → REVISE to existing.

Tool-loop:
- Memo LLM mock that emits one valid `propose_section_update` →
  research_state.md updated, diff_log.jsonl has one entry, proposals.jsonl
  has one entry, accepted_changes.jsonl has one entry.
- Memo LLM mock that emits one invalid (REJECT) + one valid → research_state
  changes only for the valid; both appear in proposals.jsonl; diff_log has
  one entry.
- Apply-during-run: second proposal sees the first's effect — mock LLM
  proposes `append` to a section it just created, both succeed.

Render:
- Open Questions section in `research_state.md` is byte-identical to the
  output of `render.render_open_questions(open_questions.json)`. Post-run
  integrity check is wired.

Reproducibility:
- `memos/<run_id>/state_snapshot/research_state.md` equals the
  `state/research_state.md` at run start (byte-identical modulo a
  snapshot header).
- Two runs with the same mock LLM transcript produce identical
  `accepted_changes.jsonl`.

Live (gated, no LLM mocks):
- One XOM run from empty state → research_state.md non-empty with ≥1
  cited section.
- Second XOM run with one new document → at least one accepted proposal,
  diff_log.jsonl gains ≥1 entry.

### 6.4 Post-run integrity check

Before the run is marked complete, run:

1. Every `[Sn]` in `research_state.md` resolves in `evidence_index.json`.
2. Every `section_key` in `state_index.json` corresponds to an H2 in
   `research_state.md`; every H2 has a matching `section_key`.
3. Open Questions and Conflicts sections in `research_state.md` are
   identical to the rendered output from their JSON sidecars.
4. `accepted_changes.jsonl` ⊆ `proposals.jsonl` for this run.
5. No `claim_status: active` quantitative claim has `extraction_method:
   llm` (existing `ClaimRecord` rule, if `lint.py` extracts claims).

Any failure rolls accepted changes back from `state_snapshot/`.

---

## 7. Decision questions for Ted

These are the questions where this addendum either supersedes the companion
doc, or where the companion doc's question still applies and the answer
shifts under markdown-first. Each is binary or small-N.

1. **Confirm the markdown-first pivot.** Replace JSON-first
   `research_state.json` with markdown-first `research_state.md` + the five
   structured sidecars in §2? (Claude recommendation: **yes**, as designed
   in §4.)

2. **StateDelta mechanism.** Adopt option D1 (post-memo `proposals.jsonl`
   file with structured envelopes + markdown bodies) as the v3.0 primary?
   (Claude recommendation: **yes**.) D2 (in-flight tool calls) is deferred
   to v3.2 because the existing harness has no structured tool-call channel
   (see §5.4). If D1 is rejected, fall back is option B (markdown patch
   with frontmatter as files in `proposals/`).

3. **One-phase vs two-phase D1.** v3.0 has the LLM write final.md and
   proposals.jsonl in one prompt pass (Claude recommendation — simpler,
   one LLM call). Two-phase (re-prompt with the draft and ask for
   proposals) is more thorough but doubles cost. Acceptable to start
   one-phase and A/B later?

4. **StateOwner v3.0 implementation.** Rule-based Python policy (Claude
   recommendation for v3.0), or skip to LLM-mediated StateOwner now? Note:
   LLM-mediated adds one LLM call per proposal, which can be 5–20 per run.

5. **Bootstrapping.** If no `research_state.md` exists for a ticker on first
   run, should the memo LLM populate it incrementally via tool-loop (Claude
   recommendation for v3.0; lets us ship without a synthesizer), or should
   a synthesizer pre-pass create skeleton sections (v3.1 deferred work)?

6. **Open section set: any guard at all?** v3.0 §5.3 rejects nothing on
   `section_key` other than slug pattern and levenshtein-close-dup. Should
   there be a *recommended-but-not-required* keyword list (business_overview,
   key_guidance, risks, …) that proposals are nudged toward via REVISE?
   (Claude recommendation: **yes, recommended-not-required**, with the
   Librarian 14-section list as the suggestion source.)

7. **Citation strictness in markdown-first.** Same content as companion §6
   Q7 but with new mechanism: (a) require ≥1 A-grade cite for any
   quantitative number? (b) cap quoted_snippet length at 600 chars in
   `evidence_index`? (c) reject body that has a number without an inline
   `[Sn]` in the same sentence? Claude recommends (a) yes, (b) yes, (c) yes
   with REVISE (not REJECT) so the LLM can fix it.

8. **Tool-loop budget.** Should the memo run cap the number of `propose_*`
   tool calls per run (suggest 20), or rely on the LLM to self-limit?
   Capping is defensive against runaway tool-call loops; not capping
   matches the "let the analyst record what they noticed" intent. Claude
   recommends **soft cap of 20 with warning, hard cap of 50**.

9. **`proposals.jsonl` vs `accepted_changes.jsonl`.** Keep both (Claude
   recommendation; the gap between them is signal — frequently rejected
   proposals tell us the policy needs work), or only the accepted set?

10. **Backward compatibility with v1 `memo_run.py`.** Same as companion §6
    Q12 — keep v1 entrypoint as legacy? Independent of markdown-first
    pivot. Claude recommendation: **yes, keep it**.

11. **Module location.** `src/research_platform/state/` (Claude
    recommendation, mirrors companion §6 Q13) with the §4.5 file layout.

12. **Demo acceptance gate.** Same content as companion §6 Q15. Claude
    recommendation under markdown-first: **two XOM runs back-to-back**,
    where (a) run 1 populates state from empty, (b) run 2 adds one new
    document and the diff_log shows a non-trivial accepted change with
    fresh cite, (c) the post-run integrity check (§6.4) passes both runs.

### Questions from companion doc that the markdown-first pivot makes moot

- Companion §6 Q1 (run-level vs company-level): no longer interesting — the
  markdown file is per-company by construction. **Resolved: company-level.**
- Companion §6 Q2 (auto-persist as candidate vs candidate-only): no longer
  meaningful in the same form — there are no "ClaimRecord" objects to mark
  candidate. The closest analog is StateOwner's ACCEPT vs REVISE. The
  candidate/accepted distinction moves up to `state_index.json` warnings
  (e.g. "single B-grade cite" or "quantitative claim without same-sentence
  cite" → section is marked with a warning, not promoted to "accepted").
- Companion §6 Q3 (rule-based vs explicit-command vs interactive
  accept-merge): becomes "what does StateOwner policy.py do" — Q4 above.
- Companion §6 Q6 (valuation snapshot inside vs sibling): **resolved:
  sibling** (`valuation_snapshot.json` as in §2). Markdown-first amplifies
  the reason — daily-churn numbers should not pollute markdown diffs.
- Companion §6 Q9 (memo input: state.md+state.json, only JSON, only
  markdown): **resolved: markdown + sidecars** — research_state.md is the
  primary read, with state_index/open_questions/conflicts as supplements.
- Companion §6 Q10 (memo harness allowed to write proposed deltas):
  **resolved: yes** — the tool-loop is the design.
- Companion §6 Q14 (synthesizer model): deferred until v3.1 since v3.0 has
  no synthesizer pass.

---

## 8. Concise summary

The companion doc had the right boundaries and the wrong shape. Markdown-first
canonical state + JSON sidecars for lifecycle invariants preserves the open
section set Ted wants, while keeping testability through (a) in-band
`<!-- key: ... -->` anchors that make section identity stable across
hand-edits, (b) an append-only `evidence_index.json` that makes `[Sn]` cites
permanent, (c) a structured `state_index.json` over the markdown for
freshness and provenance, and (d) a small Pydantic schema at the proposal
envelope. StateDelta is not a JSON state document — it is a `proposals.jsonl`
file the memo LLM writes alongside `final.md`, consumed by a rule-based
StateOwner post-pass that accepts/rejects/revises and writes the canonical
markdown. The v3.0 StateOwner is rule-based (~150 lines); LLM-mediated
StateOwner is a v3.2 refinement; in-flight tool-call StateDelta is a v3.2
upgrade conditional on adding a structured tool channel to the harness.
Implementation order: (1) state contracts + storage with anchor-based
section lookup + StateOwner policy, (2) memo prompt fragment +
proposals.jsonl wiring, (3) bootstrapping synthesizer for first-run case.

The only place we cannot avoid JSON is the proposal envelope (~50 lines of
Pydantic) plus the lifecycle sidecars. That is the floor. Everything
analyst-facing stays markdown.

---

## 9. Open questions Claude is still uncertain about

These are not for Ted to rule on yet — they need either more thought or
prototype data. (Items the earlier draft listed here that the §4.2 in-band
anchor design and the §5.2 apply-after-run pivot now resolve have been
removed.)

- **Two-phase vs one-phase D1.** v3.0 D1 has the LLM write final.md and
  proposals.jsonl in one pass. A two-phase variant has the LLM (i) write
  final.md, (ii) re-prompt with the draft attached and ask for
  proposals.jsonl. Two-phase gives the LLM a chance to look back at its
  own memo and surface state-relevant insights it noticed while writing,
  but doubles LLM cost. Worth a small A/B once v3.0 ships before committing.
- **Hand-edit reconciliation policy.** If the analyst opens
  `research_state.md` in Obsidian and reworks the Risks section by hand,
  the next run's StateOwner finds:
  (a) the in-band anchor still present → reuse `section_key`;
  (b) the H2 text reworded → harmless, anchor wins;
  (c) the section moved/reordered → harmless, anchor wins;
  (d) the anchor accidentally deleted → fallback to H2-text-match, then
      report unresolved and skip mutations to that section for this run.
  v3.0 should implement (a)-(c) cleanly and log (d) without crashing. If
  (d) recurs, v3.1 needs explicit reconciliation UX.
- **Concurrent runs against the same ticker.** Two memo runs for XOM
  started at the same time would both snapshot, both write proposals,
  both try to apply. v3.0 should take a per-ticker file lock around
  StateOwner; concurrent runs queue. Worth saying out loud.
- **Lint extraction of quantitative claims from prose.** §6.3 assumes
  `lint.py` can flag "number without inline cite." Naive regex over `\d`
  is noisy (dates, list ordinals, exhibit refs). A modest stop-list + a
  preference for matches near `%`, `$`, or unit words is probably enough
  for v3.0; named explicitly so the lint quality bar is not unbounded.
