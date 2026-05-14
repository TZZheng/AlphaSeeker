# AlphaSeeker Research Platform — Decision Boundary Checklist

Last updated: 2026-05-13 22:29 CDT  
Owner: `codex`  
Purpose: verify that module decision boundaries are explicit before implementation begins.

## Status

**Pre-v0 boundary gate is cleared.** Ted answered the Claude Code grill follow-up in email `0728f07d-d6b2-404e-823e-2cbe90da038c`.

Claude Code boundary review is saved at:

- `docs/research_platform/module_plans/CLAUDE_BOUNDARY_GRILL.md`

Claude's verdict was **conditional yes**: the module seams were clear enough for v0 Pydantic contract skeletons once three escalated decisions were confirmed. Ted has now confirmed them.

## Resolved decisions

| Decision | Resolution |
|---|---|
| Retrieval/input module location | Use `src/retrieval/`. |
| v1 retrieval scope | Include manual files, SEC filings, company profile, financial snapshot, and market snapshot. |
| Source-grade taxonomy | Use simple `A/B/C`. |
| Manual file source grade | Manual files do **not** automatically become A-grade. Use explicit user grade if supplied; deterministic A if provenance proves primary; otherwise default `B` with rationale. |
| LLM-active claim scope | Narrative/news/web qualitative claims may default `active` when evidence-backed. For revenue/EPS/guidance/capex and other quantitative financial metrics, prefer deterministic extraction; LLM extraction is not needed by default and should not silently become active. |
| LLM conflict demotion | If an LLM-active qualitative claim conflicts with deterministic A-grade evidence, demote/flag through lifecycle conflict handling rather than letting it remain unqualified active truth. Recommended implementation: auto-demote to `candidate` and create `ConflictRecord`. |
| LLM source grading | Use `proposed_source_grade` + rationale from LLM; deterministic policy accepts/overrides into final `source_grade`. Ted agreed with suggestion 2. |
| Question auto-close | Use `proposed_close` for v0/v1, not full auto-close. Ted agreed with recommendation. |
| Conflict severity | Use deterministic conflict severity rules for v1. Ted agreed with recommendation. |
| Memo blocking behavior | Do not block by default. Human is final decision maker; memo may ship with clear flaw/caveat/warning. Strict blocking can be a later workflow mode. |
| Citation direction | Final target should support Obsidian markdown-style links. v0/v1 contracts should store citation components, not only rendered strings. |
| Vault path convention | Use relative paths. Store `vault_relative_path` / Obsidian-compatible relative components rather than absolute paths in user-facing citation contracts. |
| Human review interface | v1 can use Markdown files (`question_list.md`, `review_queue.md`, `conflict_list.md`); frontend later. |
| Product artifact retention | Keep `final.md + manifest + status`; full scratch only for debug/failure. |
| Manifest concept | Accepted. Manifest is a small reproducibility receipt: run id, ticker, prompt/source refs/checksums/grades, final output path, status. |
| First demo target after v0 | XOM vertical slice using manual files + SEC/profile/financial/market snapshots. |

## Clarified concepts

### 1. Question auto-close

**Plain product meaning:** The lifecycle layer tracks open questions like:

- “What is FY2025 capex guidance?”
- “Did management address the Permian divestiture?”
- “What changed since last quarter?”

When a new source arrives that seems to answer a question, the system needs a policy for changing the question status.

Options:

1. **No auto-close:** question stays `open` until a human marks it answered.
2. **Propose close:** system attaches a candidate answer + evidence and sets `status="proposed_close"`; human can accept/reject later.
3. **Auto-close:** system marks it `answered` by itself based on deterministic/LLM rules.

**Chosen v0/v1 default:** `proposed_close`, not full auto-close. This preserves human as final arbiter, avoids false closure, and only requires `QuestionRecord.status` to include `open | proposed_close | answered | rejected` plus a `proposed_answer_ref` field.

### 2. Conflict severity

**Plain product meaning:** If two sources disagree, the lifecycle layer records a `ConflictRecord`. Severity is the priority label that decides review queue order and memo caveat strength.

Examples:

- 10-K says revenue = X; vendor snapshot says revenue = Y.
- Company press release says HQ changed; old profile still says old HQ.
- Two news sources disagree on whether a deal closed.

Chosen deterministic v1 table:

| Conflict pair | Severity | Memo behavior |
|---|---|---|
| A-grade vs A-grade, same current field | `high` | Clear caveat in memo; top review queue. |
| A-grade vs B-grade, same field | `medium` | Caveat if material; otherwise note. |
| B-grade vs B-grade | `low` | Note only. |
| Current source vs stale source | `low` | Prefer newer; log diff. |
| Material-metric mismatch on current A-grade source | `blocker_review` | Very visible warning/review item; still non-blocking by Ted's rule unless strict mode later. |

## Guardrails for active LLM qualitative claims

For every persisted LLM-extracted claim:

```text
claim_status = "active" | "candidate" | "rejected" | ...
extraction_method = "llm"
extractor_model
extractor_version / prompt version
extraction_run_id
evidence_refs[] with exact quote/snippet or span
source_grade and grade rationale
extracted_at
confidence (optional, not trusted as sole gate)
field_kind = "narrative" | "quantitative" | ...
```

Write-time rule: no evidence reference means no persisted claim.

Important separation: an LLM-active claim from a C-grade source remains a C-grade claim. `active` means “accepted into the working research state,” not “source is high quality.”

For quantitative fields such as revenue/EPS/guidance/capex, v1 should primarily use deterministic extraction from A-grade filings or structured profile/financial/market sources. If an LLM is ever used on quantitative text, persist it as `candidate` unless explicitly promoted by deterministic A-grade confirmation or human review.

## Obsidian citation contract requirements

To support future Obsidian markdown-style citation links without re-parsing memo prose, v0/v1 contracts should store components:

### `DocumentRef`

```text
document_id / source_id
vault_relative_path          # e.g. companies/XOM/sources/sec_filings/10-K-2024.md
display_title                # e.g. XOM 10-K FY2024
source_grade
source_date
checksum
anchor_strategy = heading | block_id | none
```

### `EvidenceRef`

```text
doc_ref
heading_path[]               # e.g. ["Item 1A. Risk Factors", "Commodity Prices"]
quoted_snippet
start_offset / end_offset
```

### `Citation`

```text
citation_key
doc_ref
vault_relative_path
display_title
heading_path
snippet
source_grade
```

Rendering to actual Obsidian syntax can be later, e.g.:

```text
[[companies/XOM/sources/sec_filings/10-K-2024#item-1a-risk-factors|XOM 10-K — Risk Factors]]
```

## Engineering defaults that can be decided now

- v0 uses Pydantic v2-style contracts, consistent with current harness types.
- Package starts at `src/retrieval/` for retrieval contracts/types.
- Manual unverified files default `B` plus `source_grade_rationale`, not a new grade.
- `question_status`: `open | proposed_close | answered | rejected`.
- `conflict_severity`: `low | medium | high | blocker_review`.
- Markdown review files: `question_list.md`, `review_queue.md`, `conflict_list.md`.
- Cleanup modes: `debug_keep_all`, `product_final_only`, `archive_on_failure`.
- v0 stores citation components; final Obsidian rendering can wait.
- XOM remains v0→v1 demo target.

## Minimal v0 contract implications

Contract skeletons should include or leave room for:

1. `ResearchTask.vault_root` or equivalent vault path context.
2. `SourceRecord.source_grade`, `source_grade_rationale`, `proposed_source_grade`, `proposed_source_grade_rationale`.
3. `DocumentRef.vault_relative_path`, `display_title`, `checksum`, `source_grade`.
4. `EvidenceRef.heading_path`, `quoted_snippet`, offsets.
5. `ClaimRecord.claim_status`, `extraction_method`, `extractor_model`, `extractor_version`, `evidence_refs`, `field_kind`.
6. `QuestionRecord.question_status`, `proposed_answer_ref`.
7. `ConflictRecord.severity`, `severity_rule_id`.
8. `MemoContextPackage.citations`, `caveats`, `blockers`.
9. `RetrievalRequest.allow_llm_discovery`, `allow_llm_grading`.

## Stage gate

### Pre-v0 gate

Cleared as of 2026-05-13T22:29 CDT.

### v0 gate

v0 may implement contracts/skeletons only. v0 should not implement full feature behavior beyond field names/status enums needed to express the decisions above.

### v1 gate

Before v1 feature implementation, re-check:

- source-grade behavior in real retrieval outputs;
- LLM-active/candidate record behavior in real web/news extraction;
- context-package blocker/caveat semantics;
- cleanup manifest contents;
- Obsidian path/export convention.
