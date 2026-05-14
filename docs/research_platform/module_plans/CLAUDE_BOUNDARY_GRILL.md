# AlphaSeeker Research Platform — Boundary Grill (Claude review)

Reviewer: `claude-opus-4-7`
Date: 2026-05-13
Inputs reviewed: `CONSOLIDATED_PLAN.md`, `DECISION_BOUNDARY_CHECKLIST.md`, Ted's 2026-05-13 verbal decisions.

## 1. Executive summary

**Are boundaries clear enough to implement v0 contracts? Conditional yes.**

The module seams (retrieval → ingestion → lifecycle → context assembly → harness → cleanup) are well-drawn and the port objects in §2 of `CONSOLIDATED_PLAN.md` are implementable as Pydantic skeletons today. Three boundary issues, however, are still ambiguous enough to risk locking in wrong field shapes in v0:

1. **LLM-extracted claims defaulting to `active`** (Ted's new instruction) directly contradicts the checklist's "persist as `candidate`" default. This is a status-enum decision that v0 must encode correctly.
2. **LLM choosing B vs C grade** for news/web results blurs the retrieval-vs-LLM boundary. Whoever owns `source_grade` writes it to the registry; v0 must decide if that writer can be an LLM.
3. **Obsidian-style citations** were not in the original plan. v0 `DocumentRef` and the citation envelope inside `MemoContextPackage` need fields that survive into an `[[wikilink]]` later, or v1 will end up parsing display strings.

Everything else (cleanup modes, manifest, source taxonomy A/B/C, memo non-blocking, Markdown review queues, XOM demo target) can ship as proposed.

## 2. Questions Ted must answer before v0 implementation

These three cannot be safely defaulted by engineering because they change the shape of contracts:

1. **LLM-active scope.** Ted said LLM-extracted claims can default `active` because LLM extraction is "usually accurate now." Must clarify scope:
   - Active by default for **all** LLM extractions, or only **narrative/news** claims (qualitative: catalysts, management commentary, sentiment) while **quantitative/financial** claims (revenue, EPS, guidance numbers) still default `candidate`?
   - If a deterministic A-grade source later contradicts an LLM-active claim, should the LLM claim **auto-demote to candidate**, or stay active and flagged as a conflict?
   - Required so v0 can decide whether `claim_status` is a binary (active/candidate) or carries an `extraction_method` qualifier and demotion rules.

2. **LLM source-grading authority.** "LLM can determine whether grade should be B or C" — is this:
   - (a) LLM **proposes** grade; retrieval layer accepts it deterministically (LLM still writes to registry through a single API), or
   - (b) LLM **suggests** grade and a deterministic rule chooses final grade (e.g., known-domain whitelist forces B, everything else C)?
   - Choice (a) puts an LLM inside the deterministic retrieval module, which violates §1.1 "It does not own… durable fact/claim extraction." Choice (b) preserves the boundary. Need Ted's preference because either way changes the `SourceRecord` writer.

3. **Vault location for Obsidian wikilinks.** If citations must eventually render as `[[Title]]` or `[[ticker/10-K-2025#item-1a|Risk Factors]]`, then v0 must know:
   - Is the vault stored **inside an Obsidian vault directory** (so `vault_relative_path` and Obsidian's vault root agree), or
   - Is the vault separate and the memo gets **copied into** an Obsidian vault later?
   - This determines whether `DocumentRef.canonical_path` should be vault-relative (Obsidian-friendly) or absolute (filesystem-friendly).

## 3. Questions engineering can default without escalating

- File layout under `src/retrieval/` (modules per source class).
- Pydantic vs dataclass for ports (recommend Pydantic v2 for serialization parity with existing harness contracts).
- Exact severity thresholds for the conflict rule table.
- `claim_status` and `question_status` enum values (concrete names below).
- Markdown filenames for review queues (`question_list.md`, `review_queue.md`, `conflict_list.md`).
- Manual-file default grade: keep checklist's proposal — `B` with `manual_unverified` rationale unless the file's source_type is provably primary (e.g., a downloaded SEC PDF whose checksum matches an EDGAR fetch).
- Cleanup mode names and which files survive each mode.
- XOM as v0→v1 demo target (Ted already approved).

## 4. "Question auto-close" — plain explanation and recommendation

**What it means in product terms.** The lifecycle layer tracks open questions like *"What is FY2025 capex guidance?"* or *"Has management commented on the announced Permian divestiture?"*. When a new source arrives that appears to answer the question (e.g., the 10-K with capex numbers, or a transcript with the divestiture comment), there are three possible behaviors:

- **No auto-close:** question stays `open` until a human marks it `answered`.
- **Propose close:** system attaches the candidate answer + evidence link and flips status to `proposed_close`; human accepts in the review queue.
- **Auto-close:** system flips status to `answered` itself based on rules (e.g., A-grade source contains a regex match for the asked field).

**Recommendation: "Propose close" as the v0/v1 default.** Reasons:
- The plan explicitly forbids LLM auto-closing (§3.3), but a deterministic rule could in principle auto-close. Even so, a `proposed_close` step is cheap and keeps the human as final arbiter (Ted's stated preference).
- It costs one extra enum value (`proposed_close`) and a `proposed_answer_ref` field on `QuestionRecord`. No durable behavior is lost — v2 can enable narrow auto-close rules later for high-signal cases (e.g., 10-K headline metrics).
- Obvious gotcha: a question like "did management address X?" may have ambiguous matches. Propose-close avoids embarrassing false closures in the memo.

## 5. "Conflict severity" — plain explanation and recommendation

**What it means in product terms.** Two sources disagree about a fact — e.g., yfinance reports trailing revenue $344B while the latest 10-K reports $338B; or a vendor profile says HQ is in Spring, TX while a recent press release says Houston, TX. The lifecycle layer must:
1. **Detect** the disagreement (`ConflictRecord`).
2. **Rank** how urgently a human should look at it.
3. **Decide** whether the memo can still ship with the conflict open.

Severity is the ranking signal — it drives review-queue ordering and whether memo-context-assembly raises a caveat versus a warning versus a block.

**Recommendation: deterministic rule table for v1.** Concrete defaults:

| Conflict pair | Severity | Memo behavior |
|---|---|---|
| A-grade vs A-grade, same field, both current | `high` | Caveat in memo body; review queue top |
| A-grade vs B-grade, same field | `medium` | Caveat if material; otherwise note |
| B-grade vs B-grade | `low` | Note only |
| Any grade vs stale (older than freshness cutoff) | `low` | Use newer; log diff |
| Material-metric mismatch (revenue/EPS/guidance) on current A-grade | `blocker_review` | Warning surfaced in `MemoContextPackage`; non-blocking by Ted's rule but elevated in the queue |

This keeps it rule-based (no LLM), uses the A/B/C grading already accepted, and gives `MemoContextPackage` a single `severity` value to filter on. LLM may *suggest* severity in v2 as a candidate annotation, never overwriting the rule.

## 6. Critique of "LLM-extracted claims default to `active`"

Accepting Ted's preference is reasonable for news/web text where the LLM is doing routine extraction over short passages. But "active" must not erase provenance. Guardrails required if v0 encodes this:

**Required metadata on every LLM-extracted claim:**
- `extraction_method = "llm"` (never overwritten).
- `extractor_model` (e.g., `claude-opus-4-7`) and `extractor_version` (date or git sha of the prompt template).
- `extraction_run_id` linking back to the run manifest.
- `evidence_refs[]` pointing to `DocumentRef` + character/line span or exact quoted snippet (≥1 required; reject extraction without it).
- `source_grade` of the underlying source (B or C in Ted's scope) — does **not** become A just because the LLM is confident.
- `extracted_at` timestamp.
- `confidence` (optional, model-reported) — store but don't gate on it without calibration.

**Required guardrails:**
1. **Active scope restriction.** Only allow `active` default for **narrative/qualitative** fields (catalysts, risks, commentary, sentiment). Quantitative fields (revenue, EPS, guidance numbers, capex) stay `candidate` regardless of who extracted them, until promoted by a deterministic A-grade match or human review. This is the single most important guardrail — quantitative errors in a memo are unrecoverable, narrative errors are caveat-able.
2. **Auto-demotion on conflict.** If an LLM-active claim conflicts with a deterministic A-grade claim, the LLM claim auto-demotes to `candidate` and creates a `ConflictRecord`. This must happen during ingestion, not memo time.
3. **Diff log required.** Any LLM-active claim must produce a diff entry so the human can audit retroactively ("what did the LLM assert on this run?"). Without this, "candidate vs active" becomes invisible in the memo.
4. **Citation requirement enforced at write time.** A claim without `evidence_refs` cannot be persisted, full stop. This prevents the most likely LLM failure mode (asserting without grounding).
5. **Source-grade does not change.** An LLM-active claim sourced from a C-grade blog is still C-grade; the citation in the memo must show "C source" so the reader can discount it.

**One concrete v0 contract change:** add `claim_status: Literal["active", "candidate", "proposed_close", "rejected"]` and require `extraction_method`, `evidence_refs`, and `source_grade` non-empty for any persisted claim. This is the minimum surface area to encode Ted's policy without losing audit trail.

## 7. Obsidian citation boundary — what v0/v1 must store

Obsidian wikilinks have the form `[[path/to/note#heading|display text]]` (or `[[Note Title]]` if the file is uniquely named in the vault). For citations to render as Obsidian links later without re-parsing memo prose, the contracts need to store the *components* of the link, not the rendered link string.

**v0 `DocumentRef` additions:**
- `vault_relative_path: str` — path of the canonical document within the vault root, forward-slash, no leading slash (e.g., `companies/XOM/sources/sec_filings/10-K-2024.md`). Stable and unique; this becomes the wikilink target.
- `display_title: str` — human-readable title for the link's display text (e.g., "XOM 10-K (FY2024)").
- `anchor_strategy: Literal["heading", "block_id", "none"]` — how downstream citations point into the doc.

**v0 `EvidenceRef` (used inside claims/research_objects/conflicts) additions:**
- `doc_ref` — opaque ref already exists.
- `heading_path: list[str]` — sequence of Markdown headings leading to the cited passage (e.g., `["Item 1A. Risk Factors", "Commodity Prices"]`); converts to `#item-1a-risk-factors-commodity-prices` in Obsidian.
- `quoted_snippet: str` — exact substring from the source (already partially in the plan).
- `start_offset / end_offset: int | None` — character offsets in `extracted_text_path` for reproducible highlight.

**v0 `MemoContextPackage` citation envelope:**
- Replace any free-text citation strings with a `citations: list[Citation]` table where each entry has `{citation_key, doc_ref, vault_relative_path, display_title, heading_path, snippet, source_grade}`.
- The harness then renders each in-memo reference by key, e.g., `[^xom_10k_riskfactors]`, and the rendering function for Obsidian output produces `[[companies/XOM/sources/sec_filings/10-K-2024#item-1a-risk-factors|XOM 10-K — Risk Factors]]`.

**Vault filesystem convention v0 should adopt:**
```
vault/
  companies/<ticker>/
    sources/
      sec_filings/
      ir/
      news/
      manual/
    research/
      source_index.md
      question_list.md
      review_queue.md
      conflict_list.md
      memos/<run_id>.md
```
If this matches Ted's actual Obsidian vault layout, point `vault_root` at it directly; if not, the vault remains a faithful subtree that can be copied or symlinked into Obsidian.

## 8. Minimal contract changes needed before coding v0

In order of necessity:

1. **`ResearchTask`** — add `vault_root: Path` (so citation paths can be made vault-relative deterministically).
2. **`SourceRecord` / `DocumentRef`** — add `vault_relative_path`, `display_title`; clarify `source_grade` writer (deterministic only; LLM-suggested grade lives in a separate `proposed_source_grade` field until Ted picks 2(a) vs 2(b) above).
3. **`ClaimRecord` (or whatever lives under `research_object_records`)** — add:
   - `claim_status: Literal["active", "candidate", "proposed_close", "rejected"]`
   - `extraction_method: Literal["deterministic", "llm", "manual"]`
   - `extractor_model: str | None`, `extractor_version: str | None`
   - `evidence_refs: list[EvidenceRef]` (≥1 required)
   - `field_kind: Literal["quantitative", "narrative"]` (so the "narrative LLM may be active, quantitative may not" rule is encodable as a single check).
4. **`EvidenceRef`** — add `heading_path`, `quoted_snippet`, `start_offset`, `end_offset`.
5. **`QuestionRecord`** — add `question_status: Literal["open", "proposed_close", "answered", "rejected"]` and `proposed_answer_ref`.
6. **`ConflictRecord`** — add `severity: Literal["low", "medium", "high", "blocker_review"]` and `severity_rule_id: str` (for auditability of how severity was assigned).
7. **`MemoContextPackage`** — replace free-text citation handling with a `citations: list[Citation]` table as in §7; add `caveats: list[Caveat]` and `blockers: list[Blocker]` (blockers empty by default per Ted's non-blocking rule but the field must exist).
8. **`SourceGrade`** type — keep `A | B | C | unknown`; do **not** introduce `manual_unverified` as a grade. Use `B` plus `source_grade_rationale="manual file, provenance not machine-verified"` as the checklist already proposed.
9. **`RetrievalRequest`** — keep `allow_llm_discovery: bool` already in the plan; add `allow_llm_grading: bool` defaulting `False` so engineering can flip it on once Ted answers question 2.

No other v0 contract changes are required. Everything else (cleanup, manifest, retention) is already specified.

---

## Escalate to Ted

1. LLM-active claim scope: all extractions, or narrative-only? Auto-demote on A-grade conflict?
2. LLM source-grading: does LLM write `source_grade` directly, or only `proposed_source_grade` with a deterministic rule choosing final?
3. Vault location: is the vault inside an Obsidian vault, or copied/synced into one?

## Can decide now

- v0 contract field shapes and enum names listed in §8.
- `src/retrieval/` package layout.
- Question auto-close default: `proposed_close`, no full auto-close in v0/v1.
- Conflict severity: deterministic rule table (§5), no LLM severity in v0/v1.
- Manual file default grade: `B` with `manual_unverified` rationale unless source_type is provably primary.
- Memo non-blocking: confirmed by Ted; just need `blockers: list[Blocker]` field present-but-empty.
- Markdown review queues for v1 (`question_list.md`, `review_queue.md`, `conflict_list.md`).
- XOM as v0→v1 demo target.
- Cleanup modes (`debug_keep_all`, `product_final_only`, `archive_on_failure`).
- Obsidian wikilink rendering deferred to harness; v0 stores components, not link strings.
