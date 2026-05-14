# AlphaSeeker Research Platform — Consolidated Module Plan

Last updated: 2026-05-13  
Orchestrator: `codex`  
Inputs: five avatar module plans under `docs/research_platform/module_plans/`

## 0. Context and decision

Ted's instruction was explicit: **do not implement improvements yet**. First define the modules, their ports, their decision boundaries, and a staged coding plan. This document consolidates the five module plans:

- `input_retrieval_layer.md`
- `vault_ingestion_registry.md`
- `research_lifecycle.md`
- `memo_output_harness.md`
- `integration_roadmap.md`

The current `llm_research_state.md` MVP is useful as a prototype, but it should not become the durable architecture. The durable research platform should center on source-grounded objects, source lineage, questions, conflicts, freshness, and review lifecycle. Memo/deck/briefing generation should be an output view over prepared research state, not the place where source acquisition and QC are reinvented every run.

Target architecture:

```text
ResearchTask / ticker / optional manual files
  -> Input Retrieval Layer
  -> Deterministic Vault Ingestion + Source Registry
  -> Research Lifecycle Layer
  -> Task Context Assembly
  -> Memo / Output Harness
  -> Cleanup + Vault Feedback
```

## 1. Module boundaries

### 1.1 Input Retrieval Layer

**Purpose:** convert a user task into a normalized batch of source materials.

It owns:

- interpreting workflow type: `onboard`, `memo`, `monitor`, `meeting_prep`, later `meeting_followup`;
- applying source acquisition policy for the workflow;
- deterministic retrieval using existing low-level tools under `src/tools/equity/*`;
- staging retrieved files into a retrieval workspace;
- assigning initial source class/grade/rationale according to policy;
- producing a `RetrievalBatch` manifest.

It does **not** own:

- vault DB writes;
- durable fact/claim extraction;
- conflict arbitration;
- investment memo generation;
- human investment judgment.

Existing code to reuse/move downward:

- `src.tools.equity.sec_filings.search_and_read_filings`
- `src.tools.equity.company_profile.fetch_company_profile`
- `src.tools.equity.financials.fetch_financial_metrics`
- `src.tools.equity.market_data.fetch_historical_data`
- earnings/IR/peer/news tools where appropriate.

Current `src/harness/skills/equity.py` should gradually become a thin wrapper around the same retrieval primitives, not the primary owner of source acquisition policy.

### 1.2 Deterministic Vault Ingestion + Source Registry

**Purpose:** make source materials durable, inspectable, deduplicated, and citeable.

It owns:

- validating `RetrievalBatch` or manual source metadata;
- copying/storing raw originals and extracted text;
- calculating checksums;
- registering documents and company links;
- recording source type, source grade, timestamps, origin URL/path, fetcher/parser versions, extraction status/errors;
- deterministic dedupe and version handling;
- returning stable document references to downstream modules.

It does **not** own:

- deciding what to retrieve;
- LLM summarization;
- durable investment conclusions;
- conflict arbitration;
- memo output.

This module should be deterministic. If parsing fails, it records failure explicitly; it should not silently claim success.

### 1.3 Research Lifecycle Layer

**Purpose:** maintain durable source-grounded research state.

It owns:

- `source_index` views;
- `question_list` lifecycle;
- source-grounded research objects: facts, claims, metrics, guidance, catalysts, risks, thesis-point candidates;
- evidence links from each object to source documents/spans/snippets;
- conflict detection and review queue;
- freshness/staleness checks;
- diff logs and version history;
- human review states.

It does **not** own:

- raw source retrieval;
- raw document ingestion;
- output harness orchestration;
- final investment decisions.

The key architectural correction: replace durable freeform `llm_research_state.md` with durable structured objects and generated views. A Markdown wiki/page can still exist, but it should be a rendered view over source-grounded objects, not the canonical truth.

### 1.4 Task Context Assembly

This is a seam between lifecycle and output harness. It can initially live near the harness wrapper, but it should be conceptually separate.

It owns:

- selecting relevant source docs/excerpts;
- bundling `source_index`, open questions, conflicts/review warnings, and selected evidence into a bounded package;
- preserving checksums and manifest references for reproducibility.

It does **not** own:

- source retrieval;
- vault writes;
- final memo generation;
- human review decisions.

This module produces `MemoContextPackage` or analogous packages for meeting prep / monitoring / deck generation.

### 1.5 Memo / Output Harness

**Purpose:** generate deliverables from a prepared context package.

It owns:

- consuming `MemoContextPackage`;
- running the existing multi-agent harness;
- producing `final.md` and minimal output manifest/status;
- optionally producing deck/briefing variants later;
- applying cleanup policy to temporary run artifacts.

It does **not** own:

- deciding what sources to retrieve;
- mutating durable vault truth directly;
- adjudicating conflicts;
- silently closing questions;
- source grading.

The harness may propose thesis points, follow-up questions, or memo-derived candidates, but those should feed back as pending lifecycle/review items, not auto-updated truth.

## 2. End-to-end port map

Recommended contract flow:

```text
ResearchTask
  -> RetrievalRequest
  -> RetrievalBatch[SourceRecord]
  -> IngestionResult[DocumentRef]
  -> ResearchStateDelta / CompanyResearchContext
  -> MemoContextPackage
  -> OutputRunResult / MemoResultPackage
  -> CleanupResult + VaultFeedbackRequest
```

### 2.1 ResearchTask

Minimal fields:

```text
ticker
company_name optional
workflow_type: onboard | memo | monitor | meeting_prep | meeting_followup
user_prompt optional
manual_files optional
budget / freshness policy optional
run_id optional
```

### 2.2 RetrievalRequest

Derived from `ResearchTask` plus retrieval policy:

```text
ticker
company_name
workflow_type
required_source_classes
optional_source_classes
manual_files
retrieval_budget
allow_llm_discovery: bool
freshness_cutoffs
```

### 2.3 SourceRecord

Output from retrieval layer before ingestion:

```text
source_key
local_path
title
source_type
source_grade: A | B | C | unknown
source_grade_rationale
url / original_path
retrieved_at
published_at / filed_at
fetcher_name
fetcher_version
retrieval_method: deterministic | llm_assisted_discovery | manual
metadata
```

### 2.4 RetrievalBatch

```text
batch_id
request
records: list[SourceRecord]
missing_required_sources
retrieval_warnings
manifest_path
```

### 2.5 IngestionResult / DocumentRef

```text
ingestion_id
documents: list[DocumentRef]
duplicates
new_versions
parse_failures
warnings
```

`DocumentRef` should expose stable, safe references:

```text
doc_ref
company/ticker
canonical_path
extracted_text_path
source_type
source_grade
published_at/filed_at
checksum
version
```

Internal DB IDs can exist, but user-facing docs should prefer title/path/source label/checksum snippets instead of opaque private IDs.

### 2.6 ResearchStateDelta / CompanyResearchContext

Lifecycle output:

```text
source_index_view
question_records
research_object_records
conflict_records
review_queue_items
freshness_warnings
diff_log_entries
```

### 2.7 MemoContextPackage

Prepared for output harness:

```text
ticker/company
user_prompt
source_index_path or rendered text
question_list_path or rendered text
selected_source_excerpts or context files
conflict/review warnings
freshness warnings
manifest with doc checksums/source grades
context_package_path
```

### 2.8 OutputRunResult / MemoResultPackage

```text
run_id
status
final_report_path
kept_artifacts
cleanup_policy
source_manifest_path
memo_checksum
proposed_followup_questions
proposed_thesis_candidates
warnings
```

## 3. Deterministic code vs LLM boundaries

### 3.1 Deterministic by default

These must be deterministic or explicitly rule-based:

- source acquisition policy for workflow type;
- SEC filings retrieval when deterministic APIs work;
- source staging and file manifests;
- ingestion, checksum, dedupe, versioning;
- source grade recording according to declared policy;
- source index rendering from registry;
- artifact cleanup;
- schema migration and tests.

### 3.2 LLM-assisted but bounded

LLM may help with:

- discovering ambiguous IR pages or transcripts when deterministic lookup fails;
- classifying relevance among candidate sources;
- extracting candidate claims/facts/guidance/questions from source text;
- generating draft briefings/memos from prepared context;
- suggesting follow-up questions.

### 3.3 LLM must not do these without review/guardrails

LLM must not:

- fabricate missing sources;
- mark a required source as unnecessary without policy rule or human approval;
- assign A-grade outside source-grade rules;
- resolve conflicts as truth;
- mutate durable research truth directly;
- auto-close questions without evidence and lifecycle rules;
- make investment decisions on behalf of the human.

## 4. Source-grade policy baseline

Recommended default taxonomy for v0/v1:

```text
A-grade: company/issuer primary sources, SEC filings, official earnings releases, official transcripts/presentations, exchange/regulator filings.
B-grade: reputable market-data/vendor snapshots, yfinance-style financial/profile data, major media, credible sell-side/third-party data if labeled.
C-grade: blogs, social media, unverified commentary, scraped summaries, LLM-discovered pages without strong provenance.
```

Important: B/C can be useful, but must not be silently promoted into confirmed facts. They can generate questions, conflicts, or preliminary support.

## 5. Artifact ownership and retention

Persistent vault keeps:

- raw source originals;
- extracted text;
- source registry metadata;
- source index views;
- questions/answers lifecycle;
- research objects/evidence links;
- conflicts/review queue;
- freshness/diff history;
- final memo references/manifests.

Harness keeps by default only:

- `final.md`;
- minimal `summary.md` or final run summary;
- source/output manifest;
- status/error record;
- maybe `artifact_index.md`.

Temporary by default:

- agent scratch files;
- intermediate tool outputs;
- temporary source package copies;
- planner/evaluator state.

Need cleanup modes:

```text
debug_keep_all        for development/failure diagnosis
product_final_only    for clean user-facing product runs
archive_on_failure    keep enough artifacts only when run fails or evaluator flags issue
```

Cleanup must never delete vault source originals or final memo artifacts.

## 6. Staged coding roadmap

### v0 — Contracts and skeletons only

Goal: define module seams without feature expansion.

Deliverables:

- Pydantic/dataclass contract definitions:
  - `ResearchTask`
  - `RetrievalRequest`
  - `SourceRecord`
  - `RetrievalBatch`
  - `IngestionResult`
  - `DocumentRef`
  - `ResearchStateDelta`
  - `QuestionRecord`
  - `ConflictRecord`
  - `MemoContextPackage`
  - `OutputRunResult`
  - `CleanupPolicy`
- Empty/skeleton modules, likely:
  - `src/input/` or `src/retrieval/`
  - `src/vault/registry.py` or enhanced ingestion module
  - `src/vault/lifecycle.py`
  - `src/vault/context_package.py`
  - cleanup helper in `src/harness/` or wrapper layer.
- Update docs to mark `llm_research_state.md` as transitional.
- No live retrieval requirements yet.

Validation:

- contract serialization tests;
- no-LLM unit tests;
- golden fake `RetrievalBatch` -> fake `IngestionResult` -> fake `MemoContextPackage` flow.

### v1 — First useful deterministic vertical slice

Goal: one equity memo/onboarding run using clearly separated modules.

Scope:

- manual source files + deterministic SEC/profile/financial/market retrieval where stable;
- deterministic ingestion into vault;
- render `source_index.md`;
- seed/update `question_list.md` from missing required sources and basic lifecycle rules;
- assemble `MemoContextPackage` with selected docs/excerpts + source index + questions;
- run existing harness as consumer of that package;
- keep final memo + manifest, optional debug artifacts;
- no durable `llm_research_state.md` as canonical state.

Validation:

- focused unit tests for each seam;
- integration test with local golden source bundle;
- mocked retrieval for deterministic CI;
- optional live XOM smoke test with explicit live marker;
- cleanup safety test verifying vault originals survive and harness scratch is removed under product cleanup.

### v1.1 — Retrieval hardening and citation validation

Scope:

- move/wrap SEC/profile/financial/market tools into input layer policy;
- stronger source-grade recording;
- validate citation/document refs in context packages;
- source selection/excerpt limits;
- run manifest checksum reproducibility.

### v2 — Research lifecycle expansion

Scope:

- source-grounded `research_objects` envelope;
- LLM candidate extraction into pending/review status;
- conflict scanner and review queue;
- freshness/staleness policies;
- diff logs;
- generated company views from structured state.

### v3 — Continuous workflows

Scope:

- daily/periodic monitor command;
- meeting prep command;
- meeting follow-up ingestion and question answer backfill;
- model registry/diff for analyst model snapshots;
- deck/briefing output variants.

## 7. Test and validation gates

### Contract tests

- Every port object serializes/deserializes without losing fields.
- Required fields are enforced.
- Unknown/optional metadata survives roundtrip.

### Retrieval tests

- `ResearchTask` + policy -> expected required/optional source classes.
- Missing required source is recorded, not hidden.
- LLM-assisted discovery flag is off by default in deterministic unit tests.

### Ingestion tests

- same file/checksum ingestion is idempotent;
- same source key with changed checksum creates new version/event;
- parser failure creates explicit error state;
- source grade and timestamps are preserved.

### Lifecycle tests

- questions have lifecycle states and evidence refs;
- stale warnings fire from deterministic dates;
- conflicts create review queue items, not resolved truth;
- generated views are reproducible from DB state.

### Context package tests

- package includes source index, questions, selected source refs, checksums;
- package rejects unknown document refs;
- source excerpt limits are enforced.

### Harness cleanup tests

- product cleanup keeps final memo/manifest/status;
- debug mode keeps full workspace;
- cleanup never deletes vault files;
- failed runs preserve enough diagnostics.

### End-to-end smoke tests

- local golden source bundle with no network;
- optional live XOM run behind marker/env flag;
- compare manifest/final paths rather than exact memo prose.

## 8. Open decisions for Ted

These should be clarified before implementation begins or as first v0 design questions:

1. **Package location/name:** Should new input layer live under `src/input/`, `src/retrieval/`, or `src/vault/retrieval.py`?
2. **v1 retrieval scope:** For the first separated workflow, should v1 include only manual + SEC filings, or also profile/financial/market snapshots?
3. **Source-grade taxonomy:** Is A/B/C baseline above acceptable, or do you want more nuanced grades like A1/A2/B/C?
4. **LLM extraction persistence:** Can LLM-extracted claims persist automatically as `candidate` records, or must all LLM-derived objects remain temporary until human review?
5. **Question lifecycle ownership:** Should the system auto-close questions when an A-grade source answers them, or only propose closure for human approval?
6. **Conflict severity:** Should conflict priority be rule-based initially, or may LLM suggest priority labels?
7. **Final artifact retention:** For product runs, keep only `final.md + manifest + status`, or also keep `summary.md` and `artifact_index.md`?
8. **Proprietary/manual docs:** Should manual files default to A-grade if uploaded by the human, or should they retain explicit `manual_unknown` until classified?
9. **Demo acceptance:** What is the first demo target: XOM memo, generic company onboarding, or source-index/question-list generation without memo?
10. **Human review UI:** For v1/v2, is Markdown review queue enough, or should we plan CLI commands for accepting/rejecting items?

## 9. Resolved decisions from Ted

Ted replied on 2026-05-13 with these decisions:

1. **Module location:** use `src/retrieval/` for the input/source-acquisition layer. Rationale: retrieval is upstream of the vault and should hand a `RetrievalBatch` to deterministic vault ingestion rather than living inside `src/vault/`.
2. **v1 retrieval scope:** include company profile, financial snapshot, and market snapshot in addition to manual files and SEC filings.
3. **Source grading:** use simple `A/B/C` grading; no A1/A2 nuance for now.
4. **LLM-derived claims:** LLM-extracted claims may persist automatically, but should remain explicitly marked as LLM-derived/candidate/source-cited rather than silently equivalent to deterministic exact extraction.
5. **Output retention:** product/default runs should keep only `final.md + manifest + status`; full harness scratch should be debug/failure mode only.

A **manifest** is the small reproducibility receipt for a run: run id, ticker, prompt reference, source document refs/checksums/grades, final output path, and status.

## 10. Recommended next action

Proceed to **v0 contracts/skeletons only**, with tests, before adding retrieval/features. This matches Ted's staged philosophy: define modules clearly first, then enhance each module in later versions without moving boundaries.
