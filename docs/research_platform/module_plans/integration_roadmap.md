# Integration Architecture, Module Ports, Staged Roadmap, and Test Strategy

Last updated: 2026-05-13  
Planner: `integration_roadmap_planner`  
Scope: planning only; no implementation or commits.

## 1. Module purpose

This module plan defines the integration seams for AlphaSeeker's shift from a one-shot investment memo harness toward a persistent, source-aware equity research operating system. The integration layer should not become a feature module that owns retrieval, ingestion, lifecycle, or memo writing. Its purpose is to make the ports between those modules explicit so each can be improved independently.

The target end-to-end flow is:

```text
ResearchTask(company/ticker + prompt + optional manual files)
  -> retrieval layer returns SourceRecord batch
  -> ingestion registers durable source documents
  -> lifecycle/index applies validated ResearchStateDelta objects
  -> context assembly builds MemoContextPackage
  -> harness generates OutputRunResult / final memo
  -> cleanup/vault feedback preserves outputs and validated deltas
```

The key product rule is that durable state should be source-grounded, inspectable, and versionable. `llm_research_state.md` is useful as a transitional rendered artifact, but should not become the canonical source of truth.

## 2. Owned responsibilities

The integration layer owns:

- the cross-module port map;
- shared Pydantic/dataclass contract definitions;
- staged coding order across modules;
- validation gates at every seam;
- migration guidance from the current MVP;
- identification of product-policy decisions Ted/orchestrator must make.

It should provide thin adapters around current code (`src/vault/ingest.py`, `src/vault/synthesis.py`, `src/vault/memo_flow.py`, `src/harness/*`) before deeper refactors.

## 3. Explicit non-responsibilities

Integration does not own fetcher internals, parser internals, conflict-arbitration policy, memo-agent prompting, final wiki aesthetics, or investment judgment. It should not decide whether a source is believable in context, whether a thesis changes, or what the fund should do. Those belong to deterministic module policies, LLM proposals, and ultimately Ted/human judgment.

It also should not implement code in this planning phase.

## 4. Inputs / upstream ports

The top-level input should become a normalized `ResearchTask`, rather than overloading a prompt string:

```python
class ResearchTask(BaseModel):
    task_id: str | None
    task_type: Literal["memo", "onboarding", "meeting_prep", "monitoring"]
    ticker: str
    company_name: str | None
    user_prompt: str
    manual_source_paths: list[str]
    requested_outputs: list[str]
    live_mode: bool = False
    cleanup_policy: CleanupPolicy
```

Retrieval receives:

```python
class RetrievalRequest(BaseModel):
    request_id: str
    ticker: str
    company_name: str | None
    task_type: str
    query_classes: list[str]  # manual_files, sec_filings, profile, financials, news, etc.
    manual_paths: list[str] = []
    form_types: list[str] = ["10-K", "10-Q", "8-K"]
    max_sources: int = 20
    deterministic_first: bool = True
    allow_llm_discovery: bool = False
```

Retrieval returns `RetrievalBatch(records=list[SourceRecord], failures=list[...], warnings=list[str])`. Retrieval should not mutate the durable vault; it only discovers/fetches candidates.

## 5. Outputs / downstream ports

Core shared contracts:

```python
class SourceRecord(BaseModel):
    source_id: str
    ticker: str | None
    source_type: str
    title: str | None
    url: str | None
    local_path: str | None
    text: str | None
    published_at: str | None
    retrieved_at: str
    source_grade: Literal["A", "B", "C", "unknown"]
    provenance: dict[str, Any]
    checksum: str | None
    relevance: Literal["primary", "mentioned", "candidate"] = "candidate"
```

```python
class IngestionResult(BaseModel):
    ingestion_id: str
    source_record_id: str
    ticker: str
    doc_id: str
    source_type: str
    title: str | None
    source_grade: str
    original_path: str | None
    extracted_path: str
    metadata_path: str
    checksum: str
    text_chars: int
    warnings: list[str] = []
```

```python
class ResearchStateDelta(BaseModel):
    delta_id: str
    ticker: str
    produced_by: Literal["deterministic", "llm", "human", "importer"]
    source_doc_ids: list[str]
    facts: list[dict] = []
    metrics: list[dict] = []
    questions: list[QuestionRecord] = []
    conflicts: list[ConflictRecord] = []
    citations: list[dict] = []
    status: Literal["proposed", "validated", "applied", "rejected", "needs_human"]
    raw_artifact_path: str | None
```

`MemoContextPackage` should be the harness input wrapper: manifest path, context files, included source IDs, question/conflict IDs, source/citation policy, warnings, and token/char budget. `OutputRunResult` should adapt current `HarnessResponse` with run ID, status, run root, final report path, context package ID, generated artifacts, proposed delta path, cleanup manifest path, and error.

## 6. Persistent artifacts and schema needs

Reuse current vault primitives: SQLite `documents`, `companies`, `document_companies`, `facts`, `metrics`, `questions`, `answers`, `conflicts`, `wiki_versions`; document folders with `original.*`, `extracted.md`, `metadata.json`; company pages like `source_index.md`, `question_list.md`, `conflicts.md`, `status_patrol.md`, `wiki.md`; and harness run workspaces with `context/` and `publish/final.md`.

Add lightweight JSON manifests before heavy schema migrations:

```text
data/research_vault/
  retrieval_batches/<batch_id>.json
  companies/<TICKER>/
    context_packages/<package_id>/manifest.json
    context_packages/<package_id>/context_package.md
    deltas/<delta_id>.json
    output_runs/<run_id>/output_run_result.json
    output_runs/<run_id>/final.md
    output_runs/<run_id>/cleanup_manifest.json
```

Later schema needs: first-class `state_deltas`, `citations`, `research_sections`, and `memo_versions` tables.

## 7. Deterministic code vs LLM decision boundary

Deterministic code should own source fetching where APIs/rules are known, checksums, dedupe, ingestion, source registration, explicit ticker links, mechanical source-grade defaults, context budgeting, schema validation, simple freshness/QC checks, and cleanup safety.

LLMs may synthesize source bundles, propose claims/questions/conflicts with citations, rank ambiguous source candidates, and write memo/deck outputs. LLM output should enter durable state only as a validated `ResearchStateDelta` or human-reviewed update.

Humans/Ted own investment judgment, source credibility in context, thesis changes, and policy decisions such as retention and source-grade taxonomy.

## 8. Failure modes and human escalation points

Escalate or explicitly warn on: no A-grade source coverage; failed extraction of important PDFs/models; unclear proprietary-source policy; invalid LLM JSON; citations that do not map to source excerpts; A-vs-A conflicts; stale source inventory; context package too large to fit required sources; harness timeout without final memo; final memo with unsupported claims; and any cleanup action that might delete durable source/vault artifacts.

Default early behavior should preserve artifacts and produce review questions rather than silently dropping data or auto-resolving ambiguity.

## 9. Versioned/staged coding plan

**v0 — contracts/skeletons only.** Define contract models and skeleton modules, likely `src/research_platform/contracts.py`, `retrieval.py`, `ingestion.py`, `lifecycle.py`, `context.py`, `output.py`, `cleanup.py`, and `pipeline.py`. These should adapt current functions but not replace existing flows. Add dry-run no-LLM tests.

**v1 — first useful vertical slice.** Support manual local sources first. Convert them to `SourceRecord`, ingest through current vault ingestion, reuse current LLM synthesis to produce a parsed delta plus rendered `llm_research_state.md`, persist open questions, build `MemoContextPackage`, call existing harness via `HarnessRequest.context_files`, then record/copy final memo and cleanup manifest under the company vault. Keep `llm_research_state.md` as a rendered view, not canonical truth.

**v1.1 — stabilize current MVP under contracts.** Add retrieval batch manifests, wrap SEC/profile/financials fetchers, validate LLM citations against included sources, add context budget rules, and link output runs to company vault.

**v2+ — expand modules locally.** Add daily monitoring retrieval, Excel/model registry and diffs, OCR/ASR, first-class citations/deltas/research sections, conflict scanner, meeting-prep packages, deck outputs, evaluator citation checks, human review queue, and retention policies.

## 10. Tests / validation gates

Required gates:

- contract serialization/validation tests for all shared models;
- golden local source bundle test for retrieval -> ingestion -> source index;
- seam tests for retrieval/ingestion, ingestion/lifecycle, lifecycle/context, context/harness, harness/cleanup;
- fake-LLM synthesis tests for valid JSON, fenced JSON, invalid JSON, missing citation doc IDs, and quote mismatch;
- mocked harness integration test proving context files are copied and prompt references the package without inlining;
- cleanup safety tests proving final outputs/context packages/raw vault docs are preserved and deletion stays inside harness scratch;
- optional live tests for SEC/profile/financials, real LLM synthesis, and real memo harness.

Default CI should remain deterministic and no-LLM/no-network.

## 11. Open questions for Ted/orchestrator

- What is the exact source-grade taxonomy and default mapping?
- Can LLM-generated thesis/key-takeaway sections be persisted automatically, or only proposed for review?
- Is MVP citation policy document path + quote snippet, or fully structured claim-level citations?
- What proprietary-source retention rules apply to broker reports, models, notes, and transcripts?
- Should v1 include deterministic SEC/profile retrieval or manual sources only?
- What cleanup retention policy is acceptable for harness traces and raw LLM responses?
- Should new modules live under `src/research_platform/` or remain under `src/vault/`?
- What company/source set defines the first accepted demo?

## 12. Interactions with other modules

Retrieval supplies `SourceRecord`s to ingestion and should not write the vault. Ingestion turns sources into durable documents and lineage. Lifecycle validates/applies `ResearchStateDelta`s and renders source/question/conflict/status pages. Context assembly reads vault state and emits task-specific `MemoContextPackage`s without fetching or mutating. Harness consumes context packages and returns `OutputRunResult`; it should not directly create canonical vault state. Cleanup/vault feedback records final outputs, preserves context manifests, and sends any proposed deltas back through lifecycle validation.

Migration principle: wrap first, split second, migrate durable schema last. This keeps useful current code while moving away from the prototype assumption that `llm_research_state.md` is the durable research state.