# Input / Retrieval Layer Module Plan

Last updated: 2026-05-13  
Planner: `input_retriever_planner`  
Scope: planning only; no implementation.

## 1. Module purpose

The input/retrieval layer turns a normalized research request such as `memo XOM`, `onboard XOM`, `monitor XOM`, or `meeting prep XOM` plus optional manual files into a source acquisition policy and a normalized retrieval batch. It is the platform's front door for source material: deterministic fetch first, bounded LLM-assisted discovery only when deterministic code cannot locate or choose a source, and a manifest handoff to deterministic vault ingestion.

The output is **source files plus manifests**, not research conclusions and not ingested database rows. This keeps AlphaSeeker aligned with the Librarian direction: persistent state should be source-grounded, inspectable, versionable, and graded at source entry rather than being a freeform LLM wiki.

## 2. Owned responsibilities

- Parse normalized ticker/workflow requests into default acquisition policies.
- Define required and optional source classes by workflow.
- Execute deterministic retrieval adapters for SEC filings, market/profile/financial snapshots, manual files, and later IR/news/transcripts.
- Stage all retrieved/manual files into a batch directory with checksums and metadata.
- Assign initial document-level source grades A/B/C with explicit reasons.
- Deduplicate by canonical URL, SEC accession/filing identity, provider snapshot key, and checksum.
- Record source-class status: `ok`, `skipped_by_policy`, `not_found`, `partial`, `failed`, or `needs_human`.
- Emit human-escalation records for missing required sources, ticker ambiguity, low-confidence discovery, paid-source boundaries, or manual-file classification ambiguity.

Workflow policy defaults:

- `onboard`: broad initial coverage. Required: latest 10-K/10-Q/material 8-Ks, company profile, financial snapshot, market snapshot, all manual files. Optional/recommended: earnings release, call transcript, investor deck, peer metrics, 90-day news, insider activity.
- `memo`: current evidence sufficient for a memo. Required: latest annual/quarterly filing where applicable, profile, market snapshot, financial snapshot, latest earnings material if available. Optional: recent 8-Ks, investor deck, peer metrics, 30-90 day news, manual files.
- `monitor`: incremental focus-list updates. Required: new filings since last run, market snapshot, deterministic query templates for company news/regulatory/competitor/announcements. Optional: new IR materials, earnings calendar, insider activity.
- `meeting_prep`: recent context and user materials. Required: manual question/material files if supplied, latest source index reference, new filings/news/earnings since last refresh. Optional: latest call transcript/deck.

## 3. Explicit non-responsibilities

The retrieval layer must not:

- Write vault DB rows or call `VaultStore.insert_document`, `ingest_file`, or `ingest_text`.
- Extract durable facts, claims, metrics, guidance, catalysts, risks, questions, or conflicts.
- Resolve cross-source conflicts or decide which source is more believable in context.
- Make investment judgments, thesis updates, long/short/hold decisions, or memo conclusions.
- Write final memos/decks or clean harness scratch workspaces.
- Generate charts; `plot_price_history` remains output/harness work.
- Treat LLM summaries as durable source truth. If an LLM helps discovery, its output is audit metadata only.

## 4. Inputs / upstream ports

Core API sketch:

```python
def retrieve_sources(
    ticker: str,
    workflow_type: Literal["onboard", "memo", "monitor", "meeting_prep", "custom"],
    policy: SourceAcquisitionPolicy | None = None,
    manual_files: list[ManualSourceInput] | None = None,
    company_name: str | None = None,
    prior_source_index_path: str | None = None,
    output_root: str | None = None,
    allow_llm_discovery: bool = True,
    allow_paid_sources: bool = False,
) -> RetrievalBatch: ...
```

Important input objects:

```python
class SourceAcquisitionPolicy:
    workflow_type: str
    required_sources: list[SourceRequirement]
    optional_sources: list[SourceRequirement]
    date_windows: dict[str, DateWindow]
    max_items: dict[str, int]
    provider_preferences: dict[str, list[str]]
    fallback_strategy: Literal["deterministic_only", "llm_after_deterministic", "human_only"]
    missing_required_behavior: Literal["fail_batch", "partial_with_escalation"]

class ManualSourceInput:
    path: str
    source_class_hint: str | None
    source_grade_hint: Literal["A", "B", "C"] | None
    title: str | None
    published_at: str | None
    origin_url: str | None
    notes: str | None
```

CLI sketch:

```bash
python -m src.retrieval.cli retrieve-sources \
  --ticker XOM \
  --workflow memo \
  --manual-file ./inbox/xom_model.xlsx:trusted_model:A \
  --output-root data/research_vault/source_batches
```

## 5. Outputs / downstream ports

Main return object:

```python
class RetrievalBatch:
    batch_id: str
    ticker: str
    company_name: str | None
    workflow_type: str
    created_at: str
    batch_dir: str
    manifest_path: str
    policy: SourceAcquisitionPolicy
    sources: list[RetrievedSource]
    source_class_status: dict[str, SourceClassStatus]
    failures: list[RetrievalFailure]
    human_escalations: list[HumanEscalation]
```

Filesystem contract:

```text
data/research_vault/source_batches/<batch_id>/
  retrieval_batch.json
  sources/<source_id>/
    original.<ext>              # if available
    content.md or extracted.md  # if fetcher naturally returned text
    metadata.json
  logs/retrieval.log
  logs/discovery_candidates.json
```

Downstream ingestion should consume only the manifest and local paths:

```python
ingest_retrieval_batch(manifest_path, root="data/research_vault")
```

## 6. Persistent artifacts and schema needs

`retrieval_batch.json` should be the stable handoff. Minimum per-source fields:

- `source_id`, `ticker`, `source_class`, `source_type_for_vault`
- `title`, `local_path`, `original_path`, `metadata_path`
- `url`, `canonical_url`, `provider`, `published_at`, `retrieved_at`
- `source_grade`, `grade_reason`, `grade_status`, `grade_assigned_by`
- `checksum_sha256`, `dedup_key`, `retrieval_method`
- source-specific fields such as `form_type`, `accession_number`, `period`
- `quality_flags`, `ingestion_hints`

Current vault `documents` columns already support many fields (`source_type`, `title`, `path`, `url`, `published_at`, `source_grade`, `checksum`, `metadata_json`). New retrieval metadata can initially live inside `metadata_json`: `retrieval_batch_id`, `source_class`, `provider`, `canonical_url`, `dedup_key`, `grade_reason`, `retrieval_method`, and `quality_flags`.

Recommendation: keep manifests/logs permanently for audit. Originals may be pruned after checksum-confirmed ingestion only if Ted wants disk control.

## 7. Deterministic code vs LLM decision boundary

Deterministic code always runs first. Deterministic includes SEC APIs, provider APIs, fixed query templates, known company/ticker metadata, manual file paths, checksums, dedup rules, and prior source indexes.

LLM-assisted discovery may only:

- generate/refine search queries for a missing source class;
- classify candidate search results by likely source class;
- rank candidate URLs with confidence and reasons;
- propose peer candidates from text, subject to deterministic validation;
- recommend escalation when uncertain.

LLM must not fabricate URLs/dates, skip required sources, assign A-grade outside rules, create durable facts, summarize a call as the only source artifact, resolve conflicts, make investment decisions, or write DB rows.

Source grades at retrieval time:

- `A`: primary regulatory/audited/user-verified high-lineage sources, e.g. SEC filings from sec.gov, exchange filings, user-marked trusted model files.
- `B`: primary but management/promotional or provider/professional sources, e.g. earnings releases, official call transcripts, IR decks, Yahoo/FMP profile/financial/market data, sell-side reports, meeting notes.
- `C`: secondary/search-derived/unverified sources, e.g. news, blogs, search snippets, generic web fallback financials, uncertain transcript mirrors.

Manual files default conservatively: use explicit user hints when provided; otherwise mark provisional with review flags rather than guessing A-grade.

## 8. Failure modes and human escalation points

Failure modes:

- ticker/company ambiguity, ADR/local mismatch, private/non-US issuer;
- SEC search no results or wrong-company results;
- provider empty data, throttling, missing credentials;
- paywalled or inaccessible transcripts/reports;
- weak/truncated extraction;
- duplicate source variants;
- LLM discovery low confidence;
- manual file missing, unsupported, or unclassified.

Escalate when a required source is missing, company identity is ambiguous, paid credentials are needed, manual grade affects trust, non-US sources are required but unsupported, or LLM-discovered candidates cannot be verified. Monitor workflows may continue with partial failures; onboarding/memo should surface blockers before downstream modules assume completeness.

## 9. Versioned/staged coding plan

### v0: interfaces/skeleton/contracts only

- Add `src/retrieval/` models for request, policy, batch, source, failure, escalation.
- Define `retrieval_batch.v0` JSON schema and writer/reader.
- Implement workflow policy expansion only.
- Implement manual-file staging as the minimal useful source path.
- Add a stub `retrieve_sources(...)` that writes a valid manifest and never touches the vault DB.
- Document ingestion handoff and mark current `src/vault/onboard.py` / `src/vault/sec_import.py` as transitional mixed-boundary flows.

### v1: first useful implementation

- Add deterministic SEC adapter using current `sec_filings.py`, outputting A-grade files/manifest entries instead of ingesting text.
- Add company profile, financial metrics, and market data adapters from current yfinance tools as B-grade provider sources.
- Mark financial web-search fallback as C-grade with `fallback_search` quality flag.
- Stage manual files with checksums and provisional grade/classification.
- Expose CLI/API.
- Provide compatibility wrappers so harness skills can call retrieval while old names continue working.

### v2+ enhancements

- Official earnings release, call transcript, and investor presentation retrieval.
- News/monitoring query engine with Librarian query classes.
- Peer metric retrieval for explicit peer sets; candidate peer discovery remains reviewable.
- Incremental retrieval against prior source index.
- Non-US issuer adapters.
- Credentialed/paid data source support.
- Human review UI for ambiguous candidates.

## 10. Tests / validation gates

- Policy tests: each workflow expands to expected required/optional source classes.
- Grade-rule tests: SEC=A, provider snapshot=B, web fallback/news=C, manual unclassified=provisional.
- Manifest schema tests: all sources have paths, checksums, grades, methods, statuses.
- Boundary tests: retrieval does not import/call vault ingestion/store functions.
- Adapter tests with mocked SEC/yfinance/search responses.
- Integration gate: `retrieve_sources("XOM", "memo")` produces a manifest consumable by ingestion in a separate downstream test.
- Readiness gate: no missing required sources except explicit policy-permitted partials; no blocker escalations; all files exist and checksums match.

## 11. Open questions for Ted/orchestrator

1. Should `memo <ticker>` always include news and investor decks, or only when task/topic asks for them?
2. Should user-supplied manual files default to trusted A, or conservative per-source-class grades? Recommendation: conservative.
3. Are official earnings releases/IR decks A because primary issuer sources, or B because management/promotional? Recommendation: B unless regulatory-filed.
4. Are Yahoo/FMP snapshots acceptable B-grade for MVP?
5. Is v1 US-listed equities only, or must non-US filings be included immediately?
6. Should retrieval batches retain originals after ingestion, or only manifests/logs?
7. Who owns peer discovery: retrieval, lifecycle, or context assembly? Recommendation: retrieval fetches metrics for explicit peers; discovery remains reviewable.
8. Should retrieval stop on blocker escalation or write partial batches for orchestrator review?

## 12. Interactions with other modules

- **Vault ingestion/source registry:** consumes retrieval manifests, copies/extracts/registers documents, writes DB rows. Retrieval never writes DB rows.
- **Research lifecycle:** uses ingested documents and grades to build source index, questions, facts/claims, conflicts, freshness, and wiki updates.
- **Context assembly:** requests missing sources via retrieval rather than allowing memo agents to fetch ad hoc.
- **Memo/output harness:** consumes vetted context packages; charting and final artifacts stay here.
- **Existing equity tools:** retrieval wraps `sec_filings`, `company_profile`, `financials`, `market_data`; splits `earnings_calls` so raw source preservation belongs to retrieval while LLM summary is downstream/temporary; keeps `plot_price_history` out of retrieval.
- **Orchestrator:** sequences retrieval -> ingestion -> lifecycle/context -> output and decides Ted-facing escalations.
