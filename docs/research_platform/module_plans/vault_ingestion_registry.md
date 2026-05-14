# Deterministic Vault Ingestion + Source Registry Plan

Last updated: 2026-05-13  
Planner: `vault_registry_planner`  
Parent/orchestrator: `codex`

## 1. Module purpose

This module is AlphaSeeker's deterministic source-of-record intake layer. It receives already-normalized retrieval outputs or explicitly supplied manual files, stores raw and extracted source material in the vault, registers source metadata/company links, and returns stable document references for downstream modules. It should make sources durable, inspectable, citeable, deduplicated, and version-aware.

It does **not** decide what to retrieve, what matters, or what investment conclusion follows. Its contract is:

```text
normalized source payload
  -> deterministic validation/checksum/extraction
  -> canonical vault paths
  -> source registry rows + company links + retrieval/version events
  -> document refs for synthesis, lifecycle, source_index, and memo context
```

This follows the Librarian lesson that QC starts at ingestion: every future claim, metric, question, conflict, wiki section, and memo citation should trace to a registered source with lineage, source grade, timestamp, checksum, parser version, and canonical path.

## 2. Owned responsibilities

- Define the ingestion boundary and input contract for `NormalizedRetrievedSource` and manual-file wrappers.
- Validate required metadata: ticker/company link, source type, source grade or explicit missing-grade policy, content/path, retrieved timestamp, and safe paths.
- Copy/store raw source material under canonical vault document directories, never trusting user filenames as canonical identity.
- Extract readable text deterministically into `extracted.md` for supported types.
- Compute raw/content checksums and extracted-text checksums.
- Record source registry rows, company-document links, retrieval events, extraction status, parser version, and source-specific metadata.
- Implement exact duplicate detection and changed-source version behavior.
- Return machine-facing internal refs and human-safe citation labels/paths.
- Provide read/query APIs for downstream modules: list documents by ticker/type/grade/status, get document refs, list versions/events.

Existing code seeds these responsibilities in `src/vault/ingest.py`, `schema.py`, `store.py`, `paths.py`, and `wiki.render_source_index()`, but retrieval, ingestion, downstream extraction, and rendering are currently blurred in flows like `onboard.py`.

## 3. Explicit non-responsibilities

- Search, crawling, SEC query selection, news discovery, or deciding what to fetch.
- Choosing among candidate sources or deciding whether coverage is sufficient.
- Investment claims, thesis, valuation judgment, recommendations, or memo conclusions.
- LLM summarization, claim extraction, question generation, conflict resolution, or wiki-section writing.
- Arbitrate credibility between sources. The module records source grade; it does not decide in context which source wins.
- Cell-level Excel/model registry and model diffs. It may register model files as source documents, but model lineage belongs elsewhere.
- Polished human UX pages as product surface. It supplies registry views; the lifecycle/index module should own final `source_index.md` rendering, though current compatibility rendering may remain during migration.
- Temporary harness workspace cleanup.

## 4. Inputs / upstream ports

Primary upstream is the input/retrieval layer. It should pass normalized sources after retrieval/discovery decisions are complete. Suggested fields:

- `ticker` or `tickers`, with one primary ticker;
- `company_name` optional;
- `source_type` controlled string, e.g. `sec_filing`, `company_release`, `earnings_call`, `manual_file`, `derived_profile`, `market_data_snapshot`, `sell_side_report`, `news_article`, `web_article`, `meeting_note`;
- `source_grade` as `A`, `B`, or `C`, supplied by adapter/policy/human;
- `title`;
- `url` and optional `canonical_url`;
- `local_path`, `text`, or raw content payload;
- `retrieved_at`;
- `published_at` and/or `filed_at` when known;
- `content_type` / file extension / parser hint;
- `fetcher_name`, `fetcher_version`, `retrieval_run_id`;
- `metadata` JSON for SEC accession, form type, CIK, vendor symbol, author, etc.

Source-specific adapters such as SEC import should create these objects. Manual inbox ingestion should use the same port and should not automatically imply `A` grade unless Ted confirms that policy.

## 5. Outputs / downstream ports

Each ingest returns an `IngestResult` containing:

- status: `inserted`, `duplicate`, `new_version`, `updated_metadata`, `failed_validation`, `stored_with_extraction_failure`, etc.;
- document reference with internal join key plus human citation label;
- canonical raw/extracted/metadata paths;
- URL/date/type/grade fields;
- raw and extracted checksums;
- extraction status and warnings/errors;
- dedupe/version decision;
- retrieval event ID or run ID where available.

Downstream consumers:

- retrieval layer checks registry before refetching and records repeat observations;
- research-object/lifecycle modules cite `doc_id`/document refs when creating facts, metrics, claims, guidance, questions, answers, conflicts;
- LLM synthesis/context assembly reads extracted text and metadata to build source bundles;
- memo/output harness receives human-safe citation labels/paths, not bare private IDs as the main UX;
- source index renderer consumes registry views.

## 6. Persistent artifacts and schema needs

Current `documents` has `doc_id`, `source_type`, `title`, `path`, `url`, `published_at`, `ingested_at`, `source_grade`, `checksum`, and `metadata_json`; `document_companies` links docs to tickers. This is useful but insufficient for robust lineage.

Suggested v1 additions while preserving compatibility:

- `source_key` for logical identity: canonical URL, SEC accession, or normalized local-source key;
- `canonical_url`, `original_path`, `raw_path`, `extracted_path`, `metadata_path`;
- `retrieved_at`, `filed_at`, `last_seen_at`;
- `raw_checksum`, `extracted_checksum` while keeping old `checksum` as compatibility alias;
- `parser_name`, `parser_version`;
- `extraction_status`: `ok`, `empty`, `unsupported`, `failed`, `needs_ocr`, `partial`, `unknown`;
- `extraction_error`;
- `content_type`, `file_ext`;
- `source_grade_rationale`;
- `fetcher_name`, `fetcher_version`, `retrieval_run_id`;
- optional `supersedes_doc_id` or `current_version_of` for simple version chains.

Add `document_retrieval_events`:

- `retrieval_event_id`, `doc_id`, `source_key`, `retrieval_run_id`, URL/path, `retrieved_at`, fetcher name/version, checksum, status, metadata JSON, `created_at`.

Filesystem layout should remain inspectable:

```text
data/research_vault/documents/<doc-or-version-id>/
  original.<ext>
  extracted.md
  metadata.json
  extraction_error.txt   # optional
```

SQLite is authoritative; JSON mirrors are for audit/human inspection.

## 7. Deterministic code vs LLM decision boundary

Deterministic code owns validation, copying, checksums, canonical paths, source registry writes, company links, parser/extractor execution, extraction status, duplicate/version detection, and source grade recording.

LLMs may help upstream discovery or downstream synthesis, but should not run inside this module to summarize content, classify claims, choose sources, decide credibility, or write conclusions.

Source grade should be recorded at ingestion but assigned by explicit policy/adapters/human, not inferred ad hoc. Proposed meanings:

- `A`: regulatory/company-primary or explicit primary evidence;
- `B`: credible secondary/support/derived material;
- `C`: unverified, low-confidence, social, scraped, or LLM-discovered pending review.

Open policy decision: if missing, reject strictly or store as `C/unknown` with warning.

## 8. Failure modes and human escalation points

- Missing ticker, source type, grade under strict policy, content/path, or valid timestamp: reject with structured error.
- Unsafe paths, copy failure, SQLite failure, path collision, missing existing files: fail loudly; escalate if vault corruption is suspected.
- Same source key but changed checksum: create new version/event; flag for review if unexpected.
- Same checksum with conflicting title/date/type: reuse content but record origin/metadata conflict warning.
- Unsupported/corrupt/encrypted file, empty PDF text, bad encoding, oversized file: keep raw source if allowed, mark extraction `unsupported`/`failed`/`needs_ocr`/`partial`, prevent normal downstream use unless explicitly allowed.
- High-grade source with extraction failure: human/orchestrator escalation for OCR/manual conversion.
- Ambiguous manual source grade or conflicting source policy: record warning; do not arbitrate.
- Legacy migration missing raw path/checksum: mark `legacy_unknown` rather than deleting.

## 9. Versioned/staged coding plan

### v0: interfaces/contracts only

- Define dataclasses/protocol docs for `NormalizedRetrievedSource`, `IngestResult`, and `DocumentRef`.
- Define enums for source type, grade, extraction status, relevance.
- Wrap current `ingest_file()`/`ingest_text()` behind the new conceptual API without behavior churn.
- Document dedupe/source-key/checksum policy.
- Inventory callers: `sec_import.py`, `onboard.py`, `memo_flow.py`, `synthesis.py`, `wiki.py`, `extract.py`, `status.py`.
- No broad schema migration unless Ted approves. Existing tests should still pass.

### v1: first useful implementation

- Add schema migration for richer document fields and retrieval events while keeping `documents.path` populated.
- Implement primary `ingest_source()`; make file/text wrappers build normalized sources.
- Record parser version, extraction status, raw/extracted paths/checksums.
- Implement exact duplicate idempotency and simple same-source changed-content versioning.
- Update SEC adapter to pass richer metadata and `A` grade by policy.
- Revisit manual inbox default grade.
- Update source bundle builder to use document refs and skip/warn on failed extraction.
- Update source index renderer to consume registry view rows.

### v2+ enhancements

- Split logical sources from physical document versions.
- Add origin alias table for multiple URLs/paths.
- Add extraction metadata: page count, char count, language, table count.
- Add OCR fallback policy.
- Add robust HTML extraction stack.
- Add configurable source-type policy registry.
- Add changed-source diffs and review queue integration.
- Add model-file handoff to dedicated model registry.

## 10. Tests / validation gates

Unit tests:

- checksum stability;
- ingest stores raw/extracted/metadata and registry rows;
- exact duplicate is idempotent;
- same source key with changed content creates version/event;
- same checksum from different origin records event/alias;
- invalid/missing grade policy behaves as chosen;
- PDF/text/HTML extraction statuses are correct;
- empty PDF becomes `needs_ocr`/`empty`, not silent success;
- unsupported formats are rejected or registered consistently;
- ticker links normalize uppercase and preserve relevance;
- migration preserves legacy `path` compatibility.

Integration tests:

- mocked SEC filing -> ingestion -> registry -> source bundle;
- manual file -> ingestion -> source index view;
- derived profile/financials are `B` support evidence;
- synthesis skips/warns on failed extraction.

Pre-synthesis gate: every included source has ticker link, type, grade, retrieval timestamp, date or explicit unknown marker, existing extracted path, valid extraction status, checksum, parser version, and human citation label.

## 11. Open questions for Ted/orchestrator

1. Should missing source grade reject ingestion or default to `C/unknown` with review warning?
2. Exact source-grade policy: are management transcripts/guidance `A` because company-primary or `B` because talk-book risk?
3. Should manual uploads require explicit grade/source type instead of current automatic `A` behavior?
4. Is v1 one-table documents-plus-events enough, or should logical source/version split happen immediately?
5. Should `source_index.md` renderer move now to lifecycle/index, or stay in `vault/wiki.py` as compatibility?
6. Preferred human citation handle: wikilink, title/date label, URL, short checksum, stable public source number, or combination?
7. What is the OCR policy for high-grade PDFs with empty extraction?
8. Should raw sources be append-only forever for auditability?

## 12. Interactions with other modules

- **Input/retrieval** decides what to fetch and supplies normalized sources; ingestion stores and registers them.
- **Research-object/lifecycle** consumes document refs to create source-grounded claims, metrics, guidance, questions, answers, conflicts, freshness items, and review queues.
- **Source index/lifecycle renderer** owns polished `source_index.md`; ingestion owns registry data/views.
- **LLM synthesis/context assembly** consumes extracted paths and metadata, filters by status/grade/date, and performs synthesis outside ingestion.
- **Memo/output harness** consumes cited research state and source refs, not raw ingestion internals.
- **SEC/company profile/financial adapters** should become retrieval/source adapters that call the normalized ingestion port.
- **Model registry** receives registered model files from ingestion but owns cell-level model lineage/diffs.
- **Human/orchestrator review** handles ambiguous grades, changed-source surprises, extraction failures, and credibility decisions.
